# -*- coding: utf-8 -*-
"""Independent validation harness for the Stage-A fcFSM-style classifier.

This module has two deliberately different levels:
1. analytic synthetic checks that can run in ordinary Python;
2. comparison against an externally generated CUFSM/fcFSM JSON reference.

Passing level 1 is an implementation sanity check. It is never reported as a
substitute for a real CUFSM 5.70 reference benchmark.
"""
from __future__ import print_function
import copy
import hashlib
import json
import math
import os
import numpy as np

from fcfsm_reference_basis import force_family_basis


_REQUIRED_REFERENCE_PATHS=(
    ('source','program'),('source','version'),('source','method'),
    ('geometry','name'),('geometry','nodes'),('geometry','thickness_mm'),
    ('material','E_MPa'),('material','nu'),
    ('boundary_conditions','longitudinal'),
    ('harmonic','m'),('harmonic','length_mm'),
)


def _load_json(value):
    if isinstance(value,(str,bytes,os.PathLike)):
        with open(value,'r',encoding='utf-8-sig') as stream:
            return json.load(stream)
    if not isinstance(value,dict):
        raise TypeError('Benchmark input must be a JSON path or dictionary')
    return copy.deepcopy(value)


def _lookup(record,path):
    value=record
    for key in path:
        if not isinstance(value,dict) or key not in value:
            raise ValueError('Benchmark fixture missing '+'.'.join(path))
        value=value[key]
    return value


def _finite_number(value,positive=False):
    if isinstance(value,bool):
        return False
    try:
        number=float(value)
    except (TypeError,ValueError):
        return False
    return math.isfinite(number) and (number>0 if positive else True)


def _basis_matrix(value,name):
    a=np.asarray(value,dtype=float)
    if a.ndim!=2 or a.shape[0]<1 or a.shape[1]<1 or not np.all(np.isfinite(a)):
        raise ValueError('Invalid %s family basis matrix' % name)
    return a


def validate_reference_fixture(reference_json):
    reference=_load_json(reference_json)
    if int(reference.get('schema_version',0)) != 1:
        raise ValueError('Unsupported benchmark schema_version; expected 1')
    for path in _REQUIRED_REFERENCE_PATHS:
        _lookup(reference,path)
    if 'CUFSM' not in str(reference['source']['program']).upper():
        raise ValueError('Reference source.program must identify CUFSM')
    if not str(reference['source']['version']).strip():
        raise ValueError('Reference source.version is required')
    if not str(reference['source']['method']).strip():
        raise ValueError('Reference source.method is required')
    if not _finite_number(reference['geometry']['thickness_mm'],True):
        raise ValueError('geometry.thickness_mm must be positive finite')
    nodes=np.asarray(reference['geometry']['nodes'],dtype=float)
    if nodes.ndim!=2 or nodes.shape[0]<2 or nodes.shape[1]!=2 or not np.all(np.isfinite(nodes)):
        raise ValueError('geometry.nodes must be a finite Nx2 cross-section array')
    if not _finite_number(reference['material']['E_MPa'],True):
        raise ValueError('material.E_MPa must be positive finite')
    if not _finite_number(reference['material']['nu']):
        raise ValueError('material.nu must be finite')
    if str(reference['boundary_conditions']['longitudinal']).upper()!='S-S':
        raise ValueError('Reference longitudinal boundary condition must be S-S for Stage A')
    try:
        harmonic=int(reference['harmonic']['m'])
    except (TypeError,ValueError):
        raise ValueError('harmonic.m must be a positive integer')
    if harmonic<1 or float(harmonic)!=float(reference['harmonic']['m']):
        raise ValueError('harmonic.m must be a positive integer')
    if not _finite_number(reference['harmonic']['length_mm'],True):
        raise ValueError('harmonic.length_mm must be positive finite')
    families=reference.get('families')
    if not isinstance(families,dict):
        raise ValueError('Reference fixture missing families')
    rows=None
    for family in ('L','D','G'):
        if family not in families:
            raise ValueError('Reference fixture missing families.'+family)
        record=families[family]
        if not _finite_number(record.get('share_percent')):
            raise ValueError('families.%s.share_percent must be finite' % family)
        share=float(record['share_percent'])
        if share<0 or share>100:
            raise ValueError('families.%s.share_percent must be in [0,100]' % family)
        basis=_basis_matrix(record.get('basis'),family)
        if rows is None:
            rows=basis.shape[0]
        elif basis.shape[0]!=rows:
            raise ValueError('All reference family bases must use the same DOF space')
    reference['harmonic']['m']=harmonic
    return reference


def validate_classifier_fixture(classifier_json):
    result=_load_json(classifier_json)
    for path in (
        ('geometry','name'),('geometry','nodes'),('geometry','thickness_mm'),
        ('material','E_MPa'),('material','nu'),
        ('boundary_conditions','longitudinal'),('harmonic','m'),('harmonic','length_mm')):
        _lookup(result,path)
    families=result.get('families')
    if not isinstance(families,dict):
        raise ValueError('Classifier fixture missing families')
    rows=None
    for family in ('L','D','G'):
        if family not in families or not _finite_number(families[family].get('share_percent')):
            raise ValueError('Classifier fixture missing finite families.%s.share_percent' % family)
        basis=_basis_matrix(families[family].get('basis'),family)
        if rows is None:
            rows=basis.shape[0]
        elif basis.shape[0]!=rows:
            raise ValueError('All classifier family bases must use the same DOF space')
    return result


def _compatibility_payload(record):
    return dict(
        geometry=record['geometry'],
        material=record['material'],
        boundary_conditions=record['boundary_conditions'],
        harmonic=record['harmonic'])


def _compatibility_hash(record):
    payload=json.dumps(_compatibility_payload(record),sort_keys=True,separators=(',',':'),
                       allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def principal_subspace_agreement(reference_basis,classifier_basis):
    """Return principal-angle agreement; sign/scale/internal basis rotation invariant."""
    a=_basis_matrix(reference_basis,'reference')
    b=_basis_matrix(classifier_basis,'classifier')
    if a.shape[0]!=b.shape[0]:
        raise ValueError('Family basis DOF dimensions differ')
    qa=np.linalg.qr(a,mode='reduced')[0]
    qb=np.linalg.qr(b,mode='reduced')[0]
    # Drop numerical dependent columns using SVD because reduced QR alone keeps
    # columns even if the input basis is rank deficient.
    ua,sa,unused=np.linalg.svd(a,full_matrices=False)
    ub,sb,unused=np.linalg.svd(b,full_matrices=False)
    ta=max(float(sa[0]) if len(sa) else 0.,1e-250)*1e-10
    tb=max(float(sb[0]) if len(sb) else 0.,1e-250)*1e-10
    qa=ua[:,sa>ta]; qb=ub[:,sb>tb]
    if qa.shape[1]!=qb.shape[1] or qa.shape[1]==0:
        return dict(reference_dimension=int(qa.shape[1]),
                    classifier_dimension=int(qb.shape[1]),
                    minimum_cosine_squared=0.0,
                    cosine_squared=[],
                    dimension_match=False)
    singular=np.linalg.svd(qa.T@qb,compute_uv=False)
    cos2=np.clip(singular,0.,1.)**2
    return dict(reference_dimension=int(qa.shape[1]),
                classifier_dimension=int(qb.shape[1]),
                minimum_cosine_squared=float(np.min(cos2)),
                cosine_squared=[float(x) for x in cos2],
                dimension_match=True)


def compare_with_cufsm_reference(reference_json,classifier_json,
                                 share_tolerance_pp=1.0,
                                 minimum_cosine_squared=.99):
    reference=validate_reference_fixture(reference_json)
    classifier=validate_classifier_fixture(classifier_json)
    ref_hash=_compatibility_hash(reference)
    cls_hash=_compatibility_hash(classifier)
    if ref_hash!=cls_hash:
        mismatches=[]
        for key in ('geometry','material','boundary_conditions','harmonic'):
            a=json.dumps(reference[key],sort_keys=True,separators=(',',':'))
            b=json.dumps(classifier[key],sort_keys=True,separators=(',',':'))
            if a!=b:
                mismatches.append(key)
        raise ValueError('CUFSM/classifier benchmark metadata incompatible: '+', '.join(mismatches))
    if not _finite_number(share_tolerance_pp) or float(share_tolerance_pp)<0:
        raise ValueError('share_tolerance_pp must be finite and nonnegative')
    if not _finite_number(minimum_cosine_squared) or not 0<float(minimum_cosine_squared)<=1:
        raise ValueError('minimum_cosine_squared must be in (0,1]')
    families={}
    passed=True
    for family in ('L','D','G'):
        ref=reference['families'][family]
        got=classifier['families'][family]
        diff=abs(float(got['share_percent'])-float(ref['share_percent']))
        agreement=principal_subspace_agreement(ref['basis'],got['basis'])
        ok=(diff<=float(share_tolerance_pp) and agreement['dimension_match'] and
            agreement['minimum_cosine_squared']>=float(minimum_cosine_squared))
        families[family]=dict(
            reference_share_percent=float(ref['share_percent']),
            classifier_share_percent=float(got['share_percent']),
            share_difference_pp=float(diff),
            minimum_cosine_squared=agreement['minimum_cosine_squared'],
            cosine_squared=agreement['cosine_squared'],
            reference_dimension=agreement['reference_dimension'],
            classifier_dimension=agreement['classifier_dimension'],
            passed=bool(ok))
        passed=passed and ok
    return dict(
        schema_version=1,
        compatible=True,
        passed=bool(passed),
        reference_source=reference['source'],
        classifier_source=classifier.get('source'),
        compatibility_hash=ref_hash,
        geometry=reference['geometry'],
        material=reference['material'],
        boundary_conditions=reference['boundary_conditions'],
        harmonic=reference['harmonic'],
        thresholds=dict(share_tolerance_pp=float(share_tolerance_pp),
                        minimum_cosine_squared=float(minimum_cosine_squared)),
        families=families,
        scope=('Independent family-share and subspace comparison. A passing '
               'synthetic test alone is not four-piece Abaqus validation.'))


def run_synthetic_benchmarks():
    """Analytic L/D/G sanity tests; no external CUFSM result is claimed."""
    k=np.eye(4)
    j=np.eye(4)[:,:3]
    equilibrium=np.array([[1.,0.,0.]])
    basis=force_family_basis(
        k,j,equilibrium,definition_hash='analytic_axes',
        metadata=dict(source='analytic unit axes, not CUFSM'))
    probes=dict(pure_L=np.array([0.,0.,0.,1.]),
                pure_D=np.array([0.,1.,0.,0.]),
                pure_G=np.array([1.,0.,0.,0.]))
    cases={}
    passed=True
    for name,vector in probes.items():
        projected=basis.project(vector)
        values=projected['energy_percent']
        record=dict(L_percent=float(values[0]),D_percent=float(values[1]),
                    G_percent=float(values[2]),
                    residual_percent=float(projected['residual_energy_percent']))
        cases[name]=record
        expected={'pure_L':'L_percent','pure_D':'D_percent','pure_G':'G_percent'}[name]
        passed=passed and record[expected]>99.999999
    matrices={'L':basis.C_L,'D':basis.C_D,'G':basis.C_G}
    cross=[]
    for a,b in (('L','D'),('L','G'),('D','G')):
        if matrices[a].size and matrices[b].size:
            gram=matrices[a].T@basis.K0@matrices[b]
            cross.append(float(np.max(np.abs(gram)))*100.)
    maximum=max(cross or [0.])
    passed=passed and maximum<1e-8
    return dict(
        passed=bool(passed),
        cases=cases,
        maximum_cross_family_energy_percent=maximum,
        scope='Analytic synthetic implementation sanity check; not an external CUFSM benchmark')


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',help='Externally generated CUFSM/fcFSM JSON')
    p.add_argument('--classifier',help='Stage-A classifier JSON with matching benchmark basis')
    p.add_argument('--output')
    p.add_argument('--share-tolerance-pp',type=float,default=1.0)
    p.add_argument('--minimum-cosine-squared',type=float,default=.99)
    args=p.parse_args(argv)
    result=dict(synthetic=run_synthetic_benchmarks(),external_status='NOT_REQUESTED')
    if args.reference or args.classifier:
        if not (args.reference and args.classifier):
            p.error('--reference and --classifier must be supplied together')
        result['external']=compare_with_cufsm_reference(
            args.reference,args.classifier,args.share_tolerance_pp,args.minimum_cosine_squared)
        result['external_status']='PASSED' if result['external']['passed'] else 'FAILED'
    if args.output:
        with open(args.output,'w') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result['synthetic']['passed'] or result.get('external_status')=='FAILED':
        raise SystemExit(1)
    return result


if __name__=='__main__':
    main()
