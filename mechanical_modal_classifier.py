# -*- coding: utf-8 -*-
"""Mechanical L/D/G classifier for observed Abaqus eigenmodes.

K0 energy is the primary metric. Assembly and seam quantities are independent
diagnostics and are never subtracted from the observed eigenmode before the
L/D/G projection.
"""
from __future__ import print_function
import itertools
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import abaqus_modal_validation as validation
from abaqus_resource_policy import apply_blas_thread_env


FAMILIES=('L','D','G','O')
FINAL_NAMES={'L':'LOCAL','D':'DISTORTIONAL','G':'GLOBAL'}


def default_settings(overrides=None):
    result=dict(
        dominance=0.90,
        max_mechanical_residual=0.05,
        max_harmonic_residual=0.05,
        assembly_warning_percent=15.0,
        assembly_unresolved_percent=25.0,
        metric_sensitivity_percentage_points=10.0,
        seam_warning_index=15.0,
        interpiece_warning_percent=15.0,
        basis_condition_max=1.0e10,
        cross_term_warning_percent=1.0)
    if overrides:
        for key,value in overrides.items():
            if key != 'gpu_module':
                result[key]=value
    return result


def _basis_for(provider, harmonic_n):
    if callable(provider):
        return provider(int(harmonic_n))
    if isinstance(provider, dict):
        if int(harmonic_n) in provider:
            return provider[int(harmonic_n)]
        if str(int(harmonic_n)) in provider:
            return provider[str(int(harmonic_n))]
        raise KeyError('No fcFSM basis for harmonic %s' % harmonic_n)
    if hasattr(provider,'project'):
        return provider
    raise TypeError('basis_provider must be callable, mapping, or basis object')


def reference_mode_vector(component, basis):
    """Map one harmonic's global Abaqus coefficients to CUFSM global DOF order."""
    if 'reference_vector' in component:
        q=np.asarray(component['reference_vector'],dtype=float).reshape(-1)
        if len(q) != basis.K0.shape[0] or not np.all(np.isfinite(q)):
            raise ValueError('reference_vector does not match basis K0')
        return q
    if 'U' not in component or 'UR' not in component:
        raise ValueError('Harmonic component requires reference_vector or U+UR')
    reference=getattr(basis,'reference',None)
    if reference is None:
        raise ValueError('U+UR mapping requires basis.reference')
    u=np.asarray(component['U'],dtype=float)
    ur=np.asarray(component['UR'],dtype=float)
    n=len(reference['nodes'])
    if u.shape != (n,3) or ur.shape != (n,3) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(ur)):
        raise ValueError('Harmonic U/UR do not match reference nodes')
    q=np.zeros(4*n,dtype=float)
    # CUFSM global order from assemble.m/trans.m:
    # [cross-section-X, longitudinal] for all nodes, then
    # [cross-section-Y, rotation-about-member-axis] for all nodes.
    q[0:2*n:2]=u[:,0]
    q[1:2*n:2]=u[:,2]
    q[2*n:4*n:2]=u[:,1]
    q[2*n+1:4*n:2]=ur[:,2]
    return q


def _rotation_scale_mm(basis):
    reference=getattr(basis,'reference',None)
    if reference is not None:
        thickness=float(reference['material']['thickness_mm'])
    else:
        thickness=float(getattr(basis,'metadata',{}).get('thickness_mm',1.0))
    if not math.isfinite(thickness) or thickness <= 0:
        raise ValueError('Positive shell thickness is required for kinematic metric')
    return thickness/math.sqrt(12.0)


def _kinematic_norm2(q,basis):
    q=np.asarray(q,dtype=float).reshape(-1)
    if len(q)%4:
        raise ValueError('fcFSM vector length must be four DOFs per node')
    n=len(q)//4
    weights=np.ones(len(q),dtype=float)
    weights[2*n+1:4*n:2]=_rotation_scale_mm(basis)**2
    return float(np.sum(weights*q*q))


def _energy_metric_vector(vector, stiffness):
    """Return coordinates whose Euclidean norm squared equals vector' K vector."""
    k=np.asarray(stiffness,dtype=float)
    v=np.asarray(vector,dtype=float).reshape(-1)
    eig,q=np.linalg.eigh(.5*(k+k.T))
    scale=max(float(np.max(np.abs(eig))),1e-250)
    keep=eig > 1e-10*scale
    if not np.any(keep):
        return np.zeros(0,dtype=float)
    return np.sqrt(eig[keep])*(q[:,keep].T@v)


def _diagnostic_value(diagnostics,name,default=0.0):
    try:
        value=float(diagnostics.get(name,default))
    except (TypeError,ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def classify_mode(mode_record, harmonic_result, basis_provider, diagnostics, settings):
    cfg=default_settings(settings)
    components=harmonic_result.get('components',{})
    if not components:
        raise ValueError('No retained longitudinal harmonic components')
    diag_energy={name:0.0 for name in FAMILIES}
    vector_norm={name:0.0 for name in FAMILIES}
    cross={a+':'+b:0.0 for a,b in itertools.combinations(FAMILIES,2)}
    input_energy=0.0
    metric_chunks={name:[] for name in FAMILIES}
    condition=[]
    basis_hashes={}

    for key in sorted(components,key=lambda x:int(x)):
        m=int(key)
        basis=_basis_for(basis_provider,m)
        q=reference_mode_vector(components[key],basis)
        k=np.asarray(basis.K0,dtype=float)
        if k.shape != (len(q),len(q)):
            raise ValueError('Basis K0 and harmonic vector size differ')
        report=getattr(basis,'condition_report',{})
        condition.append(float(report.get('energetic_condition',1.0)))
        basis_hashes[m]=getattr(basis,'definition_hash',None)
        if float(np.linalg.norm(q)) <= 1e-14:
            zero_metric=_energy_metric_vector(np.zeros_like(q),k)
            for name in FAMILIES:
                metric_chunks[name].append(zero_metric.copy())
            continue
        projected=basis.project(q)
        parts={name:np.asarray(projected['components'][name],dtype=float).reshape(-1)
               for name in ('L','D','G')}
        parts['O']=np.asarray(projected.get('residual',np.zeros_like(q)),dtype=float).reshape(-1)
        ein=.5*float(q@(k@q))
        if not math.isfinite(ein) or ein < -1e-12:
            raise ValueError('Harmonic has negative/nonfinite K0 energy')
        if ein <= 1e-250:
            zero_metric=_energy_metric_vector(np.zeros_like(q),k)
            for name in FAMILIES:
                metric_chunks[name].append(zero_metric.copy())
            continue
        input_energy += ein
        for name in FAMILIES:
            p=parts[name]
            diag_energy[name] += .5*float(p@(k@p))
            vector_norm[name] += _kinematic_norm2(p,basis)
            metric_chunks[name].append(_energy_metric_vector(p,k))
        for a,b in itertools.combinations(FAMILIES,2):
            cross[a+':'+b] += float(parts[a]@(k@parts[b]))

    diagonal_total=sum(max(0.0,x) for x in diag_energy.values())
    if diagonal_total <= 1e-250:
        raise ValueError('Mechanical decomposition has zero reconstructed energy')
    energy=[100.0*max(0.0,diag_energy[name])/diagonal_total for name in FAMILIES]
    vector_total=sum(vector_norm.values())
    if vector_total <= 1e-250:
        vector=[0.0]*4
    else:
        vector=[100.0*vector_norm[name]/vector_total for name in FAMILIES]
    sensitivity=max(abs(a-b) for a,b in zip(energy,vector))
    residual_percent=energy[3]
    cross_percent={name:100.0*value/diagonal_total for name,value in cross.items()}
    reconstructed=diagonal_total+sum(cross.values())
    closure=abs(reconstructed-input_energy)/max(abs(input_energy),1e-250)

    dominant=int(np.argmax(energy[:3]))
    preliminary=(FINAL_NAMES[FAMILIES[dominant]]
                 if energy[dominant] >= 100.0*float(cfg['dominance']) else 'MIXED')
    flags=[]
    unresolved=False
    if residual_percent > 100.0*float(cfg['max_mechanical_residual']):
        flags.append('HIGH_MECHANICAL_RESIDUAL'); unresolved=True
    harmonic_residual=float(harmonic_result.get('relative_residual',0.0) or 0.0)
    if harmonic_residual > float(cfg['max_harmonic_residual']):
        flags.append('HARMONIC_FIT_POOR'); unresolved=True
    if sensitivity > float(cfg['metric_sensitivity_percentage_points']):
        flags.append('METRIC_SENSITIVE')
    assembly=_diagnostic_value(diagnostics,'assembly_percent')
    if assembly > float(cfg['assembly_warning_percent']):
        flags.append('HIGH_ASSEMBLY')
    if assembly > float(cfg['assembly_unresolved_percent']):
        unresolved=True
    seam=max(_diagnostic_value(diagnostics,'normal_opening_index'),
             _diagnostic_value(diagnostics,'transverse_slip_index'),
             _diagnostic_value(diagnostics,'longitudinal_slip_index'))
    if seam > float(cfg['seam_warning_index']):
        flags.append('HIGH_SEAM_RELATIVE_MOTION')
    interaction_raw=diagnostics.get('interpiece_interaction_percent',None)
    interaction=None
    if interaction_raw is not None:
        try:
            candidate=float(interaction_raw)
            if math.isfinite(candidate):
                interaction=candidate
        except (TypeError,ValueError):
            interaction=None
    if interaction is not None and interaction > float(cfg['interpiece_warning_percent']):
        flags.append('INTERPIECE_INTERACTION_HIGH')
    max_condition=max(condition) if condition else 1.0
    if max_condition > float(cfg['basis_condition_max']):
        flags.append('BASIS_ILL_CONDITIONED'); unresolved=True
    if max([abs(x) for x in cross_percent.values()] or [0.0]) > float(cfg['cross_term_warning_percent']):
        flags.append('BASIS_CROSS_TERMS_HIGH'); unresolved=True
    family='UNRESOLVED' if unresolved else preliminary
    quality_state='UNRESOLVED' if unresolved else ('WARNING' if flags else 'OK')

    metric_components={}
    for name in FAMILIES:
        metric_components[name]=np.concatenate(metric_chunks[name]) if metric_chunks[name] else np.zeros(0)
    result=dict(mode=mode_record.get('mode'),eigenvalue=mode_record.get('eigenvalue'),
                stress_MPa=mode_record.get('stress_MPa'),
                family=family,preliminary_family=preliminary,
                energy_percent=energy,vector_percent=vector,
                L_energy_percent=energy[0],D_energy_percent=energy[1],
                G_energy_percent=energy[2],O_energy_percent=energy[3],
                L_vector_percent=vector[0],D_vector_percent=vector[1],
                G_vector_percent=vector[2],O_vector_percent=vector[3],
                mechanical_residual_percent=residual_percent,
                harmonic_residual=harmonic_residual,
                metric_sensitivity_pp=sensitivity,
                assembly_percent=assembly,
                seam_normal_opening_index=_diagnostic_value(diagnostics,'normal_opening_index'),
                seam_transverse_slip_index=_diagnostic_value(diagnostics,'transverse_slip_index'),
                seam_longitudinal_slip_index=_diagnostic_value(diagnostics,'longitudinal_slip_index'),
                interpiece_interaction_percent=interaction,
                cross_terms_percent=cross_percent,
                energy_closure_relative=closure,
                basis_condition=max_condition,basis_hashes=basis_hashes,
                dominant_m=harmonic_result.get('dominant_m'),
                half_wavelength_mm=harmonic_result.get('half_wavelength_mm'),
                quality_state=quality_state,flags=flags,
                thresholds=cfg,numeric_backend='cpu')
    result['_metric_components']=metric_components
    return result


def classify_eigenspace(cluster_rows, component_columns, settings):
    cfg=default_settings(settings)
    bounds=validation.mechanical_component_bounds(
        component_columns,dominance=float(cfg['dominance']))
    flags=[]
    isolated=all(row.get('spectral_isolated',row.get('isolated',True)) is True
                 for row in cluster_rows)
    residual_high=bounds['O_max_percent'] > 100.0*float(cfg['max_mechanical_residual'])
    if not isolated:
        flags.append('EIGENSPACE_NOT_ISOLATED')
    if residual_high:
        flags.append('HIGH_MECHANICAL_RESIDUAL')
    stable=bounds['stable_family']
    if not isolated or residual_high:
        family='UNRESOLVED'
    else:
        family=FINAL_NAMES.get(stable,'MIXED')
    result=dict(bounds)
    result.update(family=family,flags=flags,spectral_isolated=isolated,
                  modes=[row.get('mode') for row in cluster_rows])
    return result


def validate_gpu_backend(xp,tolerance=1e-12):
    try:
        energy=np.array([[99.,1.,0.,0.],[60.,30.,10.,0.]],dtype=float)
        vector=np.array([[50.,50.,0.,0.],[59.,31.,10.,0.]],dtype=float)
        cpu=np.max(np.abs(energy-vector),axis=1)
        gpu=xp.asnumpy(xp.max(xp.abs(xp.asarray(energy)-xp.asarray(vector)),axis=1))
        return bool(np.allclose(cpu,np.asarray(gpu),rtol=tolerance,atol=tolerance))
    except Exception:
        return False


def _gpu_module(settings):
    if settings.get('gpu_module') is not None:
        return settings['gpu_module']
    try:
        import cupy
        return cupy
    except Exception:
        return None


def classify_modes_parallel(records,settings,resource_plan,progress=None):
    records=list(records)
    layout=dict(resource_plan.get('worker_layout',{}))
    workers=max(1,int(layout.get('processes',1)))
    apply_blas_thread_env(dict(blas_threads=max(1,int(layout.get('blas_threads',1)))))
    def one(record):
        return classify_mode(record['mode_record'],record['harmonic_result'],
                             record['basis_provider'],record.get('diagnostics',{}),settings)
    total=len(records)
    rows=[]
    if workers == 1 or total < 2:
        for i,record in enumerate(records,1):
            rows.append(one(record))
            if progress is not None:
                progress(i,total)
    else:
        # executor.map preserves record order; enumerate its completed iterator
        # so one coordinator, not worker threads, emits aggregate progress.
        with ThreadPoolExecutor(max_workers=min(workers,total)) as pool:
            for i,row in enumerate(pool.map(one,records),1):
                rows.append(row)
                if progress is not None:
                    progress(i,total)

    if int(resource_plan.get('gpus',0) or 0) > 0:
        xp=_gpu_module(settings)
        if xp is not None and validate_gpu_backend(xp):
            energy=np.asarray([row['energy_percent'] for row in rows],dtype=float)
            vector=np.asarray([row['vector_percent'] for row in rows],dtype=float)
            threshold=float(default_settings(settings)['metric_sensitivity_percentage_points'])
            cpu=np.max(np.abs(energy-vector),axis=1) > threshold
            try:
                gpu=xp.asnumpy(xp.max(xp.abs(xp.asarray(energy)-xp.asarray(vector)),axis=1) > threshold)
                if np.array_equal(cpu,np.asarray(gpu,dtype=bool)):
                    for row in rows:
                        row['numeric_backend']='gpu_qc_validated'
                else:
                    for row in rows:
                        row['numeric_backend']='cpu_fallback_gpu_equivalence_failed'
            except Exception:
                for row in rows:
                    row['numeric_backend']='cpu_fallback_gpu_runtime_error'
        else:
            for row in rows:
                row['numeric_backend']='cpu_fallback_gpu_unavailable_or_unvalidated'
    return rows
