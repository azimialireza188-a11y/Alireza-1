"""Energy-ratio nullspace kernel: 2019 (125)/(134), built-up 2023 (35).

Search restrictions must be supplied explicitly with source provenance. This
module does not infer GBT equilibrium/warping constraints from visual labels.
"""
from dataclasses import asdict
import numpy as np
from scipy.linalg import eigh, null_space, orth
from mfsm_model import BasisPack, content_hash, matrix, normalized_columns, ALGORITHM_VERSION


def basis_cache_key(operators, space_definitions):
    definitions={}; arrays=[operators.system]
    for name in sorted(operators.components):
        arrays.append(operators.components[name])
    for name,spec in sorted(space_definitions.items()):
        definitions[name]={k:v for k,v in spec.items() if k not in ('search_basis','constraints')}
        for key in ('search_basis','constraints'):
            if key in spec: arrays.append(spec[key])
    return content_hash({'algorithm':ALGORITHM_VERSION,'source':asdict(operators.source),
                         'operator_metadata':operators.metadata,'spaces':definitions,
                         'component_names':sorted(operators.components)},arrays)


def build_mfsm_basis(operators, source_manifest, policy=None, tolerance=1e-10):
    spaces={}; ratios={}; n=len(operators.system)
    for name,spec in source_manifest.items():
        if not spec.get('equations') or 'search_basis' not in spec:
            raise ValueError('Source equations and explicit compatible search basis required')
        h=matrix(spec['search_basis'],n)
        h=orth(normalized_columns(h),rcond=tolerance)
        if not h.shape[1]:
            spaces[name]=np.empty((n,0)); ratios[name]=[]; continue
        if 'constraints' in spec:
            c=matrix(spec['constraints'])
            if c.shape[1]!=n: raise ValueError('Search constraint DOF mismatch')
            h=h@null_space(c@h,rcond=tolerance)
        if not h.shape[1]:
            spaces[name]=np.empty((n,0)); ratios[name]=[]; continue
        names=spec.get('zero_components',[])
        if not names or any(k not in operators.components for k in names):
            raise ValueError('Documented strain components required')
        km=sum(operators.components[k] for k in names)
        b=h.T@operators.system@h; a=h.T@km@h
        # Explicit energetic-null reduction; never introduce pins or gap ties.
        vals,vectors=eigh((b+b.T)/2)
        scale=max(float(np.max(np.abs(vals))),1e-250)
        if np.min(vals)<-tolerance*scale: raise ValueError('Indefinite system metric')
        keep=vals>tolerance*scale
        if not np.any(keep): raise ValueError('Search space has no energetic DOFs')
        w=h@(vectors[:,keep]/np.sqrt(vals[keep]))
        reduced=w.T@km@w
        rho,theta=eigh((reduced+reduced.T)/2)
        ratio_scale=max(float(np.max(np.abs(rho))),1e-250)
        if np.min(rho)<-tolerance*ratio_scale: raise ValueError('Indefinite strain operator')
        zero=np.abs(rho)<=tolerance*max(1.,ratio_scale)
        spaces[name]=w@theta[:,zero]; ratios[name]=rho.tolist()
    key=basis_cache_key(operators,source_manifest)
    meta=dict(operators.metadata,algorithm=ALGORITHM_VERSION,source=asdict(operators.source),
              energy_ratios=ratios,source_method='MFSM_SUPPLIED_OPERATOR_SEARCH_SPACES',
              scientific_validation='PENDING_PHYSICAL_BENCHMARKS',ratio_tolerance=tolerance)
    return BasisPack(operators.system,spaces,meta,key)


def cached_mfsm_basis(operators,source_manifest,directory,policy=None):
    import json,os,tempfile
    key=basis_cache_key(operators,source_manifest)
    os.makedirs(directory,exist_ok=True);path=os.path.join(directory,key+'.npz')
    if os.path.exists(path):
        try:
            with np.load(path,allow_pickle=False) as data:
                metadata=json.loads(str(data['metadata'].item()))
                names=json.loads(str(data['names'].item()))
                metric=data['metric']; spaces={name:data['space_'+name] for name in names}
                checksum=content_hash(metadata,[metric]+[spaces[k] for k in names])
                if checksum!=str(data['checksum'].item()): raise ValueError('Cache checksum mismatch')
                if metadata.get('cache_key')!=key: raise ValueError('Cache identity mismatch')
                pack=BasisPack(metric,spaces,metadata,key)
                return pack,dict(hit=True,path=path,key=key)
        except (OSError,EOFError,KeyError,ValueError) as exc:
            raise ValueError('Invalid mFSM cache: '+str(exc)) from exc
    pack=build_mfsm_basis(operators,source_manifest,policy)
    metadata=dict(pack.metadata,cache_key=key); names=sorted(pack.spaces)
    arrays=dict(metric=pack.metric,metadata=json.dumps(metadata,sort_keys=True),names=json.dumps(names),
                checksum=content_hash(metadata,[pack.metric]+[pack.spaces[k] for k in names]))
    arrays.update({'space_'+k:pack.spaces[k] for k in names})
    fd,temp=tempfile.mkstemp(prefix='mfsm-',suffix='.npz',dir=directory)
    try:
        with os.fdopen(fd,'wb') as f: np.savez(f,**arrays)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    pack.metadata=metadata
    return pack,dict(hit=False,path=path,key=key)
