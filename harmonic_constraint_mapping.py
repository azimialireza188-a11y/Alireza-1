"""Sparse initial C @ R for coupled longitudinal CPT coordinates.

R is the declared six-DOF reconstruction, including its unvalidated facet drill
approximation and optional independent V0. Only raw DOFs referenced by C are
evaluated. This does not generate Q, validate contact or activate classification.
"""
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from prismatic_mfsm_operators import harmonic_design,transverse_rotation_map


def _locations(reference,keys):
    n=len(reference['xy']);stations=reference['stations'];grid=reference['node_keys']
    if len(grid)!=len(stations) or any(len(row)!=n for row in grid):
        raise ValueError('Complete reference node grid required')
    lookup={tuple(key):(z,i) for z,row in enumerate(grid) for i,key in enumerate(row)}
    if len(lookup)!=n*len(stations):raise ValueError('Unique reference nodes required')
    locations=[]
    for key in keys:
        if (len(key)!=3 or isinstance(key[2],bool) or
                not isinstance(key[2],(int,np.integer)) or not 1<=key[2]<=6 or
                tuple(key[:2]) not in lookup):
            raise ValueError('Known reference node and raw DOF 1..6 required')
        locations.append((*lookup[tuple(key[:2])],int(key[2])-1))
    return locations


def selected_reconstruction(reference,terms,raw_keys,include_constant_warping=False):
    """Evaluate selected raw rows; coefficients ordered H,node,UX/UZ/UY/URZ,V0."""
    if type(include_constant_warping) is not bool:raise ValueError('Explicit V0 selection required')
    r=reference;n=len(r['xy']);keys=list(raw_keys);locations=_locations(r,keys)
    design=harmonic_design(r['stations'],terms,r['length'],r['origin']);h=len(terms)
    rotation=transverse_rotation_map(r['xy'],r['edges'],terms,r['length'])
    constant_rotation=(transverse_rotation_map(r['xy'],r['edges'],[1],r['length'])[:,1::4]
                       if include_constant_warping else None)
    size=4*n*h;rows=[];cols=[];values=[]
    canonical={0:(0,'sin'),1:(2,'sin'),2:(1,'cos'),5:(3,'sin')}
    def add(row,col,value):
        if value!=0:rows.append(row);cols.append(col);values.append(value)
    for row,(z,i,dof) in enumerate(locations):
        for term in range(h):
            if dof in canonical:
                d,parity=canonical[dof]
                add(row,4*n*term+4*i+d,design[parity][z,term])
            else:
                index=2*n*term+2*i+dof-3
                start,end=rotation.indptr[index:index+2]
                for col,value in zip(rotation.indices[start:end],rotation.data[start:end]):
                    add(row,int(col),design['cos'][z,term]*value)
        if include_constant_warping:
            if dof==2:add(row,size+i,1.)
            elif dof in (3,4):
                index=2*i+dof-3;start,end=constant_rotation.indptr[index:index+2]
                for col,value in zip(constant_rotation.indices[start:end],constant_rotation.data[start:end]):
                    add(row,size+int(col),value)
    out=coo_matrix((values,(rows,cols)),shape=(len(keys),size+(n if include_constant_warping else 0)),dtype=np.float64).tocsr()
    out.eliminate_zeros();return out


def compile_harmonic_constraints(constraints,raw_keys,reference,terms,include_constant_warping=True):
    """Retain all constraint rows and cross-harmonic terms; identify zero rows."""
    c=csr_matrix(constraints,dtype=np.float64);keys=list(raw_keys)
    if c.shape[1]!=len(keys) or len(set(map(tuple,keys)))!=len(keys) or not np.all(np.isfinite(c.data)):
        raise ValueError('Finite constraints and matching unique raw keys required')
    _locations(reference,keys)  # Unknown unused keys also invalidate provenance.
    c.eliminate_zeros();used=np.unique(c.indices)
    evaluation=selected_reconstruction(reference,terms,[keys[i] for i in used],include_constant_warping)
    out=(c[:,used]@evaluation).tocsr();out.eliminate_zeros()
    metadata=dict(algorithm='INITIAL_CPT_HARMONIC_CONSTRAINT_PULLBACK_V1',
        raw_constraint_shape=list(c.shape),harmonic_constraint_shape=list(out.shape),
        evaluated_raw_dofs=len(used),evaluation_nonzeros=evaluation.nnz,
        constraint_nonzeros=out.nnz,sparse_storage_bytes=out.data.nbytes+out.indices.nbytes+out.indptr.nbytes,
        zero_rows=np.flatnonzero(np.diff(out.indptr)==0).tolist(),
        cross_harmonic_constraints_retained=True,include_constant_warping=include_constant_warping,
        dense_raw_reconstruction_allocated=False,dense_nullspace_allocated=False,
        rank_verified=False,active_contact_verified=False,scientifically_eligible=False,
        limitation='Initial C composed with declared CPT reconstruction; not an admissible basis or physical tangent.')
    return out,metadata
