"""DOF-aware harmonic fitting and explicit U/UR mapping; no invented rotations."""
import numpy as np
from mfsm_model import check_constraints_per_mode, ModeBatch, matrix


def fit_harmonics(stations, fields, terms, length, parities, origin=0.):
    z=np.asarray(stations,dtype=float); y=matrix(fields)
    terms=np.asarray(terms,dtype=int)
    if (z.ndim!=1 or len(z)!=len(y) or not np.all(np.isfinite(z)) or length<=0
            or len(parities)!=y.shape[1] or not len(terms) or np.any(terms<1)
            or len(set(terms))!=len(terms)):
        raise ValueError('Invalid harmonic data/DOF parity')
    coefficients=[]; fitted=[]
    phase=(z[:,None]-origin)*terms[None,:]*np.pi/length
    for j,parity in enumerate(parities):
        if parity not in ('sin','cos'): raise ValueError('Explicit sin/cos parity required for every DOF')
        b=np.sin(phase) if parity=='sin' else np.cos(phase)
        c,unused,rank,s=np.linalg.lstsq(b,y[:,j],rcond=1e-12)
        if rank!=len(terms) or s[-1]<1e-12*np.sqrt(len(z)):
            raise ValueError('Harmonic basis is unresolved/aliased')
        coefficients.append(c); fitted.append(b@c)
    residual=np.linalg.norm(y-np.column_stack(fitted))/max(np.linalg.norm(y),1e-250)
    return np.column_stack(coefficients),float(residual)


def map_modes(batch, mapping, constraints=None, tolerance=1e-8):
    mapping=matrix(mapping)
    if mapping.shape[1]!=batch.vectors.shape[0]: raise ValueError('Mapping DOF mismatch')
    mapped=mapping@batch.vectors
    if constraints is not None:
        check_constraints_per_mode(mapped,constraints,tolerance)
    return ModeBatch(mapped,batch.ids)


def extract_mode_batches(frames, keys, batch_size):
    from abaqus_dsm_modal_audit import read_mapped_mode
    if batch_size<1: raise ValueError('Positive batch size required')
    vectors=[]; ids=[]
    for mode_id,frame in frames:
        vectors.append(read_mapped_mode(frame,keys)); ids.append(mode_id)
        if len(ids)==batch_size:
            yield ModeBatch(np.column_stack(vectors),tuple(ids)); vectors=[]; ids=[]
    if ids: yield ModeBatch(np.column_stack(vectors),tuple(ids))
