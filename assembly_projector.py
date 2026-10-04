"""Independent rigid-piece/seam diagnostics; never subtract them before mFSM."""
import numpy as np


def rigid_fit(xyz,u):
    center=np.mean(xyz,axis=0); r=xyz-center
    design=np.zeros((len(xyz),3,6));design[:,:,:3]=np.eye(3)
    for i,(x,y,z) in enumerate(r):
        design[i,:,3:]=[[0,z,-y],[-z,0,x],[y,-x,0]]
    coefficients=np.linalg.lstsq(design.reshape(-1,6),u.ravel(),rcond=1e-12)[0]
    return (design@coefficients),coefficients


def assembly_diagnostics(xyz,u,pieces,seams):
    xyz=np.asarray(xyz,dtype=float);u=np.asarray(u,dtype=float);pieces=np.asarray(pieces)
    if xyz.shape!=u.shape or xyz.ndim!=2 or xyz.shape[1]!=3 or len(pieces)!=len(xyz):
        raise ValueError('Invalid diagnostic geometry/displacement shape')
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(u)): raise ValueError('Finite diagnostic data required')
    common,unused=rigid_fit(xyz,u); residual=u-common
    relative=np.zeros_like(u)
    for piece in set(pieces):
        mask=pieces==piece;relative[mask],unused=rigid_fit(xyz[mask],residual[mask])
    seam=[(residual[b]-residual[a]).tolist() for a,b in seams]
    total=np.linalg.norm(u)
    return dict(relative_piece_norm=float(np.linalg.norm(relative)),
                relative_piece_norm_percent=float(100*np.linalg.norm(relative)/max(total,1e-250)),
                seam_vectors=seam,seam_relative_norm=float(np.linalg.norm(seam)),
                metric='NODAL_TRANSLATION_NORM_DIAGNOSTIC_NOT_FAMILY_ENERGY')
