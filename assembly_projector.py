"""Independent rigid-piece/seam diagnostics; never subtract them before mFSM."""
import numpy as np


def rigid_fit(xyz,u):
    center=np.mean(xyz,axis=0); r=xyz-center
    design=np.zeros((len(xyz),3,6));design[:,:,:3]=np.eye(3)
    for i,(x,y,z) in enumerate(r):
        design[i,:,3:]=[[0,z,-y],[-z,0,x],[y,-x,0]]
    coefficients=np.linalg.lstsq(design.reshape(-1,6),u.ravel(),rcond=1e-12)[0]
    return (design@coefficients),coefficients


def assembly_diagnostics(xyz,u,pieces,seams,seam_axes=None):
    xyz=np.asarray(xyz,dtype=float);u=np.asarray(u,dtype=float);pieces=np.asarray(pieces)
    if xyz.shape!=u.shape or xyz.ndim!=2 or xyz.shape[1]!=3 or len(pieces)!=len(xyz):
        raise ValueError('Invalid diagnostic geometry/displacement shape')
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(u)): raise ValueError('Finite diagnostic data required')
    if len(xyz)==0: raise ValueError('Empty diagnostic geometry')
    if any(len(pair)!=2 or any(not isinstance(i,(int,np.integer)) or i<0 or i>=len(xyz) for i in pair) for pair in seams):
        raise ValueError('Invalid diagnostic seam node indices')
    if seam_axes is not None:
        axes=np.asarray(seam_axes,dtype=float)
        if axes.shape!=(len(seams),3,3) or not np.all(np.isfinite(axes)):
            raise ValueError('Invalid seam coordinate axes')
        if any(not np.allclose(a@a.T,np.eye(3),atol=1e-8) for a in axes):
            raise ValueError('Orthonormal seam coordinate axes required')
    common,unused=rigid_fit(xyz,u); residual=u-common
    relative=np.zeros_like(u)
    for piece in set(pieces):
        mask=pieces==piece;relative[mask],unused=rigid_fit(xyz[mask],residual[mask])
    seam=[(residual[b]-residual[a]).tolist() for a,b in seams]
    seam_components=[]
    if seam_axes is not None:
        for vector,axes in zip(seam,seam_axes):
            opening,transverse,longitudinal=np.asarray(axes)@vector
            seam_components.append(dict(opening=float(opening),transverse_slip=float(transverse),
                longitudinal_slip=float(longitudinal),labels=['opening','transverse_slip','longitudinal_slip']))
    total=np.linalg.norm(u)
    return dict(relative_piece_norm=float(np.linalg.norm(relative)),
                relative_piece_norm_percent=float(100*np.linalg.norm(relative)/max(total,1e-250)),
                seam_vectors=seam,seam_components=seam_components,seam_relative_norm=float(np.linalg.norm(seam)),
                metric='NODAL_TRANSLATION_NORM_DIAGNOSTIC_NOT_FAMILY_ENERGY')
