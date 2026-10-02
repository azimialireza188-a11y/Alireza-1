# -*- coding: utf-8 -*-
"""Assembly and seam-relative kinematic diagnostics for four-piece modes."""
from __future__ import print_function
import math
import numpy as np


def _arrays(mode, reference):
    u=np.asarray(mode['U'],float); ur=np.asarray(mode['UR'],float)
    if u.shape!=ur.shape or u.ndim!=2 or u.shape[1]!=3 or len(u)!=len(reference['nodes']):
        raise ValueError('Mode U/UR must be (reference nodes,3)')
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(ur)):
        raise ValueError('Mode contains nonfinite U/UR')
    xyz=np.array([[n['x'],n['y'],0.] for n in reference['nodes']],float)
    return u,ur,xyz


def _nodal_weights(reference):
    w=np.zeros(len(reference['nodes']),float)
    lookup={n['id']:i for i,n in enumerate(reference['nodes'])}
    for e in reference['elements']:
        value=float(e['length_mm'])*float(e.get('thickness_mm',reference['material']['thickness_mm']))
        w[lookup[e['n1']]]+=value/2.; w[lookup[e['n2']]]+=value/2.
    if np.any(w<=0):
        positive=w[w>0]
        fill=float(np.mean(positive)) if len(positive) else 1.
        w[w<=0]=fill
    return w


def _design(xyz):
    A=np.zeros((len(xyz)*6,6),float)
    for i,(x,y,z) in enumerate(xyz):
        r=6*i
        A[r+0]=[1,0,0,0,z,-y]
        A[r+1]=[0,1,0,-z,0,x]
        A[r+2]=[0,0,1,y,-x,0]
        A[r+3]=[0,0,0,1,0,0]
        A[r+4]=[0,0,0,0,1,0]
        A[r+5]=[0,0,0,0,0,1]
    return A


def _stack(u,ur):
    return np.column_stack((u,ur)).reshape(-1)


def _unstack(v,n):
    a=np.asarray(v,float).reshape(n,6)
    return a[:,:3],a[:,3:]


def _metric_row_scales(weights,rotation_scale):
    scales=np.empty(len(weights)*6,float)
    for i,w in enumerate(weights):
        root=math.sqrt(float(w))
        scales[6*i:6*i+3]=root
        scales[6*i+3:6*i+6]=root*rotation_scale
    return scales


def _rigid_fit(u,ur,xyz,weights,rotation_scale):
    A=_design(xyz); y=_stack(u,ur); scale=_metric_row_scales(weights,rotation_scale)
    Aw=A*scale[:,None]; yw=y*scale
    q=np.linalg.lstsq(Aw,yw,rcond=None)[0]
    pred=A@q
    return q,_unstack(pred,len(xyz))


def _metric_norm2(u,ur,weights,rotation_scale):
    return float(np.sum(weights[:,None]*(u*u+(rotation_scale*ur)*(rotation_scale*ur))))


def assembly_diagnostics(mode,reference,metric=None):
    u,ur,xyz=_arrays(mode,reference)
    weights=_nodal_weights(reference)
    t=float(reference['material']['thickness_mm'])
    rotation_scale=(t/math.sqrt(12.)) if metric is None else float(metric.get('rotation_scale_mm',t/math.sqrt(12.)))
    common_q,(common_u,common_ur)=_rigid_fit(u,ur,xyz,weights,rotation_scale)
    rel_u=np.zeros_like(u); rel_ur=np.zeros_like(ur)
    within_u=np.zeros_like(u); within_ur=np.zeros_like(ur)
    piece_parameters={}
    for piece in sorted({n['piece'] for n in reference['nodes']}):
        ids=np.array([i for i,n in enumerate(reference['nodes']) if n['piece']==piece],dtype=int)
        q,(pu,pur)=_rigid_fit(u[ids],ur[ids],xyz[ids],weights[ids],rotation_scale)
        common_piece_u,common_piece_ur=_unstack(_design(xyz[ids])@common_q,len(ids))
        rel_u[ids]=pu-common_piece_u; rel_ur[ids]=pur-common_piece_ur
        within_u[ids]=u[ids]-pu; within_ur[ids]=ur[ids]-pur
        piece_parameters[piece]=q.tolist()
    rel=_metric_norm2(rel_u,rel_ur,weights,rotation_scale)
    within=_metric_norm2(within_u,within_ur,weights,rotation_scale)
    total_mode=_metric_norm2(u,ur,weights,rotation_scale)
    denom=rel+within
    # A purely common rigid-body motion leaves only roundoff in rel/within.
    # Do not normalize two machine-noise quantities into a spurious ~50/50 split.
    negligible=max(1e-250, 1e-20*max(total_mode, 1e-250))
    if denom <= negligible:
        ap=0.; wp=0.
    else:
        ap=100.*rel/denom
        wp=100.*within/denom
    return dict(assembly_percent=ap,within_piece_deformation_percent=wp,
                common_rigid_parameters=common_q.tolist(),
                piece_rigid_parameters=piece_parameters,
                rotation_scale_mm=rotation_scale,
                convention='relative piece rigid-like motion / non-common modal motion; not subtracted from L/D/G')


def _nearest_node(reference,piece,xy):
    candidates=[(i,n) for i,n in enumerate(reference['nodes']) if n['piece']==piece]
    if not candidates:
        raise ValueError('Unknown seam piece '+str(piece))
    p=np.asarray(xy,float)
    dist=[np.linalg.norm(np.array([n['x'],n['y']])-p) for i,n in candidates]
    j=int(np.argmin(dist))
    scale=max(1.,max(math.hypot(n['x'],n['y']) for n in reference['nodes']))
    if dist[j]>1e-6*scale:
        raise ValueError('Seam point does not map to a reference node')
    return candidates[j][0]


def _seam_map(reference):
    result=[]
    for seam in reference.get('seam_pairs',[]):
        ia=_nearest_node(reference,seam['piece_a'],seam['point_a'])
        ib=_nearest_node(reference,seam['piece_b'],seam['point_b'])
        a=np.array([reference['nodes'][ia]['x'],reference['nodes'][ia]['y'],0.],float)
        b=np.array([reference['nodes'][ib]['x'],reference['nodes'][ib]['y'],0.],float)
        gap=b-a; gap[2]=0.
        length=np.linalg.norm(gap[:2])
        if length<=1e-12:
            raise ValueError('Seam pair has zero geometric gap')
        normal=gap/length
        tangent=np.array([-normal[1],normal[0],0.])
        result.append((seam,ia,ib,a,b,normal,tangent))
    return result


def seam_relative_diagnostics(mode,reference):
    u,ur,xyz=_arrays(mode,reference)
    seams=_seam_map(reference)
    if not seams:
        return dict(normal_opening_index=0.,transverse_slip_index=0.,
                    longitudinal_slip_index=0.,seam_count=0,values=[])
    weights=_nodal_weights(reference)
    # For an exact common small rigid rotation UR is constant. Using its weighted
    # mean prevents local U-only seam motion from biasing the rigid correction.
    omega=np.sum(weights[:,None]*ur,axis=0)/np.sum(weights)
    vals=[]
    for seam,ia,ib,ra,rb,n,t in seams:
        rigid_delta=np.cross(omega,rb-ra)
        delta=(u[ib]-u[ia])-rigid_delta
        vals.append(dict(id=seam['id'],
                         normal=float(np.dot(delta,n)),
                         transverse=float(np.dot(delta,t)),
                         longitudinal=float(delta[2])))
    # Dimensionless modal indices; normalization is common to all components.
    corrected=u-(np.cross(np.broadcast_to(omega,xyz.shape),xyz))
    center=np.sum(weights[:,None]*corrected,axis=0)/np.sum(weights)
    corrected=corrected-center
    base=math.sqrt(float(np.sum(weights[:,None]*corrected*corrected)/np.sum(weights)))
    radius=max(1.0, math.sqrt(float(np.sum(weights*np.sum(xyz[:,:2]**2,axis=1))/np.sum(weights))))
    total_scale=math.sqrt(float(np.sum(weights[:,None]*(u*u+(radius*ur)*(radius*ur)))/np.sum(weights)))
    if base<=max(1e-250, 1e-12*total_scale):
        indices=[0.,0.,0.]
    else:
        indices=[100.*math.sqrt(np.mean([v[k]**2 for v in vals]))/base
                 for k in ('normal','transverse','longitudinal')]
    return dict(normal_opening_index=indices[0],transverse_slip_index=indices[1],
                longitudinal_slip_index=indices[2],seam_count=len(vals),values=vals,
                common_rotation=omega.tolist(),
                convention='RMS seam relative motion after common rigid-rotation correction')


def impose_test_seam_motion(U,reference,seam_id,component,magnitude):
    """Deterministic synthetic helper used by regression tests."""
    for seam,ia,ib,ra,rb,n,t in _seam_map(reference):
        if int(seam['id'])!=int(seam_id):
            continue
        if component=='normal': direction=n
        elif component=='tangent': direction=t
        elif component=='longitudinal': direction=np.array([0.,0.,1.])
        else: raise ValueError('Unknown test seam component')
        U[ia]-=.5*float(magnitude)*direction
        U[ib]+=.5*float(magnitude)*direction
        return
    raise ValueError('Seam id not found')
