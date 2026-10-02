# -*- coding: utf-8 -*-
"""DOF-aware S-S longitudinal decomposition for Abaqus U/UR eigenmodes."""
from __future__ import print_function
import math
import numpy as np


def ss_functions(z, length_mm, harmonic_n):
    z=np.asarray(z,dtype=float)
    L=float(length_mm); m=int(harmonic_n)
    if L<=0 or m<1:
        raise ValueError('Positive length and harmonic number required')
    phase=m*np.pi*z/L
    return dict(transverse=np.sin(phase), axial=np.cos(phase))


def _trap_weights(z):
    z=np.asarray(z,dtype=float)
    order=np.argsort(z)
    z=z[order]
    if len(z)<5 or np.any(np.diff(z)<=0):
        raise ValueError('At least five distinct ordered longitudinal stations are required')
    w=np.zeros(len(z),float)
    dz=np.diff(z)
    w[:-1]+=dz/2.; w[1:]+=dz/2.
    return order,w


def _project_family(z, values, length, max_harmonic, use_cos):
    order,w=_trap_weights(z)
    z=np.asarray(z,float)[order]
    y=np.asarray(values,float)[order]
    if y.ndim<2 or y.shape[0]!=len(z) or not np.all(np.isfinite(y)):
        raise ValueError('Invalid longitudinal modal array')
    basis=[]
    for m in range(1,max_harmonic+1):
        f=ss_functions(z,length,m)
        basis.append(f['axial' if use_cos else 'transverse'])
    B=np.column_stack(basis)
    WB=B*np.sqrt(w)[:,None]
    if np.linalg.matrix_rank(WB)<max_harmonic:
        raise ValueError('Longitudinal grid is underresolved for requested harmonics')
    projector=np.linalg.pinv(WB)*np.sqrt(w)[None,:]
    flat=y.reshape(len(z),-1)
    coef=projector@flat
    recon=B@coef
    residual=flat-recon
    num=float(np.sum(w[:,None]*residual*residual))
    den=float(np.sum(w[:,None]*flat*flat))
    return coef.reshape((max_harmonic,)+y.shape[1:]), num, den


def decompose_mode(z,U,UR,length_mm,max_harmonic,bc='S-S'):
    if str(bc).upper()!='S-S':
        raise ValueError('Stage A harmonic mapper currently supports S-S only')
    z=np.asarray(z,float)
    U=np.asarray(U,float); UR=np.asarray(UR,float)
    if U.shape!=UR.shape or U.ndim!=3 or U.shape[0]!=len(z) or U.shape[2]!=3:
        raise ValueError('U and UR must have shape (stations,nodes,3)')
    mmax=int(max_harmonic)
    if mmax<1:
        raise ValueError('max_harmonic must be positive')
    # CUFSM S-S convention: cross-section translations and longitudinal-axis
    # rotation use Ym=sin; axial translation and transverse-axis shell rotations
    # use the derivative-family cosine.
    usin,n_usin,d_usin=_project_family(z,U[:,:,:2],length_mm,mmax,False)
    ucos,n_ucos,d_ucos=_project_family(z,U[:,:,2:3],length_mm,mmax,True)
    urcos,n_urcos,d_urcos=_project_family(z,UR[:,:,:2],length_mm,mmax,True)
    ursin,n_ursin,d_ursin=_project_family(z,UR[:,:,2:3],length_mm,mmax,False)
    components={}
    translation_power=np.zeros(mmax,float)
    rotation_power=np.zeros(mmax,float)
    for k in range(mmax):
        uc=np.concatenate((usin[k],ucos[k]),axis=1)
        rc=np.concatenate((urcos[k],ursin[k]),axis=1)
        components[k+1]=dict(U=uc,UR=rc)
        translation_power[k]=float(np.sum(uc*uc))
        rotation_power[k]=float(np.sum(rc*rc))
    ptotal=float(np.sum(translation_power))
    if ptotal<=1e-250:
        ptotal=float(np.sum(rotation_power))
        power=rotation_power.copy()
    else:
        power=translation_power.copy()
    if ptotal<=1e-250:
        raise ValueError('Zero mode supplied to harmonic decomposition')
    shares=power/float(np.sum(power))
    dominant=int(np.argmax(shares))+1
    translation_res=math.sqrt((n_usin+n_ucos)/max(d_usin+d_ucos,1e-250))
    rotation_res=math.sqrt((n_urcos+n_ursin)/max(d_urcos+d_ursin,1e-250))
    return dict(components=components,
                translation_power=translation_power.tolist(),
                rotation_power=rotation_power.tolist(),
                shares=shares.tolist(),
                dominant_m=dominant,
                half_wavelength_mm=float(length_mm)/dominant,
                translation_relative_residual=translation_res,
                rotation_relative_residual=rotation_res,
                relative_residual=max(translation_res,rotation_res),
                convention='S-S: U1/U2/UR3 sine; U3/UR1/UR2 cosine')
