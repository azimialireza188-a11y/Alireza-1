# -*- coding: utf-8 -*-
"""Whole-section force-based fcFSM reference basis for S-S harmonics.

The formulas follow CUFSM 5.70 klocal.m/trans.m/assemble.m and the force-space
definitions in analysis/fcFSM/SecAnal_fcFSM.m. Physical pieces remain
disconnected; singular K0 directions are handled on the energetic subspace.
"""
from __future__ import print_function
import hashlib
import json
import math
import numpy as np
from scipy.linalg import null_space


def _null(a, rcond=1e-10):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.eye(a.shape[1], dtype=float)
    return null_space(a, rcond=rcond)


class EnergeticSolver(object):
    def __init__(self, stiffness, rtol=1e-10):
        k = np.asarray(stiffness, dtype=float)
        if k.ndim != 2 or k.shape[0] != k.shape[1] or not np.all(np.isfinite(k)):
            raise ValueError('K0 must be a finite square matrix')
        scale = max(float(np.max(np.abs(k))), 1e-250)
        if not np.allclose(k, k.T, rtol=1e-10, atol=scale*1e-12):
            raise ValueError('K0 must be symmetric')
        self.K = .5*(k+k.T)
        eig, q = np.linalg.eigh(self.K)
        emax = max(float(np.max(np.abs(eig))), 1e-250)
        tol = rtol*emax
        self.negative_count = int(np.sum(eig < -tol))
        if self.negative_count:
            raise ValueError('K0 has significant negative elastic eigenvalues')
        keep = eig > tol
        self.rank = int(np.sum(keep))
        self.nullity = len(eig)-self.rank
        if not self.rank:
            raise ValueError('K0 has no positive energetic subspace')
        self.eig = eig[keep]
        self.q = q[:, keep]
        self.tolerance = tol
        self.condition = float(self.eig[-1]/self.eig[0])

    def solve(self, rhs):
        b = np.asarray(rhs, dtype=float)
        one = b.ndim == 1
        if one:
            b = b[:, None]
        if b.shape[0] != self.K.shape[0]:
            raise ValueError('RHS row count differs from K0')
        result = self.q @ ((self.q.T @ b)/self.eig[:, None])
        return result[:, 0] if one else result

    def energetic_basis(self):
        return self.q.copy()


def _k_orthonormalize(c, k, tol=1e-10):
    c = np.asarray(c, dtype=float)
    if c.ndim != 2 or c.shape[1] == 0:
        return np.empty((k.shape[0], 0))
    gram = .5*(c.T@k@c + (c.T@k@c).T)
    eig, q = np.linalg.eigh(gram)
    limit = max(float(np.max(np.abs(eig))), 1e-250)*tol
    keep = eig > limit
    if not np.any(keep):
        return np.empty((k.shape[0], 0))
    return c @ (q[:, keep]/np.sqrt(eig[keep])[None, :])


class FamilyBasis(object):
    def __init__(self, k0, j_gd, j_d, c_l, c_d, c_g, solver,
                 definition_hash=None, metadata=None):
        self.K0 = np.asarray(k0, dtype=float)
        self.J_GD = np.asarray(j_gd, dtype=float)
        self.J_D = np.asarray(j_d, dtype=float)
        self.C_L = _k_orthonormalize(c_l, self.K0)
        self.C_D = _k_orthonormalize(c_d, self.K0)
        self.C_G = _k_orthonormalize(c_g, self.K0)
        self.solver = solver
        self.definition_hash = definition_hash
        self.metadata = dict(metadata or {})
        self.condition_report = dict(rank=solver.rank, nullity=solver.nullity,
                                     negative_count=solver.negative_count,
                                     energetic_condition=solver.condition,
                                     eigen_tolerance=solver.tolerance)

    def project(self, vector):
        x = np.asarray(vector, dtype=float)
        if x.ndim != 1 or len(x) != self.K0.shape[0] or not np.all(np.isfinite(x)):
            raise ValueError('Invalid reference mode vector')
        parts = {}
        energies = []
        for name, c in (('L', self.C_L), ('D', self.C_D), ('G', self.C_G)):
            if c.shape[1]:
                part = c @ (c.T @ self.K0 @ x)
            else:
                part = np.zeros_like(x)
            parts[name] = part
            energies.append(.5*float(part @ self.K0 @ part))
        reconstruction = parts['L']+parts['D']+parts['G']
        residual = x-reconstruction
        residual_energy = .5*float(residual @ self.K0 @ residual)
        total = sum(energies)+max(0.0, residual_energy)
        if total <= 1e-250:
            raise ValueError('Reference vector has zero K0 energy')
        percentages = [100.*e/total for e in energies]
        return dict(components=parts, residual=residual,
                    energy_percent=percentages,
                    residual_energy_percent=100.*max(0., residual_energy)/total,
                    family=('L','D','G')[int(np.argmax(percentages))])


def force_family_basis(k0, j_gd, equilibrium, definition_hash=None, metadata=None):
    k = np.asarray(k0, dtype=float)
    j = np.asarray(j_gd, dtype=float)
    e = np.asarray(equilibrium, dtype=float)
    if j.ndim != 2 or j.shape[0] != k.shape[0] or j.shape[1] == 0:
        raise ValueError('J_GD has invalid dimensions')
    if e.ndim != 2 or e.shape[1] != j.shape[1]:
        raise ValueError('Equilibrium operator has invalid dimensions')
    if not np.all(np.isfinite(j)) or not np.all(np.isfinite(e)):
        raise ValueError('J_GD/equilibrium must be finite')
    # Column normalization improves numerical invariance without changing force space.
    scales = np.linalg.norm(j, axis=0)
    if np.any(scales <= 0):
        raise ValueError('J_GD contains an empty wall-force direction')
    jn = j/scales[None, :]
    en = e/scales[None, :]
    solver = EnergeticSolver(k)
    qpos = solver.energetic_basis()
    c_l = qpos @ _null(jn.T @ qpos)
    j_d = _null(en)
    gd_response = solver.solve(jn)
    c_d = gd_response @ j_d
    j_g = _null(c_d.T @ jn)
    c_g = gd_response @ j_g
    return FamilyBasis(k, jn, j_d, c_l, c_d, c_g, solver,
                       definition_hash=definition_hash, metadata=metadata)


def basis_cache_key(reference_hash, harmonic_n, bc='S-S'):
    payload = '%s|m=%d|bc=%s|fcfsm-reference-v1' % (
        str(reference_hash), int(harmonic_n), str(bc).upper())
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _ss_integrals(m, a):
    mm = float(m)
    pi = math.pi
    return (a/2.,
            -mm*mm*pi*pi/(2.*a),
            -mm*mm*pi*pi/(2.*a),
            pi**4*mm**4/(2.*a**3),
            pi*pi*mm*mm/(2.*a))


def _strip_local_k(E, nu, t, a, b, m):
    G = E/(2.*(1.+nu))
    E1 = E/(1.-nu*nu)
    E2 = E1
    Dx = E*t**3/(12.*(1.-nu*nu))
    Dy = Dx
    D1 = nu*E*t**3/(12.*(1.-nu*nu))
    Dxy = G*t**3/12.
    I1,I2,I3,I4,I5 = _ss_integrals(m,a)
    c = m*math.pi/a
    km = np.zeros((4,4), float)
    km[0,0]=E1*I1/b+G*b*I5/3.
    km[0,1]=E2*nu*(-.5/c)*I3-G*I5/(2.*c)
    km[0,2]=-E1*I1/b+G*b*I5/6.
    km[0,3]=E2*nu*(-.5/c)*I3+G*I5/(2.*c)
    km[1,0]=E2*nu*(-.5/c)*I2-G*I5/(2.*c)
    km[1,1]=E2*b*I4/(3.*c*c)+G*I5/(b*c*c)
    km[1,2]=E2*nu*(.5/c)*I2-G*I5/(2.*c)
    km[1,3]=E2*b*I4/(6.*c*c)-G*I5/(b*c*c)
    km[2,0]=km[0,2]
    km[2,1]=E2*nu*(.5/c)*I3-G*I5/(2.*c)
    km[2,2]=km[0,0]
    km[2,3]=E2*nu*(.5/c)*I3+G*I5/(2.*c)
    km[3,0]=E2*nu*(-.5/c)*I2+G*I5/(2.*c)
    km[3,1]=km[1,3]
    km[3,2]=E2*nu*(.5/c)*I2+G*I5/(2.*c)
    km[3,3]=km[1,1]
    km *= t

    den=420.*b**3
    kf=np.zeros((4,4),float)
    kf[0,0]=(5040*Dx*I1-504*b*b*D1*I2-504*b*b*D1*I3+156*b**4*Dy*I4+2016*b*b*Dxy*I5)/den
    kf[0,1]=(2520*b*Dx*I1-462*b**3*D1*I2-42*b**3*D1*I3+22*b**5*Dy*I4+168*b**3*Dxy*I5)/den
    kf[0,2]=(-5040*Dx*I1+504*b*b*D1*I2+504*b*b*D1*I3+54*b**4*Dy*I4-2016*b*b*Dxy*I5)/den
    kf[0,3]=(2520*b*Dx*I1-42*b**3*D1*I2-42*b**3*D1*I3-13*b**5*Dy*I4+168*b**3*Dxy*I5)/den
    kf[1,0]=(2520*b*Dx*I1-462*b**3*D1*I3-42*b**3*D1*I2+22*b**5*Dy*I4+168*b**3*Dxy*I5)/den
    kf[1,1]=(1680*b*b*Dx*I1-56*b**4*D1*I2-56*b**4*D1*I3+4*b**6*Dy*I4+224*b**4*Dxy*I5)/den
    kf[1,2]=(-2520*b*Dx*I1+42*b**3*D1*I2+42*b**3*D1*I3+13*b**5*Dy*I4-168*b**3*Dxy*I5)/den
    kf[1,3]=(840*b*b*Dx*I1+14*b**4*D1*I2+14*b**4*D1*I3-3*b**6*Dy*I4-56*b**4*Dxy*I5)/den
    kf[2,0]=kf[0,2]; kf[2,1]=kf[1,2]
    kf[2,2]=kf[0,0]
    kf[2,3]=(-2520*b*Dx*I1+462*b**3*D1*I2+42*b**3*D1*I3-22*b**5*Dy*I4-168*b**3*Dxy*I5)/den
    kf[3,0]=kf[0,3]; kf[3,1]=kf[1,3]
    kf[3,2]=(-2520*b*Dx*I1+462*b**3*D1*I3+42*b**3*D1*I2-22*b**5*Dy*I4-168*b**3*Dxy*I5)/den
    kf[3,3]=kf[1,1]
    out=np.zeros((8,8),float); out[:4,:4]=km; out[4:,4:]=kf
    return out


def _transform_k(k, alpha):
    c,s=math.cos(alpha),math.sin(alpha)
    g=np.array([[c,0,0,0,-s,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,c,0,0,0,-s,0],
                [0,0,0,1,0,0,0,0],
                [s,0,0,0,c,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,s,0,0,0,c,0],
                [0,0,0,0,0,0,0,1]],float)
    return g@k@g.T


def _reference_operators(reference, harmonic_n, bc):
    if str(bc).upper() != 'S-S':
        raise ValueError('Stage A reference basis currently supports S-S only')
    nodes=reference['nodes']; n=len(nodes)
    lookup={node['id']:i for i,node in enumerate(nodes)}
    by_elem={e['id']:e for e in reference['elements']}
    E=reference['material']['E_MPa']; nu=reference['material']['nu']
    t=reference['material']['thickness_mm']; a=reference['length_mm']
    k0=np.zeros((4*n,4*n),float)
    for elem in reference['elements']:
        i,j=lookup[elem['n1']],lookup[elem['n2']]
        p1=np.array([nodes[i]['x'],nodes[i]['y']],float)
        p2=np.array([nodes[j]['x'],nodes[j]['y']],float)
        d=p2-p1; b=float(np.linalg.norm(d))
        kl=_strip_local_k(E,nu,t,a,b,harmonic_n)
        kg=_transform_k(kl,math.atan2(d[1],d[0]))
        idx=[2*i,2*i+1,2*j,2*j+1,2*n+2*i,2*n+2*i+1,2*n+2*j,2*n+2*j+1]
        k0[np.ix_(idx,idx)] += kg
    groups=reference['plate_groups']
    jgd=np.zeros((4*n,len(groups)),float)
    eq=np.zeros((3,len(groups)),float)
    for col,group in enumerate(groups):
        ids=group['element_ids']
        if not ids:
            raise ValueError('Empty physical plate group')
        first=by_elem[ids[0]]
        ia,ib=lookup[first['n1']],lookup[first['n2']]
        pa=np.array([nodes[ia]['x'],nodes[ia]['y']],float)
        pb=np.array([nodes[ib]['x'],nodes[ib]['y']],float)
        tangent=(pb-pa)/np.linalg.norm(pb-pa)
        area=0.
        representative=pa
        for eid in ids:
            elem=by_elem[eid]
            i,j=lookup[elem['n1']],lookup[elem['n2']]
            width=float(elem['length_mm']); area += width*t
            for inode in (i,j):
                jgd[2*inode,col] += width*t*tangent[0]/2.
                jgd[2*n+2*inode,col] += width*t*tangent[1]/2.
        eq[:,col]=[tangent[0]*area,tangent[1]*area,
                   (representative[0]*tangent[1]-representative[1]*tangent[0])*area]
    return .5*(k0+k0.T),jgd,eq


def build_fcfsm_basis(reference, harmonic_n, bc='S-S'):
    m=int(harmonic_n)
    if m<1:
        raise ValueError('harmonic_n must be a positive integer')
    k0,jgd,eq=_reference_operators(reference,m,bc)
    key=basis_cache_key(reference['definition_hash'],m,bc)
    metadata=dict(method='whole-built-up fcFSM force basis', harmonic_n=m, bc=str(bc).upper(),
                  reference_hash=reference['definition_hash'],
                  cross_gap_constraints='none', equilibrium_rows=['Fx','Fy','M'])
    result=force_family_basis(k0,jgd,eq,definition_hash=key,metadata=metadata)
    result.equilibrium=eq
    return result
