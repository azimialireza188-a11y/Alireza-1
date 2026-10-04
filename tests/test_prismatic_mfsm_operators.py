import importlib.util
import unittest
import numpy as np
from scipy.sparse import csr_matrix

class PrismaticOperatorTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('prismatic_mfsm_operators'),'Automatic source interpolation missing')
        import prismatic_mfsm_operators as m
        return m

    def test_source_membrane_and_bending_analytical_energies(self):
        m=self.module();b=2.;length=10.;young=200.;t=.3;k=np.pi/length
        ops=m.strip_operators(np.array([[0.,0.],[b,0.]]),[(0,1)],[1],length,young,t)
        # Global U,V,W,theta: local normal = -global W.
        for dof,expected in [(0,young*t/2*k*k*b*length/2),(1,young*t*k*k*b*length/2),
                             (2,young*t**3/12*k**4*b*length/2)]:
            v=np.zeros(8);v[[dof,4+dof]]=1
            self.assertAlmostEqual(float(v@ops['system']@v),expected,places=10)
        for name,a in ops['components'].items():
            np.testing.assert_allclose(a.toarray(),a.toarray().T,atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(a.toarray()).min(),-1e-10)

    def test_cross_section_rigid_motion_and_rotation_sign(self):
        m=self.module();coords=np.array([[2.,3.],[4.,3.],[4.,5.]])
        ops=m.strip_operators(coords,[(0,1),(1,2)],[2],10.,100.,.2)
        v=np.zeros(12)
        for i,(x,y) in enumerate(coords):v[4*i:4*i+4]=[-y,0,x,1]
        for name in ('eps_x','kappa_x'):
            self.assertLess(abs(float(v@ops['components'][name]@v)),1e-9)
        for name in ('kappa_y','kappa_xy'):self.assertGreater(float(v@ops['components'][name]@v),0)

    def test_multiharmonic_orthogonality_and_invalid_input(self):
        m=self.module();o=m.strip_operators([[0,0],[2,0]],[(0,1)],[1,3],10.,100.,.2)
        np.testing.assert_allclose(o['system'][:8,8:].toarray(),0,atol=0)
        with self.assertRaises(ValueError):m.strip_operators([[0,0],[0,0]],[(0,1)],[1],10.,100.,.2)
        with self.assertRaises(ValueError):m.strip_operators([[0,0],[2,0]],[(0,1)],[1,1],10.,100.,.2)

    def test_extruded_reference_has_no_connections_across_gaps(self):
        m=self.module()
        nodes={(p,i+1+4*j):np.array([x+offset,y,z]) for p,offset in [('A',0),('B',3)]
               for j,z in enumerate([0.,5.,10.]) for i,(x,y) in enumerate([(0.,0.),(1.,0.),(1.,1.),(2.,1.)])}
        elements={};label=0
        for p in ('A','B'):
            for j in range(2):
                for i in range(3):
                    label+=1;elements[p,label]=tuple((p,n) for n in [i+1+4*j,i+2+4*j,i+6+4*j,i+5+4*j])
        r=m.extruded_reference(dict(nodes=nodes,elements=elements))
        self.assertEqual(len(r['xy']),8);self.assertEqual(len(r['edges']),6)
        self.assertTrue(all(r['pieces'][a]==r['pieces'][b] for a,b in r['edges']))
        broken=dict(nodes=dict(nodes),elements=elements);broken['nodes']['A',5]=np.array([.01,0,5])
        with self.assertRaises(ValueError):m.extruded_reference(broken)

    def test_exact_sparse_constraint_action_matches_dense_nullspace(self):
        m=self.module();from scipy.linalg import null_space
        c=csr_matrix([[1.,0,-1.,2],[0,1.,0,0]])
        p=m.ConstraintProjector(c);x=np.arange(12.).reshape(4,3);q=null_space(c.toarray())
        np.testing.assert_allclose(p.apply(x),q@q.T@x,atol=1e-12)
        np.testing.assert_allclose(c@p.apply(x),0,atol=1e-12)
        np.testing.assert_allclose(p.apply(p.apply(x)),p.apply(x),atol=1e-12)
        with self.assertRaises(ValueError):m.ConstraintProjector(csr_matrix([[1,0],[2,0]]))

    def test_harmonic_fit_recovers_all_four_canonical_dofs(self):
        m=self.module();z=np.linspace(0,10,13);terms=[1,2,4]
        rng=np.random.default_rng(21);a=rng.normal(size=(3,2,4,3))
        v=m.reconstruct_canonical(z,a,terms,10.)
        fit=m.fit_canonical(z,v,terms,10.)
        np.testing.assert_allclose(fit['coefficients'],a,atol=1e-12)
        self.assertLess(float(np.max(fit['relative_residual'])),1e-12)
        with self.assertRaises(ValueError):m.fit_canonical([0,5,10],v[:3],terms,10.)

    def test_fit_residual_is_per_mode_and_amplitude_invariant(self):
        m=self.module();z=np.linspace(0,10,11);a=np.ones((1,1,4,2));v=m.reconstruct_canonical(z,a,[1],10.)
        v[...,1]+=np.sin(2*np.pi*z/10)[:,None,None]
        base=m.fit_canonical(z,v,[1],10.)['relative_residual'];v[...,0]*=1e12;v[...,1]*=-1e-7
        other=m.fit_canonical(z,v,[1],10.)['relative_residual'];np.testing.assert_allclose(base,other,atol=1e-14)
        self.assertLess(base[0],1e-12);self.assertGreater(base[1],.1)

    def test_shared_fitter_factorizes_each_parity_once(self):
        m=self.module();self.assertTrue(hasattr(m,'CanonicalFitter'),'Shared harmonic factorization missing')
        from unittest.mock import patch
        z=np.linspace(0,10,13);terms=[1,2,4]
        a=np.random.default_rng(32).normal(size=(3,2,4,3));y=m.reconstruct_canonical(z,a,terms,10.)
        with patch.object(np.linalg,'svd',wraps=np.linalg.svd) as factor:
            fitter=m.CanonicalFitter(z,terms,10.,rotation_length=2.)
            for unused in range(3):np.testing.assert_allclose(fitter.fit(y)['coefficients'],a,atol=1e-12)
            self.assertEqual(factor.call_count,2)

    def test_six_dof_reconstruction_matches_kirchhoff_bending(self):
        m=self.module();self.assertTrue(hasattr(m,'reconstruct_six_dofs'),'Six DOF reconstruction missing')
        xy=np.array([[0.,0.],[2.,0.]]);z=np.linspace(0,10,11);a=np.zeros((1,2,4,1));k=np.pi/10
        # Normal bending Uy=sin, URx=-k cos; no membrane/drill rotation.
        a[:,:,2,:]=1
        raw=m.reconstruct_six_dofs(xy,[(0,1)],z,a,[1],10.)
        np.testing.assert_allclose(raw[:,:,1,0],np.sin(k*z)[:,None]*np.ones((1,2)),atol=1e-12)
        np.testing.assert_allclose(raw[:,:,3,0],-k*np.cos(k*z)[:,None]*np.ones((1,2)),atol=1e-12)
        np.testing.assert_allclose(raw[:,:,[0,2,4,5],0],0,atol=1e-12)
        # In-plane beam bending: Ux=sin, Uz=-k*x*cos, URy=k*cos.
        a[:]=0;a[:,:,0,:]=1;a[0,:,1,0]=-k*xy[:,0]
        raw=m.reconstruct_six_dofs(xy,[(0,1)],z,a,[1],10.)
        np.testing.assert_allclose(raw[:,:,4,0],k*np.cos(k*z)[:,None]*np.ones((1,2)),atol=1e-12)

    def test_constant_warping_is_retained_as_separate_zero_frequency_field(self):
        m=self.module();z=np.linspace(0,10,13);a=np.ones((2,2,4,1))
        y=m.reconstruct_canonical(z,a,[1,2],10.);mean=np.array([[2.],[3.]])
        y[:,:,1,:]+=mean[None,:,:]
        fit=m.CanonicalFitter(z,[1,2],10.,include_constant_warping=True).fit(y)
        np.testing.assert_allclose(fit['coefficients'],a,atol=1e-12)
        np.testing.assert_allclose(fit['constant_warping'],mean,atol=1e-12)
        self.assertLess(float(fit['relative_residual'][0]),1e-12)
        self.assertTrue(hasattr(m,'constant_warping_operator'),'Zero-frequency shear operator missing')
        k=m.constant_warping_operator([[0,0],[2,0]],[(0,1)],10.,200.,.3)
        self.assertAlmostEqual(float(np.array([0,2])@k@np.array([0,2])),600.)
        np.testing.assert_allclose(k@np.ones(2),0,atol=1e-12)
