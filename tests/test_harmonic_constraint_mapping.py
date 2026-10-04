import unittest
import numpy as np
from scipy.sparse import csr_matrix
from prismatic_mfsm_operators import extruded_reference,reconstruct_six_dofs,transverse_rotation_map
from abaqus_mfsm_input import compile_initial_constraints


def fixture():
    points=[(0.,0.),(2.,0.),(2.,2.)];z=np.linspace(0.,10.,7)
    nodes={(p,3*j+i+1):np.array([x+shift,y,zz])
           for p,shift in [('A',0.),('B',5.)] for j,zz in enumerate(z)
           for i,(x,y) in enumerate(points)}
    elements={(p,2*j+i+1):tuple((p,n) for n in [3*j+i+1,3*j+i+2,3*j+i+5,3*j+i+4])
              for p in ('A','B') for j in range(6) for i in range(2)}
    model=dict(nodes=nodes,elements=elements,keywords=[],steps=[],sets={},boundary=[],
               mpcs=[['BEAM','B.7','A.7']],input_sha256='a'*64)
    return model,extruded_reference(model)


class HarmonicConstraintMappingTests(unittest.TestCase):
    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('harmonic_constraint_mapping'),
                             'Coupled constraint mapping is missing')
        import harmonic_constraint_mapping as m
        return m

    def test_selected_sparse_evaluation_matches_complete_six_dof_reconstruction(self):
        m=self.module();model,r=fixture();terms=[1,3];n=len(r['xy'])
        rng=np.random.default_rng(8);a=rng.normal(size=(2,n,4,3));v0=rng.normal(size=(n,3))
        raw=reconstruct_six_dofs(r['xy'],r['edges'],r['stations'],a,terms,r['length'])
        raw[:,:,2,:]+=v0[None,:,:]
        rotations=transverse_rotation_map(r['xy'],r['edges'],[1],r['length'])[:,1::4]@v0
        raw[:,:,[3,4],:]+=rotations.reshape(n,2,3)[None,:,:,:]
        locations=[(z,i,d) for z in range(7) for i in range(n) for d in range(6)]
        selected=locations[::3]+locations[1::7]
        keys=[(*r['node_keys'][z][i],d+1) for z,i,d in selected]
        evaluation=m.selected_reconstruction(r,terms,keys,include_constant_warping=True)
        self.assertIsInstance(evaluation,csr_matrix)
        values=np.vstack([a.reshape(2*n*4,3),v0])
        np.testing.assert_allclose(evaluation@values,np.array([raw[z,i,d] for z,i,d in selected]),atol=1e-13)

    def test_offset_beam_pullback_preserves_actual_cross_harmonic_coupling(self):
        m=self.module();model,r=fixture();c,keys,meta=compile_initial_constraints(model)
        pullback,info=m.compile_harmonic_constraints(c,keys,r,[1,2,3])
        self.assertEqual(pullback.shape,(6,3*len(r['xy'])*4+len(r['xy'])))
        evaluation=m.selected_reconstruction(r,[1,2,3],keys,include_constant_warping=True)
        np.testing.assert_allclose(pullback.toarray(),(c@evaluation).toarray(),atol=1e-14)
        self.assertTrue(info['cross_harmonic_constraints_retained'])
        self.assertLess(info['evaluated_raw_dofs'],len(keys))
        size=4*len(r['xy']);gram=pullback.T@pullback
        self.assertGreater(np.linalg.norm(gram[:size,size:2*size].toarray()),0.)
        self.assertFalse(info['active_contact_verified'])

    def test_endpoint_transverse_constraints_have_exact_zero_rows(self):
        m=self.module();model,r=fixture();keys=[(*key,d) for key in model['nodes'] for d in range(1,7)]
        column=keys.index(('A',1,1));c=csr_matrix(([1.],([0],[column])),shape=(1,len(keys)))
        pullback,info=m.compile_harmonic_constraints(c,keys,r,[1,2,3])
        self.assertEqual(pullback.nnz,0);self.assertEqual(info['zero_rows'],[0])
        # Zero rows are identified, not silently treated as independent reduction rows.
        self.assertEqual(pullback.shape[0],1)

    def test_invalid_keys_or_nonfinite_constraints_fail_closed(self):
        m=self.module();model,r=fixture();c,keys,meta=compile_initial_constraints(model)
        for changed in ([('UNKNOWN',1,1)]+keys[1:],keys[:-1]):
            with self.assertRaises(ValueError):m.compile_harmonic_constraints(c,changed,r,[1])
        bad=c.copy();bad.data[0]=np.nan
        with self.assertRaises(ValueError):m.compile_harmonic_constraints(bad,keys,r,[1])
        with self.assertRaises(ValueError):m.selected_reconstruction(r,[1],[('A',1,7)])
