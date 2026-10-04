import unittest
import numpy as np
from mfsm_model import BasisPack,ModeBatch
from mechanical_modal_classifier import EnergyProjector,classify_modes,classify_cluster

class ClassifierTests(unittest.TestCase):
    def basis(self):
        return BasisPack(np.diag([2.,3.,4.,5.]),{k:np.eye(4)[:,i:i+1] for i,k in enumerate(('L','D','G','TE'))},{},'id')
    def test_energy_share_is_not_displacement_share(self):
        batch=ModeBatch(np.array([[1.],[1.],[0.],[0.]]),(31,))
        rows=classify_modes(batch,self.basis())
        self.assertAlmostEqual(rows[0]['shares_percent']['L'],40.)
        self.assertAlmostEqual(rows[0]['shares_percent']['D'],60.)
        self.assertEqual(rows[0]['quality_state'],'MIXED')
        self.assertFalse(rows[0]['scientifically_eligible'])
    def test_sign_amplitude_and_complete_non_ldg_space(self):
        p=EnergyProjector(self.basis())
        a=p.project(np.eye(4));b=p.project(-7*np.eye(4))
        for k in a['shares_percent']: np.testing.assert_allclose(a['shares_percent'][k],b['shares_percent'][k])
        row=classify_modes(ModeBatch(np.eye(4)[:,3:],(4,)),self.basis())[0]
        self.assertIsNone(row['dominant_family'])
        self.assertAlmostEqual(row['shares_percent']['TE'],100.)
    def test_cluster_rotation_invariance(self):
        x=np.eye(4)[:,:2]
        rotation=np.array([[.6,-.8],[.8,.6]])
        a=classify_cluster(ModeBatch(x,(1,2)),self.basis())
        b=classify_cluster(ModeBatch(x@rotation,(1,2)),self.basis())
        for k in a['trace_shares_percent']: self.assertAlmostEqual(a['trace_shares_percent'][k],b['trace_shares_percent'][k])
        self.assertIsNone(a['stable_family'])
    def test_joint_fit_signed_cross_and_residual_closure(self):
        basis=BasisPack(np.eye(3),{'L':np.array([[1.],[0.],[0.]]),'D':np.array([[1.],[1.],[0.]])},{},'id')
        result=EnergyProjector(basis).project(np.array([1.,1.,2.]))
        self.assertLess(result['closure_error'],1e-12)
        self.assertGreater(result['relative_residual'][0],.05)
    def test_overlapping_spaces_rejected(self):
        b=BasisPack(np.eye(2),{'L':np.eye(2)[:,0:1],'D':np.eye(2)[:,0:1]},{},'id')
        with self.assertRaises(ValueError): EnergyProjector(b)
    def test_scaled_family_columns_and_cluster_vectors_preserve_span(self):
        b=BasisPack(np.eye(2),{'L':np.diag([1.,1e-12])},{},'id')
        r=classify_modes(ModeBatch(np.array([[0.],[1.]]),(1,)),b)[0]
        self.assertEqual(r['dominant_family'],'LOCAL')
        cluster=classify_cluster(ModeBatch(np.diag([1.,1e-12]),(1,2)),b)
        self.assertEqual(cluster['stable_family'],'L')
    def test_cluster_cannot_bypass_signed_cross_term_gate(self):
        b=BasisPack(np.eye(2),{'L':np.array([[1.],[0.]]),'D':np.array([[1.],[1.]])},{},'id')
        x=ModeBatch(np.array([[1.],[-.1]]),(1,))
        self.assertEqual(classify_modes(x,b)[0]['quality_state'],'UNRESOLVED')
        self.assertIsNone(classify_cluster(x,b)['stable_family'])
