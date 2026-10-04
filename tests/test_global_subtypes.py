import unittest
import numpy as np
from mfsm_model import BasisPack,ModeBatch
from mechanical_modal_classifier import classify_modes,EnergyProjector

class GlobalSubtypeTests(unittest.TestCase):
    def test_only_verified_mechanical_global_content_gets_subtype(self):
        basis=BasisPack(np.eye(3),{'G':np.eye(3)[:,:2],'L':np.eye(3)[:,2:]},{},'base')
        sub=BasisPack(np.eye(3),{'FLEXURAL':np.eye(3)[:,0:1],'TORSIONAL':np.eye(3)[:,1:2]},
            {'mechanical_definition_review':True},'global')
        x=np.array([[1.,0.,1.],[0.,1.,1.],[0.,0.,0.]])
        rows=classify_modes(ModeBatch(x,(1,2,3)),basis,global_projector=EnergyProjector(sub))
        self.assertEqual([r['global_subtype'] for r in rows],['FLEXURAL','TORSIONAL','FLEXURAL_TORSIONAL'])
        self.assertAlmostEqual(rows[-1]['global_shares_percent']['TORSIONAL'],50.)
    def test_missing_review_or_non_global_mode_has_no_subtype(self):
        b=BasisPack(np.eye(2),{'G':np.eye(2)[:,0:1],'L':np.eye(2)[:,1:2]},{},'base')
        sub=BasisPack(np.eye(2),{'FLEXURAL':np.eye(2)[:,0:1]},{},'g')
        rows=classify_modes(ModeBatch(np.eye(2),(1,2)),b,global_projector=EnergyProjector(sub))
        self.assertEqual([r['global_subtype'] for r in rows],[None,None])
    def test_repeated_global_eigenspace_subtype_is_indeterminate(self):
        from mechanical_modal_classifier import classify_cluster
        b=BasisPack(np.eye(2),{'G':np.eye(2)},{},'base')
        sub=BasisPack(np.eye(2),{'FLEXURAL':np.eye(2)[:,0:1],'TORSIONAL':np.eye(2)[:,1:2]},
            {'mechanical_definition_review':True},'global')
        result=classify_cluster(ModeBatch(np.eye(2),(1,2)),b,global_projector=EnergyProjector(sub))
        self.assertEqual(result['stable_family'],'G')
        self.assertIsNone(result['global_subtype'])
        self.assertAlmostEqual(result['global_bounds_percent']['FLEXURAL'][0],0.)
