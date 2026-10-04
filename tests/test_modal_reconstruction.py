import unittest
import numpy as np
from modal_reconstruction import ReconstructionCheck

class ReconstructionTests(unittest.TestCase):
    def test_loss_is_per_mode_and_amplitude_independent(self):
        mapping=np.array([[1.,0.]])
        check=ReconstructionCheck(mapping,mapping.T,[1.,4.],1e-8)
        raw=np.array([[1e10,1.],[0.,1.]])
        result=check.evaluate(raw,mapping@raw)
        self.assertEqual(result['accepted'],[True,False])
        self.assertAlmostEqual(result['relative_errors'][1],np.sqrt(4/5))
        other=check.evaluate(raw[:,1:]*-7,mapping@raw[:,1:]*-7)
        self.assertAlmostEqual(other['relative_errors'][0],result['relative_errors'][1])
    def test_inverse_incompatible_and_bad_units_rejected(self):
        with self.assertRaisesRegex(ValueError,'inverse'):
            ReconstructionCheck([[1.,0.]],[[0.],[1.]],[1.,1.])
        with self.assertRaises(ValueError):ReconstructionCheck([[1.,0.]],[[1.],[0.]],[1.,0.])
    def test_scale_safe_fp64_error(self):
        check=ReconstructionCheck(np.eye(2),np.eye(2),[1.,2.])
        self.assertEqual(check.evaluate(np.eye(2)*1e200,np.eye(2)*1e200)['relative_errors'],[0.,0.])
