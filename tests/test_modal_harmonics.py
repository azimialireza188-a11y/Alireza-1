import unittest
import numpy as np
from abaqus_modal_harmonics import fit_harmonics, map_modes
from mfsm_model import ModeBatch

class HarmonicTests(unittest.TestCase):
    def test_dof_appropriate_functions_and_multiterm(self):
        z=np.linspace(0,10,31)
        fields=np.column_stack((2*np.sin(np.pi*z/10)+np.sin(2*np.pi*z/10),3*np.cos(np.pi*z/10)))
        c,res=fit_harmonics(z,fields,[1,2],10,['sin','cos'])
        np.testing.assert_allclose(c,[[2,3],[1,0]],atol=1e-12)
        self.assertLess(res,1e-12)
    def test_mapping_dof_failure_and_constraint_check(self):
        batch=ModeBatch(np.eye(2),(1,2))
        with self.assertRaises(ValueError): map_modes(batch,np.ones((2,3)))
        with self.assertRaises(ValueError): map_modes(batch,np.eye(2),np.array([[1.,0.]]))
    def test_aliasing_is_rejected(self):
        with self.assertRaises(ValueError): fit_harmonics([0,10],np.zeros((2,1)),[1,2],10,['sin'])
    def test_constraint_check_is_independent_per_mode(self):
        batch=ModeBatch(np.array([[1e10,0.],[0.,1.]]),(1,2))
        with self.assertRaisesRegex(ValueError,'constraints'):
            map_modes(batch,np.eye(2),np.array([[0.,1.]]))
