import unittest
import numpy as np
from builtup_reference_section import build_reference
from shell_strain_operators import assemble_operators

class OperatorTests(unittest.TestCase):
    def test_piece_nodes_not_merged_across_gap(self):
        ref=build_reference({1:[[0,0,1,0]],2:[[1,0,2,0]]},1.)
        self.assertEqual(len(ref.nodes),4)
        self.assertNotEqual(ref.edges[0][1],ref.edges[1][0])
    def test_strain_patch_and_poisson_guard(self):
        ref=build_reference({1:[[0,0,1,0]]},1.)
        operators=assemble_operators(ref, {'eps_x':np.array([[1.,-1.]]),'eps_y':np.array([[1.,1.]])}, {'E':2.,'nu_class':0.,'thickness':1.}, [1.])
        self.assertAlmostEqual(np.ones(2)@operators.components['eps_x']@np.ones(2),0.)
        np.testing.assert_allclose(operators.system,4*np.eye(2))
        with self.assertRaises(ValueError):
            assemble_operators(ref, {'eps_x':np.eye(2)}, {'E':2.,'nu_class':.3,'thickness':1.},[1.,1.])
    def test_zero_length_wall_rejected(self):
        with self.assertRaises(ValueError): build_reference({1:[[0,0,0,0]]},1.)
