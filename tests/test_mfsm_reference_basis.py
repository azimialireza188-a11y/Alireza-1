import unittest
import numpy as np
from mfsm_model import OperatorPack,SourceDefinition
from mfsm_reference_basis import build_mfsm_basis, basis_cache_key

class BasisTests(unittest.TestCase):
    def test_energy_ratio_nullspace_with_connection_metric(self):
        a=OperatorPack(np.diag([2.,3.]),{'eps_x':np.diag([0.,1.])},SourceDefinition('paper','35','auxiliary_nu_zero'))
        basis=build_mfsm_basis(a,{'L':{'zero_components':['eps_x'],'equations':'2019 133','search_basis':np.eye(2)}})
        q=basis.spaces['L']
        self.assertEqual(q.shape,(2,1))
        np.testing.assert_allclose(q.T@a.system@q,np.eye(1),atol=1e-13)
        np.testing.assert_allclose(a.components['eps_x']@q,0,atol=1e-13)
    def test_no_implicit_search_space_or_negative_operator(self):
        a=OperatorPack(np.eye(2),{'eps_x':np.eye(2)},SourceDefinition('p','35','auxiliary_nu_zero'))
        with self.assertRaises(ValueError): build_mfsm_basis(a,{'L':{'zero_components':['eps_x'],'equations':'133'}})
        b=OperatorPack(np.eye(2),{'eps_x':np.diag([-1.,0.])},a.source)
        with self.assertRaises(ValueError): build_mfsm_basis(b,{'L':{'zero_components':['eps_x'],'equations':'133','search_basis':np.eye(2)}})
    def test_layout_changes_metric_cache_key(self):
        a=OperatorPack(np.eye(2),{'eps_x':np.eye(2)},SourceDefinition('p','35','auxiliary_nu_zero'))
        b=OperatorPack(np.diag([1.,2.]),a.components,a.source)
        self.assertNotEqual(basis_cache_key(a,{}),basis_cache_key(b,{}))
    def test_equivalent_scaled_search_columns_preserve_nullspace(self):
        a=OperatorPack(np.eye(2),{'eps_x':np.diag([1.,0.])},SourceDefinition('p','35','auxiliary_nu_zero'))
        def build(h): return build_mfsm_basis(a,{'L':dict(zero_components=['eps_x'],equations='125',search_basis=h)}).spaces['L']
        q=build(np.diag([1.,1e-6]))
        self.assertEqual(q.shape,(2,1))
        np.testing.assert_allclose(q@q.T,build(np.eye(2))@build(np.eye(2)).T,atol=1e-12)
