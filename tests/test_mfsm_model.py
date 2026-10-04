import unittest
import numpy as np
from mfsm_model import OperatorPack, BasisPack, SourceDefinition, ValidationEvidence

class ModelTests(unittest.TestCase):
    def test_source_and_operator_shapes(self):
        with self.assertRaises(ValueError):
            SourceDefinition('', 'eq35', 'auxiliary_nu_zero')
        source = SourceDefinition('2019/2023', 'eq35', 'auxiliary_nu_zero')
        with self.assertRaises(ValueError):
            OperatorPack(np.eye(2), {'eps_x': np.eye(3)}, source)
    def test_no_self_certification(self):
        evidence = ValidationEvidence('model', 'algorithm', {})
        self.assertFalse(evidence.eligible('model', 'algorithm'))
        evidence = ValidationEvidence('model', 'algorithm', {k: True for k in ValidationEvidence.REQUIRED})
        self.assertTrue(evidence.eligible('model', 'algorithm'))
        self.assertFalse(evidence.eligible('other', 'algorithm'))
    def test_basis_dimension_and_finite_data(self):
        with self.assertRaises(ValueError):
            BasisPack(np.eye(2), {'L': np.ones((3,1))}, {}, 'id')
        with self.assertRaises(ValueError):
            BasisPack(np.eye(2), {'L': np.array([[np.nan],[0]])}, {}, 'id')
