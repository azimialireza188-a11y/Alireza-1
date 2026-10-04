import unittest
import numpy as np
from mfsm_model import OperatorPack,SourceDefinition
from builtup_connection_operator import beam_mpc_matrix, reduce_constraints, assemble_connections

class ConnectionTests(unittest.TestCase):
    def pack(self): return OperatorPack(np.eye(2),{'eps_x':np.eye(2)},SourceDefinition('paper','eq35','auxiliary_nu_zero'))
    def test_beam_offset_rigid_motion(self):
        c=beam_mpc_matrix([0,0,0],[2,0,0])
        u=np.zeros(12); u[5]=1; u[7]=2;u[11]=1
        np.testing.assert_allclose(c@u,0,atol=1e-14)
    def test_unknown_contact_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'CONTACT'):
            reduce_constraints(self.pack(),np.array([[1.,-1.]]),{'status':'unknown'})
    def test_connector_not_added_to_family_operator(self):
        a=self.pack(); b=assemble_connections(a, np.array([[2.,-2.],[-2.,2.]]),'elastic-test')
        np.testing.assert_allclose(b.components['eps_x'],a.components['eps_x'])
        self.assertFalse(np.array_equal(a.system,b.system))
    def test_finite_connection_limit_matches_exact_reduction(self):
        a=self.pack(); c=np.array([[1.,-1.]])
        reduced,q=reduce_constraints(a,c,{'status':'inactive','evidence':'analytical example'})
        exact=q@np.linalg.solve(reduced.system,q.T@np.array([1.,0.]))
        errors=[np.linalg.norm(np.linalg.solve(a.system+k*c.T@c,[1.,0.])-exact) for k in (1e2,1e4,1e6)]
        self.assertTrue(errors[2]<errors[1]<errors[0])
        self.assertLess(errors[2],1e-6)
