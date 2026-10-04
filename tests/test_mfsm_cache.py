import unittest,tempfile
import numpy as np
from mfsm_model import OperatorPack,SourceDefinition
from mfsm_reference_basis import cached_mfsm_basis

class CacheTests(unittest.TestCase):
    def test_cache_reuse_and_corrupt_cache_rejected(self):
        a=OperatorPack(np.eye(2),{'eps_x':np.diag([0.,1.])},SourceDefinition('p','35','auxiliary_nu_zero'))
        spaces={'L':{'zero_components':['eps_x'],'equations':'133','search_basis':np.eye(2)}}
        with tempfile.TemporaryDirectory() as root:
            x,info=cached_mfsm_basis(a,spaces,root)
            y,other=cached_mfsm_basis(a,spaces,root)
            self.assertFalse(info['hit']);self.assertTrue(other['hit'])
            np.testing.assert_allclose(x.spaces['L'],y.spaces['L'])
            with open(info['path'],'wb') as stream:stream.write(b'broken')
            with self.assertRaises(ValueError):cached_mfsm_basis(a,spaces,root)
