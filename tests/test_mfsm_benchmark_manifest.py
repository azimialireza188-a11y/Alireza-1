import unittest
from benchmarks.mfsm.run_validation import assess

class EvidenceTests(unittest.TestCase):
    def test_units_or_wrong_model_cannot_enable_primary(self):
        data={'algorithm':'mfsm-energy-operators-v1','model_hash':'model','checks':{'unit_tests':True}}
        self.assertFalse(assess(data,'model')['eligible'])
        data['checks']={k:True for k in ('source_derivation','published_benchmark','connection_limit','active_contact','abaqus_mesh','harmonic_convergence','eigenspace_match')}
        self.assertFalse(assess(data,'different')['eligible'])
    def test_assertions_without_measured_artifacts_are_pending(self):
        data={'algorithm':'mfsm-energy-operators-v1','model_hash':'model','checks':{}}
        self.assertEqual(assess(data,'model')['status'],'PENDING')
