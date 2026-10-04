import unittest
from unittest import mock
import runtime_resources as r
import abaqus_complete_model_m20 as builder

class ResourceTests(unittest.TestCase):
    def test_zero_reserve_full_request(self):
        inv = r.ResourceInventory(24, 12, 64*1024**3, 50*1024**3, [{'id':'0','free_bytes':10*1024**3}, {'id':'1','free_bytes':8*1024**3}])
        p = r.resolve_policy(inv)
        self.assertEqual(p.cpus, 24)
        self.assertEqual(p.gpus, 2)
        self.assertEqual(p.memory_reserve_bytes, 0)
        self.assertEqual(p.gpu_reserve_bytes, 0)
        self.assertEqual(r.abaqus_job_settings(p, {'numGPUs':True})['memory'], 100)
        self.assertFalse(r.abaqus_job_settings(p, {})['getMemoryFromAnalysis'])
        self.assertNotIn('numGPUs', r.abaqus_job_settings(p, {}))
    def test_explicit_overrides_and_no_gpu(self):
        inv = r.ResourceInventory(24,12,100,90,[])
        self.assertEqual(r.resolve_policy(inv, 8, '0').cpus, 8)
        self.assertEqual(r.resolve_policy(inv).gpus, 0)
        with self.assertRaises(ValueError): r.resolve_policy(inv,0)
    def test_builder_auto_and_legacy_cpus(self):
        with mock.patch.object(r, 'detect_resources', return_value=r.ResourceInventory(24,12,100,90,[])):
            self.assertEqual(builder.parse_arguments([]).cpus,24)
            self.assertEqual(builder.parse_arguments(['--cpus','8']).cpus,8)
            self.assertEqual(builder.parse_arguments(['--cpus','auto','--gpus','0']).gpus,'0')
    def test_batch_capacity_has_no_safety_factor(self):
        self.assertEqual(r.batch_capacity(1000,100,250),7)
        with self.assertRaises(MemoryError): r.batch_capacity(50,100,0)
    def test_builder_arguments_are_json_serializable(self):
        import json
        json.dumps(vars(builder.parse_arguments([])))

    def test_threads_configuration(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            r.configure_threads(24)
            import os
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'24')
    def test_observed_memory_is_finite_and_not_an_allocation_request(self):
        stats=r.process_memory()
        self.assertGreater(stats['peak_rss_bytes'],0)
        if stats['current_rss_bytes'] is not None:self.assertGreaterEqual(stats['peak_rss_bytes'],stats['current_rss_bytes'])
        self.assertEqual(stats['scope'],'CURRENT_PROCESS_LIFETIME')
    def test_live_gpu_batch_uses_exact_free_bytes_and_working_sets(self):
        from modal_batch_executor import device_batch_capacity
        self.assertEqual(device_batch_capacity(1000,100,200),8)
        with self.assertRaises(MemoryError):device_batch_capacity(250,100,200)
