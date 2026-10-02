import unittest
from unittest import mock
import abaqus_resource_policy as r


class ResourcePolicyTests(unittest.TestCase):
    def test_auto_uses_all_detected_logical_cpus_and_no_memory_reserve(self):
        with mock.patch.object(r.os, 'cpu_count', return_value=24), \
             mock.patch.object(r, 'detect_gpus', return_value=[]):
            plan = r.resolve_resource_plan(work_items=71)
        self.assertEqual(plan['logical_cpus_detected'], 24)
        self.assertEqual(plan['cpus'], 24)
        self.assertEqual(plan['memory_percent'], 100)
        self.assertEqual(plan['worker_layout']['processes'] * plan['worker_layout']['blas_threads'], 24)

    def test_explicit_cpu_cap_is_honored(self):
        with mock.patch.object(r.os, 'cpu_count', return_value=24), \
             mock.patch.object(r, 'detect_gpus', return_value=[]):
            plan = r.resolve_resource_plan(requested_cpus=8, work_items=100)
        self.assertEqual(plan['cpus'], 8)
        self.assertEqual(plan['worker_layout']['processes'] * plan['worker_layout']['blas_threads'], 8)

    def test_worker_layout_uses_full_capacity_without_nested_oversubscription(self):
        for cores, work in ((24, 71), (24, 5), (12, 2), (7, 100)):
            with self.subTest(cores=cores, work=work):
                layout = r.worker_layout(cores, work)
                self.assertEqual(layout['processes'] * layout['blas_threads'], cores)
                self.assertLessEqual(layout['processes'], max(1, work))

    def test_gpu_backend_absence_falls_back_to_cpu(self):
        with mock.patch.object(r, 'detect_gpus', return_value=[]):
            plan = r.resolve_resource_plan(work_items=10)
        self.assertEqual(plan['gpus'], 0)
        self.assertEqual(plan['gpu_backend'], 'cpu')

    def test_auto_gpu_uses_all_detected_supported_devices(self):
        devices = [{'index': 0, 'name': 'GPU0'}, {'index': 1, 'name': 'GPU1'}]
        with mock.patch.object(r.os, 'cpu_count', return_value=8), \
             mock.patch.object(r, 'detect_gpus', return_value=devices):
            plan = r.resolve_resource_plan(work_items=8)
        self.assertEqual(plan['gpus_detected'], 2)
        self.assertEqual(plan['gpus'], 2)


if __name__ == '__main__':
    unittest.main()
