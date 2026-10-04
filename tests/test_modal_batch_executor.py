import unittest
import numpy as np
from modal_batch_executor import execute_batches,select_numpy_backend,execute_device_batches
from runtime_resources import ResourceInventory,resolve_policy

class ExecutorTests(unittest.TestCase):
    def test_dynamic_ordered_results_and_no_gpu_fallback(self):
        p=resolve_policy(ResourceInventory(4,2,10000,9000,[]))
        result=list(execute_batches(range(12),lambda x:x*x,p))
        self.assertEqual(result,[x*x for x in range(12)])
        xp,info=select_numpy_backend(p)
        self.assertIs(xp,np)
        self.assertEqual(info['backend'],'numpy_cpu')
    def test_worker_exception_propagates(self):
        p=resolve_policy(ResourceInventory(2,2,100,90,[]))
        def fail(x): raise ValueError('numerical failure')
        with self.assertRaises(ValueError): list(execute_batches([1],fail,p))

class DeviceQueueTests(unittest.TestCase):
    def test_every_device_scheduled_without_overlapping_same_device(self):
        import threading,time
        active=set();seen=set(); lock=threading.Lock()
        def work(x,device):
            with lock:
                self.assertNotIn(device,active);active.add(device);seen.add(device)
            time.sleep(.002)
            with lock:active.remove(device)
            return x*x
        result=dict(execute_device_batches(range(8),work,[0,1]))
        self.assertEqual([result[i] for i in range(8)],[i*i for i in range(8)])
        self.assertEqual(seen,{0,1})

class RetryTests(unittest.TestCase):
    def test_allocation_retry_halves_work_and_preserves_order(self):
        from modal_batch_executor import retry_projection
        calls=[]
        def op(x):
            calls.append(len(x))
            if len(x)>2:raise MemoryError('allocation')
            return [int(i)*2 for i in x]
        result,attempts=retry_projection(np.arange(7),op)
        self.assertEqual(result,[i*2 for i in range(7)])
        self.assertGreater(attempts,0)
        self.assertEqual(calls[0],7)
    def test_single_failure_is_explicit_and_can_fallback(self):
        from modal_batch_executor import retry_projection
        def op(x):raise MemoryError('one item')
        result,retries=retry_projection(np.arange(2),op,fallback=lambda x:[int(x[0])])
        self.assertEqual(result,[0,1]);self.assertGreater(retries,0)

class BlasBudgetTests(unittest.TestCase):
    def test_selected_blas_budget_is_applied_to_real_numeric_workers(self):
        from threadpoolctl import threadpool_info
        p=resolve_policy(ResourceInventory(4,4,100000,90000,[]),cpu_override=2)
        def work(index):
            result=np.ones((30,30))@np.ones((30,30))
            active=[entry['num_threads'] for entry in threadpool_info()
                    if entry['user_api']=='blas']
            return result,active
        for result,active in execute_batches(range(3),work,p,blas_threads=2):
            np.testing.assert_allclose(result,30.)
            self.assertTrue(active)
            self.assertTrue(all(n==2 for n in active))
