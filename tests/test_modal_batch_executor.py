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
