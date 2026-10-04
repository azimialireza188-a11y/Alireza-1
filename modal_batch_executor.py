"""Dynamic CPU work queue and measured optional FP64 GPU backend selection."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import nullcontext
import time
import numpy as np
from runtime_resources import available_memory,batch_capacity


def execute_batches(batches,operation,policy,working_set_bytes=1):
    workers=min(policy.cpus,batch_capacity(available_memory()[1],working_set_bytes))
    try:
        from threadpoolctl import threadpool_limits
        limits=threadpool_limits(limits=1)
    except ImportError:
        # Avoid nested all-core BLAS when the installed runtime cannot control it.
        workers=1;limits=nullcontext()
    with limits,ThreadPoolExecutor(max_workers=workers) as pool:
        pending=deque(); iterator=iter(batches)
        for unused in range(workers):
            try: pending.append(pool.submit(operation,next(iterator)))
            except StopIteration: break
        while pending:
            yield pending.popleft().result()
            try: pending.append(pool.submit(operation,next(iterator)))
            except StopIteration: pass


def select_numpy_backend(policy,probe=None):
    info=dict(backend='numpy_cpu',device=None,gpu_timings_seconds=[],reason='NO_SUPPORTED_GPU')
    if not policy.gpus: return np,info
    try:
        import cupy as cp
        count=min(policy.gpus,cp.cuda.runtime.getDeviceCount())
        if not count: return np,info
        if probe is None:
            return np,dict(info,reason='GPU_AVAILABLE_BUT_NO_REPRESENTATIVE_TIMING')
        start=time.perf_counter();cpu_result=probe(np);cpu_seconds=time.perf_counter()-start
        timings=[]
        for device in range(count):
            with cp.cuda.Device(device):
                start=time.perf_counter();gpu_result=probe(cp);cp.cuda.Stream.null.synchronize()
                seconds=time.perf_counter()-start
                if not np.allclose(cpu_result,gpu_result,rtol=1e-9,atol=1e-11):
                    raise ValueError('GPU/CPU FP64 numerical mismatch')
                timings.append((seconds,device))
        fastest,device=min(timings)
        if fastest<cpu_seconds:
            return cp,dict(info,backend='cupy_gpu',device=device,
                           gpu_timings_seconds=timings,cpu_seconds=cpu_seconds,
                           faster_devices=[d for seconds,d in timings if seconds<cpu_seconds],reason='MEASURED_FASTER_FP64')
        return np,dict(info,gpu_timings_seconds=timings,cpu_seconds=cpu_seconds,reason='CPU_MEASURED_FASTER')
    except (ImportError,OSError,RuntimeError,MemoryError) as exc:
        return np,dict(info,reason='GPU_RUNTIME_UNAVAILABLE: '+str(exc))


def execute_device_batches(tasks, operation, device_ids):
    """One in-flight task per GPU; refill the device that completes first."""
    from concurrent.futures import wait, FIRST_COMPLETED
    devices=tuple(device_ids)
    if not devices or len(set(devices))!=len(devices):
        raise ValueError('Distinct device IDs required')
    iterator=iter(enumerate(tasks))
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        pending={}
        def submit(device):
            try: index, task=next(iterator)
            except StopIteration: return
            pending[pool.submit(operation,task,device)]=(index,device)
        for device in devices: submit(device)
        while pending:
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                index,device=pending.pop(future)
                yield index,future.result()
                submit(device)
