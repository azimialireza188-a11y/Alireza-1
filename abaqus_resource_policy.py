# -*- coding: utf-8 -*-
"""Aggressive, capability-driven runtime resource policy."""
from __future__ import print_function
import os
import subprocess


def detect_gpus():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, universal_newlines=True, timeout=5)
    except Exception:
        return []
    devices = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",", 1)]
        try:
            index = int(parts[0])
        except (ValueError, IndexError):
            continue
        devices.append(dict(index=index, name=parts[1] if len(parts) > 1 else "GPU%d" % index))
    return devices


def detect_memory_bytes():
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            if pages and size:
                return int(pages) * int(size)
    except Exception:
        pass
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            return int(state.ullTotalPhys)
    except Exception:
        pass
    return None


def detect_resources():
    cpus = int(os.cpu_count() or 1)
    gpus = detect_gpus()
    return dict(logical_cpus=cpus, memory_bytes=detect_memory_bytes(),
                gpus=gpus, gpu_count=len(gpus))


def _largest_divisor_at_most(n, limit):
    limit = max(1, min(int(limit), int(n)))
    for value in range(limit, 0, -1):
        if n % value == 0:
            return value
    return 1


def worker_layout(logical_cpus, work_items):
    cores = max(1, int(logical_cpus))
    items = max(1, int(work_items or 1))
    processes = _largest_divisor_at_most(cores, min(cores, items))
    threads = max(1, cores // processes)
    return dict(processes=processes, blas_threads=threads,
                aggregate_threads=processes * threads)


def resolve_resource_plan(requested_cpus=None, requested_gpus=None, work_items=None):
    detected = detect_resources()
    cpus = detected["logical_cpus"] if requested_cpus in (None, "auto", "all") else int(requested_cpus)
    if cpus < 1:
        raise ValueError("requested_cpus must be positive or auto")
    gpu_count = detected["gpu_count"]
    if requested_gpus in (None, "auto", "all"):
        gpus = gpu_count
    else:
        gpus = int(requested_gpus)
        if gpus < 0:
            raise ValueError("requested_gpus must be nonnegative or auto")
        gpus = min(gpus, gpu_count)
    layout = worker_layout(cpus, work_items or cpus)
    return dict(
        logical_cpus_detected=detected["logical_cpus"],
        memory_bytes_detected=detected["memory_bytes"],
        gpus_detected=gpu_count,
        gpu_devices=detected["gpus"],
        cpus=cpus,
        gpus=gpus,
        memory_percent=100,
        worker_layout=layout,
        gpu_backend=("cuda-detected" if gpus else "cpu"),
        policy="aggressive_no_artificial_reserve")


def apply_blas_thread_env(layout):
    threads = str(max(1, int(layout["blas_threads"])))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = threads
    return threads
