# -*- coding: utf-8 -*-
"""Resolve and run the compact Parquet exporter without requiring PyArrow in Abaqus."""
from __future__ import print_function
import json
import os
import shutil
import subprocess
import sys

import modal_analysis_parquet as bundle


def _probe_external(command):
    cmd=list(command)+['-c','import pyarrow; print(pyarrow.__version__)']
    try:
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,timeout=20)
    except Exception as exc:
        return dict(ok=False,error='%s: %s' % (type(exc).__name__,exc))
    if result.returncode:
        return dict(ok=False,error=(result.stderr or result.stdout or
                                    'external PyArrow probe failed').strip())
    return dict(ok=True,version=(result.stdout or '').strip())


def _candidate_commands(preferred=None):
    seen=set()
    if preferred:
        path=os.path.abspath(os.path.expanduser(preferred))
        yield [path]
        seen.add(os.path.normcase(path))
    for name in ('python','python3','py'):
        path=shutil.which(name)
        if not path:
            continue
        key=os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        if os.path.basename(path).lower() in ('py','py.exe'):
            yield [path,'-3']
        else:
            yield [path]


def find_runtime(preferred=None):
    current=bundle.runtime_probe()
    attempts=[dict(kind='in_process',python=current.get('python'),
                   available=bool(current.get('available')),
                   detail=current.get('pyarrow_version') or current.get('error'))]
    if current.get('available'):
        return dict(available=True,kind='in_process',
                    pyarrow_version=current.get('pyarrow_version'),
                    python=current.get('python'),attempts=attempts)
    for command in _candidate_commands(preferred):
        probe=_probe_external(command)
        attempts.append(dict(kind='external',command=command,
                             available=bool(probe.get('ok')),
                             detail=probe.get('version') or probe.get('error')))
        if probe.get('ok'):
            return dict(available=True,kind='external',command=command,
                        pyarrow_version=probe.get('version'),attempts=attempts)
    return dict(available=False,status='PARQUET_RUNTIME_UNAVAILABLE',
                attempts=attempts,
                installation_hint='Install PyArrow in normal Python: python -m pip install pyarrow')


def prepare_runtime(policy='auto',preferred=None):
    policy=str(policy or 'auto').lower()
    if policy not in ('auto','off','required'):
        raise ValueError('Parquet export policy must be auto, off or required')
    if policy=='off':
        return dict(available=False,policy='off',status='PARQUET_EXPORT_DISABLED')
    result=find_runtime(preferred)
    result['policy']=policy
    if not result.get('available') and policy=='required':
        raise RuntimeError(
            'Parquet export is required but no Python runtime with PyArrow was found. '
            'Install it before the Abaqus run with: python -m pip install pyarrow')
    return result


def export_run(run_dir,audit_dir,runtime,output_root=None,
               harmonic_section_min_share=.001):
    if not runtime or not runtime.get('available'):
        return dict(status='PARQUET_EXPORT_UNAVAILABLE',
                    reason=(runtime or {}).get('status','runtime unavailable'))
    if runtime.get('kind')=='in_process':
        result=bundle.export_run(
            run_dir,audit_dir,output_root,
            harmonic_section_min_share=harmonic_section_min_share)
    else:
        script=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'modal_analysis_parquet.py')
        command=list(runtime['command'])+[script,
            '--run-dir',os.path.abspath(run_dir),
            '--audit-dir',os.path.abspath(audit_dir),
            '--harmonic-section-min-share',str(float(harmonic_section_min_share))]
        if output_root:
            command.extend(['--output-root',os.path.abspath(output_root)])
        completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True)
        if completed.returncode:
            raise RuntimeError('External Parquet exporter failed: '+
                               (completed.stderr or completed.stdout or 'unknown error').strip())
        try:
            result=json.loads(completed.stdout)
        except Exception as exc:
            raise RuntimeError('External Parquet exporter returned invalid JSON: '+
                               completed.stdout[-2000:]) from exc
    result=dict(result)
    result['status']='EXPORTED'
    result['runtime']=dict((k,v) for k,v in runtime.items() if k!='attempts')
    return result
