# -*- coding: utf-8 -*-
"""Reproducible pipeline provenance and timing report for Abaqus modal runs."""
from __future__ import print_function
import csv
import datetime
import hashlib
import json
import os
import ntpath
import platform
import socket
import subprocess
import sys
import zipfile


FORMAT_VERSION='pipeline_run_report_v1'


def _iso(epoch):
    if epoch is None:
        return None
    try:
        return datetime.datetime.fromtimestamp(float(epoch)).astimezone().isoformat()
    except Exception:
        return datetime.datetime.fromtimestamp(float(epoch)).isoformat()


def _sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _git_head(script_dir):
    directory=os.path.abspath(script_dir)
    try:
        value=subprocess.check_output(
            ['git','-C',directory,'rev-parse','HEAD'],
            stderr=subprocess.STDOUT).decode('ascii','replace').strip()
        return value or None
    except Exception:
        return None


def _absolute_path(value):
    text=str(value)
    if len(text)>=3 and text[1]==':' and text[2] in ('\\','/'):
        return ntpath.normpath(text)
    return os.path.abspath(text)


def capture_invocation(script_path,effective_args,cwd=None):
    """Capture the reproducible command after CMD/Abaqus have tokenized it.

    CMD caret continuations and the user's exact whitespace/quoting are not
    recoverable after shell parsing, so both process argv and a normalized
    executable command are persisted.
    """
    script_path=_absolute_path(script_path)
    args=[str(x) for x in list(effective_args or [])]
    command=['abaqus','cae','noGUI='+ntpath.basename(script_path),'--']+args
    normalized=subprocess.list2cmdline(command)
    launch_cwd=_absolute_path(cwd or os.getcwd())
    script_sha=None
    try:
        script_sha=_sha256(script_path)
    except Exception:
        pass
    return dict(
        normalized_command=normalized,
        normalized_command_tokens=command,
        effective_args=args,
        process_argv=[str(x) for x in sys.argv],
        launch_cwd=launch_cwd,
        script_path=script_path,
        script_sha256=script_sha,
        git_commit=_git_head(os.path.dirname(script_path)),
        host=socket.gethostname(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=sys.version,
        python_executable=sys.executable,
        process_id=os.getpid(),
        environment=dict(
            NUMBER_OF_PROCESSORS=os.environ.get('NUMBER_OF_PROCESSORS'),
            OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),
            MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),
            OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS')))


def _tracker_epoch_map(tracker,name):
    value=getattr(tracker,name,{}) or {}
    return dict(value)


def stage_rows(tracker,audit_summary=None,now_epoch=None):
    now=float(now_epoch if now_epoch is not None else __import__('time').time())
    names=list(getattr(tracker,'stage_names',[]) or [])
    weights=list(getattr(tracker,'weights',[]) or [])
    completed=set(getattr(tracker,'completed',set()) or set())
    current_index=getattr(tracker,'stage_index',None)
    current_stage=getattr(tracker,'stage',None)
    current_started=getattr(tracker,'stage_started',None)
    durations=dict(getattr(tracker,'stage_durations',{}) or {})
    starts=_tracker_epoch_map(tracker,'stage_started_epochs')
    finishes=_tracker_epoch_map(tracker,'stage_finished_epochs')
    rows=[]
    for index,stage in enumerate(names):
        if index in completed:
            status='COMPLETED'
            duration=durations.get(stage)
        elif current_index==index and current_stage==stage:
            status='RUNNING'
            start=starts.get(stage,current_started)
            duration=(None if start is None else max(0.0,now-float(start)))
        else:
            status='PENDING'
            duration=None
        rows.append(dict(
            scope='PIPELINE',parent_stage=None,stage=stage,status=status,
            weight_percent=(100.0*float(weights[index]) if index<len(weights) else None),
            duration_seconds=(None if duration is None else float(duration)),
            started_at=_iso(starts.get(stage)),
            finished_at=_iso(finishes.get(stage))))
    progress=((audit_summary or {}).get('basis_metadata') or {}).get('progress_timing') or {}
    nested=progress.get('stage_durations_seconds') or {}
    for stage,duration in nested.items():
        rows.append(dict(
            scope='MODAL_AUDIT',parent_stage='MODAL_AUDIT',stage=str(stage),
            status='COMPLETED',weight_percent=None,duration_seconds=float(duration),
            started_at=None,finished_at=None))
    return rows


def _discover_audit_summary(output_dir):
    candidates=[]
    try:
        names=os.listdir(output_dir)
    except Exception:
        return None
    for name in names:
        path=os.path.join(output_dir,name)
        if not (os.path.isdir(path) and name.startswith('modal_dsm_audit')):
            continue
        report=os.path.join(path,'modal_audit.json')
        if os.path.isfile(report):
            candidates.append(report)
    if not candidates:
        return None
    path=max(candidates,key=os.path.getmtime)
    try:
        with open(path,encoding='utf-8') as stream:
            return json.load(stream)
    except Exception:
        return None


def _output_inventory(output_dir):
    result={}
    for root,unused_dirs,files in os.walk(output_dir):
        for filename in files:
            if filename in ('pipeline_run_report.json','pipeline_stage_timings.csv'):
                continue
            path=os.path.join(root,filename)
            rel=os.path.relpath(path,output_dir)
            try:
                result[rel]=dict(bytes=int(os.path.getsize(path)),
                                 modified_at=_iso(os.path.getmtime(path)))
            except OSError:
                continue
    return result


def _resource_plan(state):
    settings=(state or {}).get('settings') or {}
    return settings.get('resource_plan') or {}


def write_report(output_dir,tracker,invocation,state,started_epoch,status,
                 now_epoch=None,audit_summary=None):
    output_dir=os.path.abspath(output_dir)
    os.makedirs(output_dir,exist_ok=True)
    now=float(now_epoch if now_epoch is not None else __import__('time').time())
    audit=audit_summary if audit_summary is not None else _discover_audit_summary(output_dir)
    stages=stage_rows(tracker,audit_summary=audit,now_epoch=now)
    settings=(state or {}).get('settings') or {}
    progress={}
    try:
        progress=tracker.summary()
    except Exception:
        progress=(state or {}).get('progress') or {}
    report=dict(
        format=FORMAT_VERSION,
        status=str(status),
        started_at=_iso(started_epoch),
        updated_at=_iso(now),
        total_elapsed_seconds=max(0.0,now-float(started_epoch)),
        invocation=invocation or {},
        resource_plan=_resource_plan(state),
        effective_settings=settings,
        submitted=bool((state or {}).get('submitted',False)),
        solver_status=(state or {}).get('solver_status'),
        solver_api_status=(state or {}).get('solver_api_status'),
        completion_evidence=(state or {}).get('completion_evidence'),
        progress=progress,
        stages=stages,
        outputs=_output_inventory(output_dir))
    json_path=os.path.join(output_dir,'pipeline_run_report.json')
    csv_path=os.path.join(output_dir,'pipeline_stage_timings.csv')
    replay_path=os.path.join(output_dir,'pipeline_replay.cmd')
    with open(json_path,'w',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
    fields=['scope','parent_stage','stage','status','weight_percent',
            'duration_seconds','started_at','finished_at']
    with open(csv_path,'w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in stages:
            writer.writerow({key:row.get(key) for key in fields})
    launch=(invocation or {}).get('launch_cwd') or output_dir
    command=(invocation or {}).get('normalized_command') or ''
    with open(replay_path,'w',encoding='utf-8') as stream:
        stream.write('@echo off\n')
        stream.write('cd /d "'+str(launch).replace('"','""')+'"\n')
        stream.write(str(command)+'\n')
    return dict(json_path=json_path,csv_path=csv_path,
                replay_path=replay_path,report=report)


def append_final_report_to_zip(zip_path,json_path,csv_path,replay_path=None):
    """Replace/add final normal timing reports inside the upload ZIP."""
    zip_path=os.path.abspath(zip_path)
    temp=zip_path+'.tmp'
    replace={'pipeline_run_report.json':json_path,
             'pipeline_stage_timings.csv':csv_path}
    if replay_path:
        replace['pipeline_replay.cmd']=replay_path
    with zipfile.ZipFile(zip_path,'r') as source, \
         zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as target:
        for item in source.infolist():
            if item.filename in replace:
                continue
            target.writestr(item,source.read(item.filename))
        for arcname,path in replace.items():
            target.write(path,arcname=arcname)
    os.replace(temp,zip_path)
    return zip_path
