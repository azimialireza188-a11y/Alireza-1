"""Read-only Abaqus ODB -> portable FP64 U/UR shards; no classification/solver.

Run with Abaqus Python. Portable NPZ readers need NumPy, not odbAccess.
Shard size is upload granularity, not a RAM reservation or worker cap.
Compression uses all visible logical CPUs; ODB field extraction stays serial.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
import numpy as np
from abaqus_mfsm_input import read_input
from runtime_resources import detect_resources, resolve_policy


def _file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def _stable_snapshot(path, check_odb_lock=False):
    path=Path(path)
    lock=path.with_suffix('.lck')
    def state():
        if check_odb_lock and lock.exists():raise ValueError('ODB writer lock exists: '+str(lock))
        s=path.stat()
        return (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino)
    before=state();digest=_file_hash(path);after=state()
    if before!=after:raise ValueError('Source changed while hashing: '+str(path))
    return dict(sha256=digest,state=after)


def export_modal_data(odb,model,output_dir,odb_sha256,modes_per_shard=8,source_guard=None):
    from abaqus_dsm_modal_audit import read_mapped_mode
    if type(modes_per_shard) is not int or modes_per_shard<1 or not odb_sha256:
        raise ValueError('Positive shard size and source ODB hash required')
    names={str(name).upper():name for name in odb.rootAssembly.instances}
    if set(names)!=set(model['instances']):raise ValueError('ODB/INP instances differ')
    coordinates=[];keys=[];node_keys=[]
    for name in model['instances']:
        inst=odb.rootAssembly.instances[names[name]]
        actual={int(n.label):np.asarray(n.coordinates,dtype=float) for n in inst.nodes}
        expected={label:xyz for (part,label),xyz in model['nodes'].items() if part==name}
        if set(actual)!=set(expected):raise ValueError('ODB/INP node labels differ')
        for label,xyz in expected.items():
            if not np.allclose(actual[label],xyz,rtol=0,atol=max(1e-5,1e-6*np.max(np.abs(xyz)))):
                raise ValueError('ODB/INP coordinates differ')
            coordinates.append(actual[label]);node_keys.append((names[name],label))
            keys.extend((names[name],label,dof) for dof in range(1,7))
        actual_elements={int(e.label):tuple(int(n) for n in e.connectivity) for e in inst.elements if str(e.type).upper()=='S4R'}
        expected_elements={label:tuple(n[1] for n in conn) for (part,label),conn in model['elements'].items() if part==name}
        if len(actual_elements)!=len(inst.elements) or actual_elements!=expected_elements:
            raise ValueError('ODB/INP S4R connectivity differs')
    if len(model['steps'])!=1 or 'buckle_data' not in model['steps'][0]:
        raise ValueError('Exactly one source BUCKLE step required')
    step_name=model['steps'][0]['name']
    matched=[name for name in odb.steps if str(name).upper()==step_name]
    if len(matched)!=1:raise ValueError('ODB/INP buckling step mismatch')
    frames=[f for f in odb.steps[matched[0]].frames if int(getattr(f,'mode',0))>0]
    ids=[int(f.mode) for f in frames]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Empty/duplicate buckling mode IDs')
    root=Path(output_dir)
    try:root.mkdir(parents=True,exist_ok=False)
    except FileExistsError as exc:raise ValueError('Use a new export directory') from exc
    policy=resolve_policy(detect_resources());start=time.perf_counter()
    report=dict(kind='PORTABLE_RAW_MODAL_DATA_ONLY',source_odb_sha256=odb_sha256,
        source_inp_sha256=model['input_sha256'],step=str(matched[0]),mode_count=len(frames),
        node_count=len(node_keys),raw_dof_count=len(keys),modes=[],shards=[],
        scientifically_eligible=False,default_classifier_activation=False,
        source_stability_verified=False,
        contact_state='UNKNOWN_REQUIRES_BASE_STATE_EVIDENCE',resources=policy.provenance(),
        modes_per_shard=modes_per_shard,processing_precision='FP64_PRESERVING_SOURCE_DATA',
        limitations=['No S4R/contact tangent or mechanical basis in this archive.',
                     'FP64 storage does not recover accuracy lost in source output.',
                     'Actual Abaqus runtime verification is separate from synthetic tests.'])
    with tempfile.TemporaryDirectory(prefix='modal-stage-',dir=root) as stage:
        np.savez_compressed(Path(stage,'raw_dof_map.npz'),instances=[k[0] for k in keys],
            labels=[k[1] for k in keys],dofs=[k[2] for k in keys],node_instances=[k[0] for k in node_keys],
            node_labels=[k[1] for k in node_keys],node_coordinates=np.asarray(coordinates),
            metadata=json.dumps(dict(source_inp_sha256=model['input_sha256'],source_odb_sha256=odb_sha256)))
        pending=set()
        def drain(all_tasks=False):
            nonlocal pending
            done,pending=wait(pending,return_when='ALL_COMPLETED' if all_tasks else FIRST_COMPLETED)
            for task in done:task.result()
        with ThreadPoolExecutor(max_workers=policy.cpus) as pool:
            for offset in range(0,len(frames),modes_per_shard):
                if len(pending)>=policy.cpus:drain()
                selected=frames[offset:offset+modes_per_shard]
                while True:
                    try:
                        vectors=np.empty((len(keys),len(selected)),dtype=np.float64);break
                    except MemoryError:
                        if not pending:raise
                        drain()  # Actual allocation failure; no standing RAM reserve.
                for j,frame in enumerate(selected):
                    match=re.search(r'EigenValue\s*=\s*([-+0-9.EeDd]+)',frame.description,re.I)
                    if not match:raise ValueError('Missing eigenvalue in mode '+str(frame.mode))
                    eigenvalue=float(match.group(1).replace('D','E').replace('d','e'))
                    if not np.isfinite(eigenvalue):raise ValueError('Nonfinite eigenvalue')
                    # Acquire each native ODB value sequence once, then reuse
                    # it for mapping and precision provenance.
                    if any(field not in frame.fieldOutputs for field in ('U','UR')):
                        raise ValueError('Required U/UR field missing from ODB')
                    cached_fields={field:SimpleNamespace(values=frame.fieldOutputs[field].values) for field in ('U','UR')}
                    vectors[:,j]=read_mapped_mode(SimpleNamespace(fieldOutputs=cached_fields),keys)
                    report['modes'].append(dict(mode=int(frame.mode),eigenvalue=eigenvalue,
                        fields=sorted(frame.fieldOutputs.keys()),
                        source_precision={field:sorted({str(v.precision) for v in cached_fields[field].values})
                                          for field in ('U','UR')}))
                filename='modes_%04d.npz'%(len(report['shards'])+1)
                report['shards'].append(filename)
                pending.add(pool.submit(np.savez_compressed,Path(stage,filename),vectors=vectors,
                    modes=np.array([int(f.mode) for f in selected]),eigenvalues=np.array([r['eigenvalue'] for r in report['modes'][-len(selected):]])))
            if pending:drain(True)
        artifacts=['raw_dof_map.npz']+report['shards']
        report['artifact_sha256']={name:_file_hash(Path(stage,name)) for name in artifacts}
        if source_guard is not None:
            source_guard()  # Must succeed BEFORE publishing artifacts/manifest.
            report['source_stability_verified']=True
        report['elapsed_seconds']=time.perf_counter()-start
        Path(stage,'modal_export.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        for name in artifacts+['modal_export.json']:os.replace(Path(stage,name),root/name)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--odb',required=True);p.add_argument('--inp',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--modes-per-shard',type=int,default=8)
    args=p.parse_args()
    initial_odb=_stable_snapshot(args.odb,True);initial_inp=_stable_snapshot(args.inp)
    model=read_input(args.inp)
    if model['input_sha256']!=initial_inp['sha256']:raise ValueError('INP source changed during parsing')
    def guard():
        if _stable_snapshot(args.odb,True)!=initial_odb:raise ValueError('ODB source changed during export')
        if _stable_snapshot(args.inp)!=initial_inp:raise ValueError('INP source changed during export')
    from odbAccess import openOdb
    odb=openOdb(path=os.path.abspath(args.odb),readOnly=True)
    try:r=export_modal_data(odb,model,args.output_dir,initial_odb['sha256'],args.modes_per_shard,source_guard=guard)
    finally:odb.close()
    print(json.dumps(dict(mode_count=r['mode_count'],shards=r['shards'],output_dir=args.output_dir,
                         scientifically_eligible=False)))


if __name__=='__main__':main()
