"""Verified portable-mode input and automatic source CPT mechanical preparation.

No L/D/G label is assigned by this upstream preparation step. Unknown active
contact, unvalidated curved/CPT/S4R equivalence and coordinate mappings retain
UNAVAILABLE. Auxiliary component energies are NOT family participations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from contextlib import nullcontext
import hashlib,json,multiprocessing,os,re,threading,time,tempfile
from pathlib import Path
import numpy as np
from scipy.sparse import block_diag
from prismatic_mfsm_operators import (extruded_reference,strip_operators,CanonicalFitter,
    COMPONENTS,transverse_rotation_map,constant_warping_operator,ConstraintProjector)
from runtime_resources import detect_resources,resolve_policy,process_memory,available_memory,batch_capacity,configure_threads
from modal_batch_executor import select_numpy_backend,execute_batches,execute_device_batches
from harmonic_constraint_mapping import compile_harmonic_constraints


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


class PortableParquetModes:
    def __init__(self,root,model,policy=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.root=Path(root);self.policy=policy or resolve_policy(detect_resources());pa.set_cpu_count(self.policy.cpus)
        self.manifest_hash=file_hash(self.root/'modal_export.json')
        r=json.loads((self.root/'modal_export.json').read_text());self.manifest=r
        if r.get('kind')!='PORTABLE_RAW_MODAL_DATA_ONLY' or r.get('format')!='parquet' or r.get('source_inp_sha256')!=model['input_sha256']:
            raise ValueError('Supported Parquet export and matching source INP hash required')
        if not r.get('rotations_available') or not re.fullmatch('[a-fA-F0-9]{64}',r.get('source_odb_sha256','')):
            raise ValueError('Actual rotations and source ODB hash required')
        expected={'mesh_nodes.parquet','raw_dof_map.parquet','elements.parquet','modes.parquet'}
        for s in r['shards']:
            name=s['parquet_file']
            if not re.fullmatch(r'mode_shapes_[0-9]{4}\.parquet',name) or name in expected:raise ValueError('Invalid/duplicate shard filename')
            expected.add(name)
        if set(r['artifacts'])!=expected or len(r['artifacts'])!=len(expected) or set(r['artifact_sha256'])!=expected:
            raise ValueError('Exact known portable artifact inventory required')
        self.hashes=dict(r['artifact_sha256']);self.ensure_unchanged()
        self.reference=extruded_reference(model)
        nodes=pq.read_table(self.root/'mesh_nodes.parquet').to_pydict();n=len(nodes['label']);self.n=n
        if n!=r['node_count'] or nodes['node_index']!=list(range(n)):raise ValueError('Invalid node index/count')
        self.nodekeys=[(str(a).upper(),int(b)) for a,b in zip(nodes['instance'],nodes['label'])]
        if len(set(self.nodekeys))!=n or set(self.nodekeys)!=set(model['nodes']):raise ValueError('ODB/INP node topology differs')
        actual=np.column_stack([nodes[k] for k in ('x','y','z')]);expected=np.array([model['nodes'][k] for k in self.nodekeys])
        if not np.all(np.isfinite(actual)) or np.any(np.abs(actual-expected).max(axis=1)>np.maximum(1e-5,1e-6*np.abs(expected).max(axis=1))):raise ValueError('ODB/INP coordinates differ')
        lookup={k:i for i,k in enumerate(self.nodekeys)}
        self.grid=np.array([[lookup[k] for k in row] for row in self.reference['node_keys']])
        dofs=pq.read_table(self.root/'raw_dof_map.parquet').to_pydict()
        if (r['raw_dof_count']!=n*6 or dofs['raw_index']!=list(range(n*6)) or
            dofs['node_index']!=np.repeat(np.arange(n),6).tolist() or dofs['dof']!=np.tile(np.arange(1,7),n).tolist()):
            raise ValueError('Complete node-major six-DOF map required')
        elements=pq.read_table(self.root/'elements.parquet').to_pydict();got={}
        for i,(part,label) in enumerate(zip(elements['instance'],elements['label'])):
            key=(str(part).upper(),int(label))
            if key in got:raise ValueError('Duplicate element')
            got[key]=tuple((key[0],int(elements['n'+str(j)][i])) for j in range(1,5))
        if got!=model['elements'] or len(got)!=r['s4r_element_count']:raise ValueError('ODB/INP S4R connectivity differs')
        modes=pq.read_table(self.root/'modes.parquet').to_pydict();ids=modes['mode']
        if not ids or len(set(ids))!=len(ids) or any(type(i) is not int or i<1 for i in ids):raise ValueError('Unique positive mode IDs required')
        if (len(ids)!=r['mode_count'] or ids!=[v['mode'] for v in r['modes']] or ids!=[v for s in r['shards'] for v in s['modes']] or
            modes['eigenvalue']!=[v['eigenvalue'] for v in r['modes']] or not np.all(np.isfinite(modes['eigenvalue']))):raise ValueError('Mode/eigenvalue manifest differs')
        if any(set(v['fields'])!={'U','UR'} and not {'U','UR'}.issubset(v['fields']) for v in r['modes']):raise ValueError('U and UR required for every mode')
        self.eigenvalues=dict(zip(ids,modes['eigenvalue']))

    def ensure_unchanged(self):
        if file_hash(self.root/'modal_export.json')!=self.manifest_hash:raise ValueError('Portable manifest changed')
        def check(item):
            name,digest=item
            if file_hash(self.root/name)!=digest:raise ValueError('Portable artifact hash differs: '+name)
        with ThreadPoolExecutor(max_workers=self.policy.cpus) as pool:list(pool.map(check,self.hashes.items()))

    def read_shard(self,index,modes=None):
        import pyarrow.parquet as pq
        s=self.manifest['shards'][index];selected=list(s['modes'] if modes is None else modes)
        if not selected or len(set(selected))!=len(selected) or any(m not in s['modes'] for m in selected):raise ValueError('Nonempty distinct shard mode subset required')
        path=self.root/s['parquet_file'];schema=pq.read_schema(path);meta=schema.metadata or {}
        if (meta.get(b'vector_coordinate_system')!=b'GLOBAL' or meta.get(b'source_odb_sha256',b'').decode()!=self.manifest['source_odb_sha256'] or
            meta.get(b'modes',b'').decode()!=','.join(map(str,s['modes']))):raise ValueError('Shard coordinate/source/mode declaration differs')
        columns={f'm{mode:04d}_{d}' for mode in s['modes'] for d in ('u1','u2','u3','ur1','ur2','ur3')}
        if set(schema.names)!=columns|{'node_index'}:raise ValueError('Missing/extra modal U/UR columns')
        wanted=['node_index']+[f'm{m:04d}_{d}' for m in selected for d in ('u1','u2','u3','ur1','ur2','ur3')]
        table=pq.read_table(path,columns=wanted)
        if table.num_rows!=self.n or table.column('node_index').to_pylist()!=list(range(self.n)):raise ValueError('Shard node ordering differs')
        raw=np.empty((self.n*6,len(selected)),dtype=np.float64)
        for j,mode in enumerate(selected):
            for dof,d in enumerate(('u1','u2','u3','ur1','ur2','ur3')):
                name=f'm{mode:04d}_{d}'
                if str(schema.field(name).type) not in ('float','double'):raise ValueError('Unsupported nodal precision')
                raw[dof::6,j]=table.column(name).to_numpy()
        if not np.all(np.isfinite(raw)):raise ValueError('Nonfinite U/UR data')
        allsix=raw.reshape(self.n,6,-1)[self.grid]
        return dict(modes=selected,eigenvalues=[self.eigenvalues[m] for m in selected],raw=raw,
            raw_grid=allsix,canonical=allsix[:,:,[0,2,1,5],:])


def physical_material(model):
    if len(model['elastic'])!=1 or len(model['sections'])!=1 or 'PLASTIC' in model['keywords']:
        raise ValueError('One homogeneous isotropic elastic centered shell section required')
    elastic=model['elastic'][0];section=model['sections'][0]
    if (elastic['definition'].get('TYPE','ISOTROPIC').upper()!='ISOTROPIC' or
        set(elastic['definition'])-{'TYPE'} or set(section['definition'])-{'ELSET','MATERIAL','OFFSET'} or
        float(section['definition'].get('OFFSET','0'))!=0 or len(elastic['data'])!=1 or len(section['data'])!=1):
        raise ValueError('Unsupported material/section definition')
    values=[float(v) for v in elastic['data'][0].split(',') if v.strip()]
    thickness=[float(v) for v in section['data'][0].split(',') if v.strip()]
    if len(values)!=2 or len(thickness) not in (1,2):raise ValueError('Simple homogeneous E,nu and thickness required')
    young,nu=values;t=thickness[0]
    if not np.all(np.isfinite([young,nu,t])) or min(young,t)<=0 or not -1<nu<.5:raise ValueError('Invalid physical material')
    return dict(E=young,nu=nu,thickness=t)


def _single_harmonic(args):
    xy,edges,term,length,young,thickness=args
    return strip_operators(xy,edges,[term],length,young,thickness)['components']


def _child_threads():
    configure_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        global _thread_control
        _thread_control=threadpool_limits(limits=1)
    except ImportError:pass


class PreparationKernel:
    def __init__(self,reference,harmonic_counts,young,thickness,policy=None,constraints=None,raw_keys=None):
        self.reference=reference;self.counts=list(harmonic_counts);self.thickness=float(thickness);self.lock=threading.Lock();self.device_cache={}
        if (not self.counts or any(type(c) is not int or c<1 for c in self.counts) or
                self.counts!=sorted(set(self.counts)) or self.counts[-1]>len(reference['stations'])-2):raise ValueError('Increasing station-resolved harmonic counts required')
        r=reference;terms=list(range(1,self.counts[-1]+1));start=time.perf_counter()
        args=[(r['xy'],r['edges'],term,r['length'],young,thickness) for term in terms]
        if policy and policy.cpus>1 and len(args)>1:
            with ProcessPoolExecutor(max_workers=policy.cpus,mp_context=multiprocessing.get_context('spawn'),initializer=_child_threads) as pool:blocks=list(pool.map(_single_harmonic,args))
        else:blocks=list(map(_single_harmonic,args))
        self.components={k:block_diag([b[k] for b in blocks],format='csr') for k in COMPONENTS}
        self.fits={};self.rotations={};self.level_components={}
        self.constant_operator=constant_warping_operator(r['xy'],r['edges'],r['length'],young,thickness)
        self.constant_rotation=transverse_rotation_map(r['xy'],r['edges'],[1],r['length'])[:,1::4]
        for count in self.counts:
            self.fits[count]=CanonicalFitter(r['stations'],list(range(1,count+1)),r['length'],r['origin'],rotation_length=thickness,include_constant_warping=True)
            self.rotations[count]=transverse_rotation_map(r['xy'],r['edges'],list(range(1,count+1)),r['length'])
            n=count*len(r['xy'])*4
            self.level_components[count]={k:v[:n,:n] for k,v in self.components.items()}
        self.constraint_maps={};self.constraint_metadata=None
        if (constraints is None)!=(raw_keys is None):raise ValueError('Constraints and raw keys must be supplied together')
        if constraints is not None:
            full,self.constraint_metadata=compile_harmonic_constraints(constraints,raw_keys,r,terms)
            constant_columns=np.arange(4*len(r['xy'])*self.counts[-1],full.shape[1])
            for count in self.counts:
                columns=np.concatenate([np.arange(4*len(r['xy'])*count),constant_columns])
                self.constraint_maps[count]=full[:,columns]
        self.build_seconds=time.perf_counter()-start

    def _arrays(self,count,xp):
        if xp is np:return self.level_components[count],self.rotations[count],self.constant_operator,self.constant_rotation,self.constraint_maps.get(count)
        key=(int(xp.cuda.runtime.getDevice()),count)
        with self.lock:
            if key not in self.device_cache:
                from cupyx.scipy.sparse import csr_matrix as gpu_csr
                constraint=self.constraint_maps.get(count)
                self.device_cache[key]=({k:gpu_csr(v) for k,v in self.level_components[count].items()},gpu_csr(self.rotations[count]),gpu_csr(self.constant_operator),gpu_csr(self.constant_rotation),
                    None if constraint is None else gpu_csr(constraint))
            return self.device_cache[key]

    def run(self,batch,xp=np):
        # Resident fields shared across all harmonic levels on a device.
        fields=xp.asarray(batch['canonical']);raw=xp.asarray(batch['raw_grid']);history=[];r=self.reference
        for count in self.counts:
            fit=self.fits[count].fit(fields,xp=xp,host_output=False);a=fit['coefficients'];v=a.reshape(count*len(r['xy'])*4,-1)
            matrices,rotation,constant_matrix,constant_rotation,constraint=self._arrays(count,xp)
            constant=fit['constant_warping']
            energies=xp.stack([.5*xp.sum(v*(matrices[k]@v),axis=0) for k in COMPONENTS])
            energies[2]+=.5*xp.sum(constant*(constant_matrix@constant),axis=0)
            cosine=xp.asarray(self.fits[count].design['cos'][:,1:]);sine=xp.asarray(self.fits[count].design['sin'])
            predicted=xp.empty_like(raw)
            for canonical,physical,design in [(0,0,sine),(1,2,cosine),(2,1,sine),(3,5,sine)]:
                predicted[:,:,physical,:]=(design@a[:,:,canonical,:].reshape(count,-1)).reshape(len(r['stations']),len(r['xy']),-1)
            predicted[:,:,2,:]+=constant[None,:,:]
            constant_rotations=(constant_rotation@constant).reshape(len(r['xy']),2,-1)
            rotations=(rotation@v).reshape(count,len(r['xy']),2,-1)
            for i in range(2):predicted[:,:,3+i,:]=(cosine@rotations[:,:,i,:].reshape(count,-1)).reshape(len(r['stations']),len(r['xy']),-1)+constant_rotations[None,:,i,:]
            metric=xp.asarray([1,1,1,self.thickness,self.thickness,self.thickness])[None,None,:,None]*xp.asarray(np.sqrt(self.fits[count].design['weights']))[:,None,None,None]
            observed=raw*metric;difference=(raw-predicted)*metric
            scale=xp.max(xp.abs(observed),axis=(0,1,2));scale=xp.where(scale>0,scale,1.)
            den=xp.sum((observed/scale)**2,axis=(0,1,2));num=xp.sum((difference/scale)**2,axis=(0,1,2))
            six_residual=xp.sqrt(xp.divide(num,den,out=xp.zeros_like(num),where=den>0))
            canonical_residual=fit['relative_residual']
            constraint_residual=None
            if constraint is not None:
                coefficients=xp.vstack([v,constant])
                amplitude=xp.max(xp.abs(coefficients),axis=0)
                coefficients=coefficients/xp.where(amplitude>0,amplitude,1.)
                numerator=xp.abs(constraint@coefficients)
                denominator=abs(constraint)@xp.abs(coefficients)
                ratios=xp.divide(numerator,denominator,out=xp.zeros_like(numerator),where=denominator>0)
                constraint_residual=xp.max(ratios,axis=0) if constraint.shape[0] else xp.zeros(coefficients.shape[1])
            if xp is not np:
                energies=xp.asnumpy(energies);six_residual=xp.asnumpy(six_residual);canonical_residual=xp.asnumpy(canonical_residual)
                if constraint_residual is not None:constraint_residual=xp.asnumpy(constraint_residual)
            if not np.all(np.isfinite(energies)) or np.any(energies < -1e-10*np.maximum(np.max(np.abs(energies),axis=0),1e-250)):
                raise ValueError('Nonfinite/negative auxiliary component energy')
            history.append((count,np.maximum(energies,0),np.asarray(canonical_residual),np.asarray(six_residual),constraint_residual))
        rows=[]
        for j,mode in enumerate(batch['modes']):
            convergence=[];previous=None;previous_components=None
            for count,energies,canonical,six,constraint_residual in history:
                total=float(energies[:,j].sum())
                convergence.append(dict(harmonics=count,canonical_relative_residual=float(canonical[j]),
                    raw_six_dof_relative_residual=float(six[j]),auxiliary_energy=total,
                    component_energies={k:float(energies[i,j]) for i,k in enumerate(COMPONENTS)},
                    relative_energy_change=None if previous is None else abs(total-previous)/max(abs(total),abs(previous),1e-250),
                    component_relative_energy_change=None if previous_components is None else {
                        k:abs(float(energies[i,j])-previous_components[i])/max(abs(float(energies[i,j])),abs(previous_components[i]),1e-250)
                        for i,k in enumerate(COMPONENTS)},
                    initial_harmonic_constraint_relative_residual=None if constraint_residual is None else float(constraint_residual[j])))
                previous=total
                previous_components=energies[:,j].tolist()
            rows.append(dict(mode=mode,eigenvalue=batch['eigenvalues'][j],source_method='MFSM_SOURCE_CPT_PREPARATION_ONLY',
                dominant_family=None,global_subtype=None,quality_state='UNAVAILABLE',scientifically_eligible=False,
                default_classifier_activation=False,unavailable_reasons=['ACTIVE_CONTACT_TANGENT_UNVERIFIED','S4R_CPT_MAPPING_UNVERIFIED',
                    'FACETED_CURVATURE_AND_MAIN_NODE_EQUILIBRIUM_UNREVIEWED','ADMISSIBLE_SOURCE_HIERARCHY_NOT_GENERATED'],
                harmonic_convergence=convergence))
        return rows


def load_probe_sample(reader,index=0):
    """Reduce selected Parquet columns only after an actual allocation failure."""
    modes=list(reader.manifest['shards'][index]['modes'])
    retries=0
    while True:
        try:
            sample=reader.read_shard(index,modes)
            return sample,dict(modes=list(modes),allocation_retries=retries)
        except MemoryError as exc:
            import traceback
            traceback.clear_frames(exc.__traceback__)
            if len(modes)==1:raise
            retries+=1
        modes=modes[:max(1,len(modes)//2)]


def run_sample_with_retry(operation,sample):
    """Keep representative probe and topology tuning inside allocation recovery."""
    from modal_batch_executor import retry_projection
    positions={mode:i for i,mode in enumerate(sample['modes'])}
    def selected(modes):
        indices=[positions[mode] for mode in modes]
        part=dict(sample,modes=list(modes),
                  eigenvalues=[sample['eigenvalues'][i] for i in indices])
        for key in ('raw','raw_grid','canonical'):
            part[key]=sample[key][...,indices]
        return operation(part)
    return retry_projection(list(sample['modes']),selected)


def benchmark_cpu_topology(operation,sample,policy,working_set_bytes):
    """Optional measured tuning must not invalidate a feasible all-core run."""
    try:
        from threadpoolctl import threadpool_limits
        controls=threadpool_limits
        controlled=True
    except ImportError:
        controls=lambda **kwargs:nullcontext()
        controlled=False
    with controls(limits=policy.cpus):
        start=time.perf_counter()
        unused,serial_retries=run_sample_with_retry(operation,sample)
        serial=time.perf_counter()-start
    info=dict(selected='SINGLE_ALL_CORE_BLAS',workers=1,blas_threads=policy.cpus,
              all_core_seconds_per_sample=serial,parallel_seconds_per_sample=None,
              allocation_fallback=False,failed_worker_counts=[],
              serial_allocation_retries=serial_retries,
              working_set_bytes=working_set_bytes,threadpool_control_available=controlled)
    if not controlled:return info
    try:
        workers=min(policy.cpus,batch_capacity(available_memory()[1],working_set_bytes))
    except MemoryError:
        workers=1
    while workers>1:
        budget=max(1,policy.cpus//workers)
        try:
            with controls(limits=budget),ThreadPoolExecutor(max_workers=workers) as pool:
                start=time.perf_counter()
                list(pool.map(lambda unused:run_sample_with_retry(operation,sample),range(workers)))
                parallel=(time.perf_counter()-start)/workers
            info['parallel_seconds_per_sample']=parallel
            if parallel<serial:
                info.update(selected=('CPU_QUEUE_ONE_BLAS_THREAD' if budget==1
                                      else 'CPU_QUEUE_MEASURED_BLAS_BUDGET'),
                            workers=workers,blas_threads=budget)
            return info
        except MemoryError as exc:
            import traceback
            traceback.clear_frames(exc.__traceback__)
            info['allocation_fallback']=True
            info['failed_worker_counts'].append(workers)
        workers=(workers+1)//2
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--portable-dir',required=True);p.add_argument('--inp',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--harmonic-counts',default='auto');args=p.parse_args()
    policy=resolve_policy(detect_resources());configure_threads(policy.cpus)
    try:
        from threadpoolctl import threadpool_limits
        controls=lambda n:threadpool_limits(limits=n)
    except ImportError:controls=lambda n:nullcontext()
    from abaqus_mfsm_input import read_input,compile_initial_constraints
    inp_hash=file_hash(args.inp);model=read_input(args.inp);start=time.perf_counter();memory=process_memory()
    reader=PortableParquetModes(args.portable_dir,model,policy);r=reader.reference
    if args.harmonic_counts=='auto':
        maximum=len(r['stations'])-2;counts=[];n=1
        while n<maximum:counts.append(n);n*=2
        counts.append(maximum)
    else:counts=[int(n) for n in args.harmonic_counts.split(',')]
    root=Path(args.output_dir)
    if root.exists():raise ValueError('Use a new output directory')
    root.mkdir(parents=True)
    material=physical_material(model);young=material['E'];nu=material['nu'];thickness=material['thickness']
    c,keys,meta=compile_initial_constraints(model);projector=ConstraintProjector(c)
    with controls(policy.cpus):kernel=PreparationKernel(r,counts,young,thickness,policy,constraints=c,raw_keys=keys)
    lookup={(name,label,dof):i*6+dof-1 for i,(name,label) in enumerate(reader.nodekeys) for dof in range(1,7)}
    permutation=np.array([lookup[k] for k in keys]);sample,probe_loading=load_probe_sample(reader)
    probe_retries={}
    def probe(xp):
        rows,retries=run_sample_with_retry(lambda part:kernel.run(part,xp),sample)
        probe_retries['cpu' if xp is np else 'gpu']=retries
        return np.array([[item for v in row['harmonic_convergence'] for item in ([v['canonical_relative_residual'],v['raw_six_dof_relative_residual'],v['initial_harmonic_constraint_relative_residual']]+[v['component_energies'][k] for k in COMPONENTS])] for row in rows])
    with controls(policy.cpus):xp,backend=select_numpy_backend(policy,probe)
    cpu_fallback_lock=threading.Lock();retry_counts={};retry_lock=threading.Lock()
    def work(index,device=None):
        from modal_batch_executor import retry_projection
        def operation(mode_ids):
            batch=reader.read_shard(index,modes=list(mode_ids))
            if device is None:rows=kernel.run(batch)
            else:
                import cupy as cp
                with cp.cuda.Device(device):
                    try:rows=kernel.run(batch,cp);cp.cuda.Stream.null.synchronize()
                    except Exception as exc:
                        if not isinstance(exc,MemoryError) and type(exc).__name__!='OutOfMemoryError':raise
                        import traceback
                        traceback.clear_frames(exc.__traceback__)
                        with kernel.lock:
                            kernel.device_cache={k:v for k,v in kernel.device_cache.items() if k[0]!=device}
                        for fitter in kernel.fits.values():
                            with fitter.lock:fitter.cache.pop(('gpu',device),None)
                        cp.get_default_memory_pool().free_all_blocks()
                        with cpu_fallback_lock:rows=kernel.run(batch)
                        for row in rows:row['device_fallback']='GPU_ALLOCATION_FAILURE_TO_CPU'
            raw=batch['raw'][permutation];projected=projector.apply(raw);difference=projected-raw
            for j,row in enumerate(rows):
                den=max(float(np.max(np.abs(raw[:,j]))),1e-250)
                row['initial_admissibility_projection_relative_correction']=float(np.max(np.abs(difference[:,j]))/den)
            return rows
        rows,retries=retry_projection(reader.manifest['shards'][index]['modes'],operation)
        with retry_lock:retry_counts[index]=retries
        return rows
    indices=list(range(len(reader.manifest['shards'])));timing={}
    if xp is np:
        estimate=sample['raw'].nbytes*7+sample['canonical'].nbytes*5
        timing=benchmark_cpu_topology(kernel.run,sample,policy,estimate)
        if timing['workers']>1:
            queue_policy=resolve_policy(policy.inventory,cpu_override=timing['workers'])
            rows=sum(list(execute_batches(indices,work,queue_policy,
                working_set_bytes=estimate,blas_threads=timing['blas_threads'])),[])
        else:
            with controls(policy.cpus):rows=sum([work(i) for i in indices],[])
    else:
        with controls(policy.cpus):result=dict(execute_device_batches(indices,work,backend['faster_devices']))
        rows=sum([result[i] for i in indices],[]);timing['selected']='MEASURED_FASTER_GPUS'
    reader.ensure_unchanged()
    if file_hash(args.inp)!=inp_hash:raise ValueError('INP changed during preparation')
    report=dict(kind='SOURCE_MECHANICAL_PREPARATION_NOT_CLASSIFICATION',source_inp_sha256=inp_hash,
        portable_manifest_sha256=reader.manifest_hash,source_odb_sha256=reader.manifest['source_odb_sha256'],
        source_odb_hash_independently_verified=False,resources=policy.provenance(),backend=backend,cpu_topology_timing=timing,probe_loading=probe_loading,probe_allocation_retries=probe_retries,allocation_retries=retry_counts,
        physical_material=dict(E=young,nu=nu,thickness=thickness),auxiliary_material=dict(E=young,nu=0.,G=young/2,thickness=thickness),
        reference=dict(nodes=len(r['xy']),strips=len(r['edges']),stations=len(r['stations']),length=r['length'],origin=r['origin']),
        harmonic_constraint_mapping=kernel.constraint_metadata,
        harmonic_counts=counts,constant_warping_extension='INDEPENDENT_V0_Y_CONSTANT_ONLY_GAMMA_XY;NOT_ORIGINAL_M_POSITIVE_SERIES',operator_build_seconds=kernel.build_seconds,elapsed_seconds=time.perf_counter()-start,
        memory_before=memory,memory_after=process_memory(),rows=rows,scientifically_eligible=False,default_classifier_activation=False,
        limitations=['Auxiliary CPT component energies are not L/D/G shares or physical S4R energies.',
            'Initial projector is a Euclidean kinematic action, not a reduced stiffness metric or active-contact validation.',
            'Facet-average drill/rotations and unweighted section-node reconstruction norm require independent mapping review.',
            'Independent constant warping extends the m>=1 source series; shear energy and rigid-null field are explicit.',
            'Published, S4R/contact and actual-family benchmarks remain unavailable; no primary method activation.'])
    with tempfile.TemporaryDirectory(prefix='preparation-stage-',dir=root) as stage:
        Path(stage,'mechanical_preparation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        os.replace(Path(stage,'mechanical_preparation.json'),root/'mechanical_preparation.json')
    print(json.dumps(dict(mode_count=len(rows),output_dir=str(root),resources=policy.provenance(),cpu_topology_timing=timing,scientifically_eligible=False)))


if __name__=='__main__':main()
