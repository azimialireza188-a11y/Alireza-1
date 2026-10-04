"""ODB adapter for supplied, documented mFSM system/strain operators.

No automatic S4R/contact-tangent reconstruction is claimed. Numerical results
remain separate from screening and DSM inputs pending physical validation.
"""
from contextlib import nullcontext
import csv,json,os,tempfile,time,hashlib
import numpy as np
from mfsm_model import OperatorPack,SourceDefinition,ModeBatch,matrix,check_constraints_per_mode
from runtime_resources import ResourceInventory,resolve_policy,available_memory,batch_capacity,process_memory
from modal_batch_executor import select_numpy_backend,execute_device_batches,device_batch_capacity,retry_projection


def availability(pack_path):
    return dict(status='UNAVAILABLE' if not pack_path else 'SUPPLIED_PACK_REQUIRES_VALIDATION',
        reason='NO_DOCUMENTED_SYSTEM_STRAIN_SEARCH_OPERATOR_PACK' if not pack_path else None,
        scientifically_eligible=False,default_classifier_activation=False,
        required_evidence=['S4R/reference mapping','BEAM_MPC/contact base-state operators',
                           'source modal hierarchy','published and actual-model benchmarks'])


def load_operator_pack(path,odb,odb_hash,signature):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata'].item()))
        if meta.get('source_odb_sha256')!=odb_hash or meta.get('model_signature')!=signature or not signature:
            raise ValueError('mFSM operator provenance differs from ODB/model')
        if meta.get('nu_class')!=0 or meta.get('contact_status') not in ('INACTIVE_VERIFIED','TANGENT_VERIFIED'):
            raise ValueError('Auxiliary nu=0 and verified contact state/operators required')
        if meta.get('connection_status') not in ('ELASTIC_VERIFIED','RIGID_REDUCTION_VERIFIED'):
            raise ValueError('Verified physical connection mapping required')
        if not meta.get('coordinate_space_review') or not meta.get('constraint_mapping_review'):
            raise ValueError('Documented coordinate/constraint reviews required')
        keys=[(str(n),int(label),int(dof)) for n,label,dof in zip(data['instances'],data['labels'],data['dofs'])]
        if not keys or len(set(keys))!=len(keys) or len({len(data[k]) for k in ('instances','labels','dofs')})!=1:
            raise ValueError('Invalid/duplicate raw U/UR DOF map')
        node_dofs={}
        for name,label,dof in keys: node_dofs.setdefault((name,label),set()).add(dof)
        if any(dofs!=set(range(1,7)) for dofs in node_dofs.values()):
            raise ValueError('Full U+UR data required for every mapped node')
        coords=matrix(data['coordinates'],len(keys))
        if coords.shape[1]!=3: raise ValueError('Mapped coordinates must be 3D')
        lookup={name:{n.label:n.coordinates for n in odb.rootAssembly.instances[name].nodes} for name in {k[0] for k in keys}}
        actual=np.array([lookup[name][label] for name,label,unused in keys])
        if not np.allclose(coords,actual,rtol=0,atol=max(1e-5,1e-6*np.max(np.abs(actual)))):
            raise ValueError('Operator coordinates differ from ODB')
        operators=OperatorPack(data['K_system'],{k:data['K_'+k] for k in meta['components']},
            SourceDefinition(**meta['source']),dict(meta))
        mapping=matrix(data['mapping'],len(operators.system))
        if mapping.shape[1]!=len(keys): raise ValueError('Reduced/raw DOF map mismatch')
        spaces={};hierarchy=None
        construction=meta.get('basis_construction','SUPPLIED_SEARCH_SPACES')
        if construction=='KHEZRI_2019_HIERARCHY':
            hierarchy=dict(meta['hierarchy'])
            equilibrium=hierarchy.get('equilibrium_construction','SUPPLIED_Z_TE')
            if equilibrium=='APPENDIX_A1_ROWS':
                from source_mfsm_hierarchy import transverse_equilibrium_rows
                if 'Z_te' in data: raise ValueError('Ambiguous supplied/derived transverse equilibrium')
                canonical=data['K_kappa_x_canonical'];q=matrix(data['canonical_reduction'],len(canonical))
                if q.shape[1]!=len(operators.system): raise ValueError('Canonical reduction DOF mismatch')
                expected=q.T@canonical@q;actual=operators.components['kappa_x']
                if not np.allclose(expected,actual,rtol=1e-10,atol=max(np.max(np.abs(actual)),1e-250)*1e-12):
                    raise ValueError('Canonical/reduced kappa_x mismatch')
                hierarchy['transverse_equilibrium']=transverse_equilibrium_rows(canonical,data['unprescribed_rows'],q)
                hierarchy['coordinate_metric']=q.T@q
            elif equilibrium=='SUPPLIED_Z_TE':
                if 'Z_te' not in data: raise ValueError('Source hierarchy requires transverse equilibrium Z_te')
                hierarchy['transverse_equilibrium']=data['Z_te'].copy()
            else: raise ValueError('Unsupported equilibrium construction')
            if 'K_gamma_open' in data: hierarchy['gamma_open']=data['K_gamma_open'].copy()
            if 'warping_projection' in data: hierarchy['warping_projection']=data['warping_projection'].copy()
            if 'S_open' in data: hierarchy['open_shear_basis']=data['S_open'].copy()
            if 'coordinate_metric' in data:
                if equilibrium=='APPENDIX_A1_ROWS' and not np.allclose(
                    hierarchy['coordinate_metric'],data['coordinate_metric'],rtol=1e-10,atol=1e-12):
                    raise ValueError('Canonical Euclidean metric mismatch')
                hierarchy['coordinate_metric']=data['coordinate_metric'].copy()
            if meta.get('spaces') or meta.get('global_subspaces'):
                raise ValueError('Source hierarchy cannot be combined with supplied family/subtype bases')
        elif construction=='SUPPLIED_SEARCH_SPACES':
            for name,spec in meta['spaces'].items():
                spaces[name]=dict(spec,search_basis=data['H_'+name])
                if 'C_'+name in data: spaces[name]['constraints']=data['C_'+name]
        else: raise ValueError('Unsupported basis construction')
        constraints=matrix(data['raw_constraints']) if 'raw_constraints' in data else None
        if constraints is not None and constraints.shape[1]!=len(keys): raise ValueError('Raw constraint DOF mismatch')
        inverse=data['reconstruction'].copy() if 'reconstruction' in data else None
        diagonal=data['raw_metric_diagonal'].copy() if 'raw_metric_diagonal' in data else None
        global_spaces={}
        if meta.get('global_definition_review'):
            SourceDefinition(**meta['global_source'])
            global_spaces={name:data['G_'+name].copy() for name in meta.get('global_subspaces',[])}
            if any(name not in ('FLEXURAL','TORSIONAL') for name in global_spaces):
                raise ValueError('Unsupported global subtype space')
        adapter=dict(reconstruction=inverse,raw_metric_diagonal=diagonal,coordinates=coords.copy(),metadata=meta,global_spaces=global_spaces,hierarchy=hierarchy)
        if (inverse is None)!=(diagonal is None): raise ValueError('Reconstruction and dimensional metric must be supplied together')
        return operators,spaces,mapping,keys,constraints,adapter


def evaluate(odb,frames,summary,pack_path,resource_metadata,cache_dir,thresholds=None):
    from mfsm_reference_basis import cached_mfsm_basis
    from mechanical_modal_classifier import EnergyProjector,classify_modes,classify_cluster
    thresholds=dict(dict(dominance=.9,max_residual=.05,cluster_tolerance=.001),**(thresholds or {}))
    memory_before=process_memory()
    start=time.perf_counter()
    phase_seconds={}
    operators,spaces,mapping,keys,constraints,adapter=load_operator_pack(
        pack_path,odb,summary['source_odb_sha256'],summary['model_signature'])
    from modal_reconstruction import ReconstructionCheck
    from assembly_projector import assembly_diagnostics
    reconstruction=None
    if adapter['reconstruction'] is not None:
        reconstruction=ReconstructionCheck(mapping,adapter['reconstruction'],adapter['raw_metric_diagonal'],
            adapter['metadata'].get('mapping_tolerance',1e-8))
    node_keys=list(dict.fromkeys((name,label) for name,label,unused in keys))
    node_index={key:i for i,key in enumerate(node_keys)}
    raw_index={key:i for i,key in enumerate(keys)}
    translation_indices=np.array([[raw_index[(name,label,dof)] for dof in (1,2,3)] for name,label in node_keys])
    xyz=adapter['coordinates'][translation_indices[:,0]]
    pieces=[key[0] for key in node_keys]
    seam_pairs=[];seam_axes=[]
    for seam in adapter['metadata'].get('diagnostic_seams',[]):
        seam_pairs.append(tuple(node_index[(str(k[0]),int(k[1]))] for k in (seam['a'],seam['b'])))
        seam_axes.append(seam['axes'])
    mode_diagnostics={}
    phase_seconds['operator_load_and_adapter_setup']=time.perf_counter()-start
    policy=resolve_policy(ResourceInventory(**resource_metadata['inventory']),
                          resource_metadata['cpus'],resource_metadata['gpus'])
    try:
        from threadpoolctl import threadpool_limits
        context=threadpool_limits(limits=policy.cpus)
    except ImportError:
        context=nullcontext()
    from abaqus_dsm_modal_audit import read_mapped_mode,close_clusters
    from abaqus_modal_harmonics import map_modes
    with context:
        basis_start=time.perf_counter()
        subtype_basis=None
        if adapter['hierarchy'] is not None:
            from source_mfsm_hierarchy import cached_source_hierarchy
            basis,subtype_basis,cache=cached_source_hierarchy(operators,adapter['hierarchy'],cache_dir)
        else:
            basis,cache=cached_mfsm_basis(operators,spaces,cache_dir,policy)
        projector=EnergyProjector(basis)
        global_projector=EnergyProjector(subtype_basis) if subtype_basis is not None and any(
            space.shape[1] for space in subtype_basis.spaces.values()) else None
        if adapter['global_spaces']:
            from mfsm_model import BasisPack,content_hash
            global_projector=EnergyProjector(BasisPack(basis.metric,adapter['global_spaces'],
                dict(mechanical_definition_review=True,source=adapter['metadata']['global_source']),
                content_hash(adapter['metadata']['global_source'],list(adapter['global_spaces'].values()))))
        phase_seconds['basis_and_projector']=time.perf_counter()-basis_start
        phase_seconds['extraction_mapping_diagnostics']=0.
        # Disk-backed retained vectors avoid an all-modes RAM allocation.
        ids=[r['mode'] for r in summary['modes']];n=len(operators.system)
        bytes_per_mode=8*(len(keys)+n*(6+2*len(basis.spaces)))
        capacity=min(len(ids),batch_capacity(available_memory()[1],bytes_per_mode*max(1,policy.gpus)))
        results=[];gpu_info=None;backend=np;retries=0;device_telemetry=[]
        import threading
        telemetry_lock=threading.Lock();cpu_fallback_lock=threading.Lock()
        device_bytes_per_mode=8*((len(projector.slices)+6)*n+projector.b.shape[1]+4+2*len(projector.slices))
        persistent_bytes=sum(a.nbytes for a in (projector.lt,projector.b,projector.inverse))
        with tempfile.TemporaryDirectory(prefix='mfsm-modes-') as directory:
            retained=np.memmap(os.path.join(directory,'vectors.bin'),dtype='float64',mode='w+',shape=(n,len(ids)))
            def batches():
                offset=0
                while offset<len(ids):
                    extraction_start=time.perf_counter()
                    live=min(capacity,len(ids)-offset,batch_capacity(available_memory()[1],bytes_per_mode*max(1,policy.gpus)))
                    while True:
                        try:
                            vectors=np.empty((len(keys),live),dtype=np.float64);break
                        except MemoryError:
                            if live==1:raise
                            live=max(1,live//2)
                    selected=ids[offset:offset+live]
                    for j,mode in enumerate(selected):vectors[:,j]=read_mapped_mode(frames[mode],keys)
                    raw=ModeBatch(vectors,tuple(selected))
                    if constraints is not None:
                        check_constraints_per_mode(raw.vectors,constraints)
                    mapped=map_modes(raw,mapping)
                    check=reconstruction.evaluate(raw.vectors,mapped.vectors) if reconstruction else None
                    for j,mode in enumerate(selected):
                        mode_diagnostics[mode]=dict(
                            mapping_status=('VERIFIED_RECONSTRUCTION' if check['accepted'][j] else 'FAILED_RECONSTRUCTION') if check else 'UNVERIFIED_RECONSTRUCTION',
                            mapping_relative_error=check['relative_errors'][j] if check else None,
                            assembly_diagnostics=assembly_diagnostics(xyz,raw.vectors[translation_indices,j],pieces,seam_pairs,
                                seam_axes=seam_axes if seam_pairs else None))
                    retained[:,offset:offset+len(selected)]=mapped.vectors
                    phase_seconds['extraction_mapping_diagnostics']+=time.perf_counter()-extraction_start
                    offset+=len(selected)
                    yield mapped
            # ODB reads stay on the parent thread; a device owns its projection queue.
            iterator=batches();first=next(iterator)
            # Time the actual first batch where capacity permits, including transfers.
            probe_indices=[j for j,mode in enumerate(first.ids) if mode_diagnostics[mode]['mapping_status']=='VERIFIED_RECONSTRUCTION']
            probe_count=len(probe_indices)
            probe_capacity_reason=None
            if policy.gpus:
                try:
                    import cupy as cp
                    for device in range(min(policy.gpus,cp.cuda.runtime.getDeviceCount())):
                        with cp.cuda.Device(device):
                            free,total=cp.cuda.runtime.memGetInfo()
                            probe_count=min(probe_count,device_batch_capacity(free,device_bytes_per_mode,persistent_bytes))
                except (ImportError,OSError,RuntimeError,MemoryError) as exc:
                    probe_capacity_reason=str(exc)
            probe=first.vectors[:,probe_indices[:probe_count]]
            backend_start=time.perf_counter()
            if probe.shape[1]:
                backend,gpu_info=select_numpy_backend(policy,lambda xp:np.concatenate(
                    list(projector.project(probe,xp=xp)['shares_percent'].values())))
            else:
                backend=np;gpu_info=dict(backend='numpy_cpu',device=None,reason='NO_ACCEPTED_MODES_IN_INITIAL_BATCH',gpu_timings_seconds=[])
            phase_seconds['backend_probe']=time.perf_counter()-backend_start
            gpu_info['probe_modes']=probe.shape[1]
            gpu_info['probe_capacity_reason']=probe_capacity_reason
            def project_valid_batch(mapped,device=None):
                nonlocal retries
                if backend is np:
                    def operation(indices):return classify_modes(ModeBatch(mapped.vectors[:,indices],tuple(mapped.ids[i] for i in indices)),basis,
                        thresholds=thresholds,policy=policy,projector=projector,global_projector=global_projector)
                    rows,count=retry_projection(np.arange(len(mapped.ids)),operation)
                    with telemetry_lock:retries+=count
                    return rows
                with backend.cuda.Device(device):
                    pool=backend.get_default_memory_pool()
                    free,total=backend.cuda.runtime.memGetInfo()
                    available=free+max(0,pool.total_bytes()-pool.used_bytes())
                    missing_persistent=0 if device in projector.device_arrays else persistent_bytes
                    try:size=min(len(mapped.ids),device_batch_capacity(available,device_bytes_per_mode,missing_persistent))
                    except MemoryError:size=1
                    def operation(indices):return classify_modes(ModeBatch(mapped.vectors[:,indices],tuple(mapped.ids[i] for i in indices)),basis,
                        thresholds=thresholds,policy=policy,projector=projector,backend=backend,global_projector=global_projector)
                    def fallback(indices):
                        with cpu_fallback_lock:
                            return classify_modes(ModeBatch(mapped.vectors[:,indices],tuple(mapped.ids[i] for i in indices)),basis,
                                thresholds=thresholds,projector=projector,global_projector=global_projector)
                    rows=[];count=0
                    for offset in range(0,len(mapped.ids),size):
                        part,attempts=retry_projection(np.arange(offset,min(offset+size,len(mapped.ids))),operation,
                            fallback=fallback,release=pool.free_all_blocks)
                        rows.extend(part);count+=attempts
                    free_after,unused=backend.cuda.runtime.memGetInfo()
                    with telemetry_lock:
                        retries+=count
                        if count:gpu_info['fallback_reason']='ACTUAL_ALLOCATION_RETRY_OR_CPU_FALLBACK'
                        device_telemetry.append(dict(device=device,batch_modes=len(mapped.ids),device_chunk_size=size,
                            free_bytes_before=int(free),free_bytes_after=int(free_after),total_bytes=int(total),
                            pool_used_bytes=int(pool.used_bytes()),pool_total_bytes=int(pool.total_bytes()),allocation_retries=count))
                    return rows
            def project_batch(mapped,device=None):
                accepted=[j for j,mode in enumerate(mapped.ids) if mode_diagnostics[mode]['mapping_status']=='VERIFIED_RECONSTRUCTION']
                by_mode={}
                if accepted:
                    valid=mapped if len(accepted)==len(mapped.ids) else ModeBatch(mapped.vectors[:,accepted],tuple(mapped.ids[j] for j in accepted))
                    by_mode={row['mode']:row for row in project_valid_batch(valid,device)}
                return [by_mode.get(mode) or dict(mode=mode,dominant_family=None,quality_state='UNRESOLVED',
                    global_subtype=None,shares_percent={},cross_percent={},relative_residual=None,closure_error=None,
                    condition=projector.condition,flags=[],scientifically_eligible=False,
                    source_method=basis.metadata['source_method'],metric_definition='AUXILIARY_SYSTEM_ENERGY_NU_ZERO') for mode in mapped.ids]
            import itertools
            pipeline_start=time.perf_counter()
            queue=itertools.chain([first],iterator)
            if backend is np:
                for mapped in queue: results.extend(project_batch(mapped))
            else:
                ordered=dict(execute_device_batches(queue,project_batch,gpu_info['faster_devices']))
                for index in sorted(ordered): results.extend(ordered[index])
            phase_seconds['streamed_projection_pipeline_wall']=time.perf_counter()-pipeline_start
            cluster_start=time.perf_counter()
            for row in results:
                row.update(mode_diagnostics[row['mode']])
                if row['mapping_status']!='VERIFIED_RECONSTRUCTION':
                    row['dominant_family']=None;row['quality_state']='UNRESOLVED';row['global_subtype']=None
                    row['flags'].append(row['mapping_status'])
            grouped=close_clusters(summary['modes'],thresholds.get('cluster_tolerance',.001));index={mode:j for j,mode in enumerate(ids)}
            clusters=[]
            for members in grouped:
                modes=tuple(r['mode'] for r in members)
                accepted=all(mode_diagnostics[i]['mapping_status']=='VERIFIED_RECONSTRUCTION' for i in modes)
                if accepted:
                    try:
                        cluster=classify_cluster(ModeBatch(np.asarray(retained[:,[index[i] for i in modes]]),modes),basis,thresholds=thresholds,projector=projector,global_projector=global_projector)
                    except ValueError as exc:
                        if str(exc)!='Rank-deficient observed eigenspace':raise
                        cluster=dict(modes=list(modes),stable_family=None,global_subtype=None,scientifically_eligible=False,
                            quality_state='UNRESOLVED',reason='RANK_DEFICIENT_OBSERVED_EIGENSPACE')
                else:
                    cluster=dict(modes=list(modes),stable_family=None,global_subtype=None,scientifically_eligible=False,
                        quality_state='UNRESOLVED',reason='FAILED_OR_UNVERIFIED_RECONSTRUCTION')
                cluster['spectral_boundary_open']=ids[-1] in modes
                cluster['mapping_accepted']=accepted
                if cluster['spectral_boundary_open'] or not accepted:
                    cluster['stable_family']=None;cluster['global_subtype']=None
                if len(modes)>1 and cluster.get('global_subtype') is None:
                    for row in results:
                        if row['mode'] in modes:
                            row['observed_global_subtype']=row['global_subtype']
                            row['global_subtype']=None
                            if row['observed_global_subtype']:row['flags'].append('REPEATED_EIGENSPACE_SUBTYPE_INDETERMINATE')
                clusters.append(cluster)
            retained.flush();del retained
            phase_seconds['cluster_and_final_qc']=time.perf_counter()-cluster_start
    return dict(status='NUMERICAL_DECOMPOSITION_AVAILABLE_PHYSICAL_VALIDATION_PENDING',
        scientifically_eligible=False,default_classifier_activation=False,
        source_method=basis.metadata['source_method'],metric_definition='AUXILIARY_SYSTEM_ENERGY_NU_ZERO',
        basis_definition_id=basis.definition_id,basis_metadata=basis.metadata,
        phase_seconds=phase_seconds,modes=results,clusters=clusters,resources=policy.provenance(),backend=gpu_info,
        memory=dict(before=memory_before,after=process_memory(),devices=device_telemetry,
            device_estimated_bytes_per_mode=device_bytes_per_mode,device_persistent_bytes=persistent_bytes),thresholds=thresholds,allocation_retries=retries,cache=cache,batch_size=capacity,elapsed_seconds=time.perf_counter()-start,
        limitations=['Supplied operators/search spaces require independent physical validation.',
                    'No automatic S4R/contact-tangent reconstruction or DSM acceptance.',
                    'Original screening/DSM reports remain separate; no silent primary-method replacement.'])


def _render_report(directory,result):
    os.makedirs(directory,exist_ok=True)
    with open(os.path.join(directory,'mfsm_audit.json'),'w',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    if not result.get('modes'): return
    names=sorted(set().union(*(r['shares_percent'] for r in result['modes'])))
    with open(os.path.join(directory,'mfsm_percentages.csv'),'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['mode','dominant_family','quality_state','global_subtype']+[k+'_energy_percent' for k in names]+['relative_residual','scientifically_eligible'])
        writer.writeheader()
        for row in result['modes']:
            flat={k:row[k] for k in ('mode','dominant_family','quality_state','global_subtype','relative_residual','scientifically_eligible')}
            flat.update({k+'_energy_percent':row['shares_percent'].get(k,0.) for k in names});writer.writerow(flat)
    if any('assembly_diagnostics' in r for r in result['modes']):
        with open(os.path.join(directory,'mfsm_mapping_assembly.csv'),'w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=['mode','mapping_status','mapping_relative_error','relative_piece_norm_percent','seam_vectors','seam_components','diagnostic_metric'])
            writer.writeheader()
            for row in result['modes']:
                diag=row.get('assembly_diagnostics',{})
                writer.writerow(dict(mode=row['mode'],mapping_status=row.get('mapping_status'),
                    mapping_relative_error=row.get('mapping_relative_error'),relative_piece_norm_percent=diag.get('relative_piece_norm_percent'),
                    seam_vectors=json.dumps(diag.get('seam_vectors')),seam_components=json.dumps(diag.get('seam_components')),
                    diagnostic_metric=diag.get('metric')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5))
    for name in names:
        ax.plot([r['mode'] for r in result['modes']],[r['shares_percent'].get(name,0.) for r in result['modes']],label=name)
    ax.set(xlabel='Abaqus mode',ylabel='Percent of auxiliary system energy',
           title='mFSM supplied-operator decomposition | physical validation pending')
    if names:ax.legend()
    fig.tight_layout();fig.savefig(os.path.join(directory,'mfsm_percentages.png'),dpi=160);plt.close(fig)


def write_report(directory,result):
    """Stage all artifacts; replace each atomically, JSON commit record last."""
    os.makedirs(directory,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mfsm-report-',dir=directory) as staging:
        _render_report(staging,result)
        artifacts={}
        for name in sorted(os.listdir(staging)):
            if name=='mfsm_audit.json':continue
            with open(os.path.join(staging,name),'rb') as f:artifacts[name]=hashlib.sha256(f.read()).hexdigest()
        published=dict(result,report_artifacts=artifacts)
        with open(os.path.join(staging,'mfsm_audit.json'),'w',encoding='utf-8') as f:
            json.dump(published,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        for name in artifacts:os.replace(os.path.join(staging,name),os.path.join(directory,name))
        os.replace(os.path.join(staging,'mfsm_audit.json'),os.path.join(directory,'mfsm_audit.json'))


def load_report(directory):
    """Reject a mixed/interrupted artifact generation using the JSON commit."""
    with open(os.path.join(directory,'mfsm_audit.json'),encoding='utf-8') as f:result=json.load(f)
    for name,expected in result.get('report_artifacts',{}).items():
        if os.path.basename(name)!=name:raise ValueError('Invalid report artifact path')
        try:
            with open(os.path.join(directory,name),'rb') as f:actual=hashlib.sha256(f.read()).hexdigest()
        except OSError as exc:raise ValueError('Missing report artifact: '+name) from exc
        if actual!=expected:raise ValueError('Report artifact checksum mismatch: '+name)
    return result
