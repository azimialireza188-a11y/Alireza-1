"""ODB adapter for supplied, documented mFSM system/strain operators.

No automatic S4R/contact-tangent reconstruction is claimed. Numerical results
remain separate from screening and DSM inputs pending physical validation.
"""
from contextlib import nullcontext
import csv,json,os,tempfile,time
import numpy as np
from mfsm_model import OperatorPack,SourceDefinition,ModeBatch,matrix,check_constraints_per_mode
from runtime_resources import ResourceInventory,resolve_policy,available_memory,batch_capacity
from modal_batch_executor import select_numpy_backend,execute_device_batches


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
        spaces={}
        for name,spec in meta['spaces'].items():
            spaces[name]=dict(spec,search_basis=data['H_'+name])
            if 'C_'+name in data: spaces[name]['constraints']=data['C_'+name]
        constraints=matrix(data['raw_constraints']) if 'raw_constraints' in data else None
        if constraints is not None and constraints.shape[1]!=len(keys): raise ValueError('Raw constraint DOF mismatch')
        return operators,spaces,mapping,keys,constraints


def evaluate(odb,frames,summary,pack_path,resource_metadata,cache_dir,thresholds=None):
    from mfsm_reference_basis import cached_mfsm_basis
    from mechanical_modal_classifier import EnergyProjector,classify_modes,classify_cluster
    thresholds=dict(dict(dominance=.9,max_residual=.05,cluster_tolerance=.001),**(thresholds or {}))
    start=time.perf_counter()
    operators,spaces,mapping,keys,constraints=load_operator_pack(
        pack_path,odb,summary['source_odb_sha256'],summary['model_signature'])
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
        basis,cache=cached_mfsm_basis(operators,spaces,cache_dir,policy)
        projector=EnergyProjector(basis)
        # Disk-backed retained vectors avoid an all-modes RAM allocation.
        ids=[r['mode'] for r in summary['modes']];n=len(operators.system)
        bytes_per_mode=8*(len(keys)+n*(6+2*len(basis.spaces)))
        capacity=min(len(ids),batch_capacity(available_memory()[1],bytes_per_mode*max(1,policy.gpus)))
        results=[];gpu_info=None;backend=np;retries=0
        with tempfile.TemporaryDirectory(prefix='mfsm-modes-') as directory:
            retained=np.memmap(os.path.join(directory,'vectors.bin'),dtype='float64',mode='w+',shape=(n,len(ids)))
            def batches():
                for offset in range(0,len(ids),capacity):
                    selected=ids[offset:offset+capacity]
                    raw=ModeBatch(np.column_stack([read_mapped_mode(frames[i],keys) for i in selected]),tuple(selected))
                    if constraints is not None:
                        check_constraints_per_mode(raw.vectors,constraints)
                    mapped=map_modes(raw,mapping)
                    retained[:,offset:offset+len(selected)]=mapped.vectors
                    yield mapped
            # ODB reads stay on the parent thread; a device owns its projection queue.
            iterator=batches();first=next(iterator)
            probe=first.vectors[:,:min(8,len(first.ids))]
            backend,gpu_info=select_numpy_backend(policy,lambda xp:np.concatenate(
                list(projector.project(probe,xp=xp)['shares_percent'].values())))
            def project_batch(mapped,device=None):
                nonlocal retries
                try:
                    device_context=backend.cuda.Device(device) if backend is not np else nullcontext()
                    with device_context:
                        return classify_modes(mapped,basis,thresholds=thresholds,policy=policy,projector=projector,backend=backend)
                except Exception as exc:
                    if not isinstance(exc,MemoryError) and type(exc).__name__!='OutOfMemoryError': raise
                    retries+=1
                    gpu_info['fallback_reason']='ACTUAL_ALLOCATION_FAILURE: '+str(exc)
                    rows=[]
                    for j,mode in enumerate(mapped.ids):
                        rows.extend(classify_modes(ModeBatch(mapped.vectors[:,j:j+1],(mode,)),basis,thresholds=thresholds,projector=projector))
                    return rows
            import itertools
            queue=itertools.chain([first],iterator)
            if backend is np:
                for mapped in queue: results.extend(project_batch(mapped))
            else:
                ordered=dict(execute_device_batches(queue,project_batch,gpu_info['faster_devices']))
                for index in sorted(ordered): results.extend(ordered[index])
            grouped=close_clusters(summary['modes'],thresholds.get('cluster_tolerance',.001));index={mode:j for j,mode in enumerate(ids)}
            clusters=[]
            for members in grouped:
                modes=tuple(r['mode'] for r in members)
                cluster=classify_cluster(ModeBatch(np.asarray(retained[:,[index[i] for i in modes]]),modes),basis,thresholds=thresholds,projector=projector)
                cluster['spectral_boundary_open']=ids[-1] in modes
                if cluster['spectral_boundary_open']: cluster['stable_family']=None
                clusters.append(cluster)
            retained.flush();del retained
    return dict(status='NUMERICAL_DECOMPOSITION_AVAILABLE_PHYSICAL_VALIDATION_PENDING',
        scientifically_eligible=False,default_classifier_activation=False,
        source_method=basis.metadata['source_method'],metric_definition='AUXILIARY_SYSTEM_ENERGY_NU_ZERO',
        basis_definition_id=basis.definition_id,basis_metadata=basis.metadata,
        modes=results,clusters=clusters,resources=policy.provenance(),backend=gpu_info,
        thresholds=thresholds,allocation_retries=retries,cache=cache,batch_size=capacity,elapsed_seconds=time.perf_counter()-start,
        limitations=['Supplied operators/search spaces require independent physical validation.',
                    'No automatic S4R/contact-tangent reconstruction or DSM acceptance.',
                    'Original screening/DSM reports remain separate; no silent primary-method replacement.'])


def write_report(directory,result):
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5))
    for name in names:
        ax.plot([r['mode'] for r in result['modes']],[r['shares_percent'].get(name,0.) for r in result['modes']],label=name)
    ax.set(xlabel='Abaqus mode',ylabel='Percent of auxiliary system energy',
           title='mFSM supplied-operator decomposition | physical validation pending')
    ax.legend();fig.tight_layout();fig.savefig(os.path.join(directory,'mfsm_percentages.png'),dpi=160);plt.close(fig)
