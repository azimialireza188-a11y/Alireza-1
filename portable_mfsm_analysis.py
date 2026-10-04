"""Stream verified portable Parquet modes through a documented mFSM operator pack.

This supplied-operator route does not generate or certify S4R/contact operators.
"""
import argparse,json
from pathlib import Path
from runtime_resources import configure_threads,detect_resources,resolve_policy


def analyze(model,reader,pack_path,cache_dir):
    import numpy as np
    from mfsm_audit import load_operator_pack
    from portable_mfsm_preparation import file_hash
    from abaqus_mfsm_input import compile_initial_constraints
    policy=reader.policy;pack_hash=file_hash(pack_path)
    with np.load(pack_path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata'].item()))
    if metadata.get('source_inp_sha256')!=model['input_sha256']:
        raise ValueError('Operator pack INP provenance differs from portable source')
    summary=dict(source_odb_sha256=reader.manifest['source_odb_sha256'],model_signature=metadata.get('model_signature'),
        modes=[dict(mode=m,eigenvalue=e) for m,e in reader.eigenvalues.items()])
    loaded=load_operator_pack(pack_path,None,summary['source_odb_sha256'],summary['model_signature'],node_coordinates=model['nodes'])
    keys=loaded[3];raw_lookup={(name,label,dof):6*i+dof-1 for i,(name,label) in enumerate(reader.nodekeys) for dof in range(1,7)}
    if set(keys)!=set(raw_lookup):raise ValueError('Operator pack requires complete portable raw map')
    permutation=np.array([raw_lookup[key] for key in keys])
    c,constraint_keys,constraint_metadata=compile_initial_constraints(model)
    constraint_permutation=np.array([raw_lookup[key] for key in constraint_keys])
    row_scale=np.asarray(abs(c).sum(axis=1)).ravel()
    shards={mode:i for i,s in enumerate(reader.manifest['shards']) for mode in s['modes']}
    def read_vectors(selected,unused):
        vectors=np.empty((len(keys),len(selected)),dtype=np.float64)
        for shard in dict.fromkeys(shards[mode] for mode in selected):
            positions=[i for i,m in enumerate(selected) if shards[m]==shard]
            part=reader.read_shard(shard,modes=[selected[i] for i in positions])['raw']
            constrained=part[constraint_permutation]
            scale=np.max(np.abs(constrained),axis=0)
            normalized=constrained/np.where(scale>0,scale,1.)
            residual=np.abs(c@normalized)/np.maximum(row_scale[:,None],1e-250)
            if np.any(residual>1e-8):raise ValueError('Modes violate initial INP constraints')
            vectors[:,positions]=part[permutation]
        return vectors
    result=evaluate_loaded(summary,loaded,read_vectors,policy.provenance(),str(cache_dir))
    reader.ensure_unchanged()
    if file_hash(pack_path)!=pack_hash:raise ValueError('Operator pack changed during analysis')
    for row in result['modes']:row['eigenvalue']=reader.eigenvalues[row['mode']]
    result.update(input_kind='PORTABLE_PARQUET_SUPPLIED_OPERATOR_PACK',source_inp_sha256=model['input_sha256'],
        source_odb_sha256=summary['source_odb_sha256'],operator_pack_sha256=pack_hash,
        portable_manifest_sha256=reader.manifest_hash,initial_constraints=constraint_metadata,
        model_signature=summary['model_signature'])
    return result


def evaluate_loaded(*args,**kwargs):
    # Public seam also permits downstream integrations to time the common engine.
    from mfsm_audit import evaluate_loaded as evaluate
    return evaluate(*args,**kwargs)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--portable-dir',required=True);p.add_argument('--inp',required=True)
    p.add_argument('--mfsm-pack',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--cache-dir');args=p.parse_args(argv)
    policy=resolve_policy(detect_resources());configure_threads(policy.cpus)
    from abaqus_mfsm_input import read_input
    from portable_mfsm_preparation import PortableParquetModes,file_hash
    from mfsm_audit import write_report
    output=Path(args.output_dir)
    if output.exists():raise ValueError('Use a new output directory')
    model=read_input(args.inp);reader=PortableParquetModes(args.portable_dir,model,policy)
    result=analyze(model,reader,args.mfsm_pack,args.cache_dir or output.parent/'.mfsm-basis-cache')
    if file_hash(args.inp)!=model['input_sha256']:raise ValueError('Source INP changed during analysis')
    output.mkdir(parents=True,exist_ok=False)
    write_report(output,result)
    print(json.dumps(dict(status=result['status'],modes=len(result['modes']),output_dir=str(output),scientifically_eligible=False)))
    return result


if __name__=='__main__':main()
