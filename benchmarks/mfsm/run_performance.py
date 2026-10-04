"""Measured numerical-kernel benchmark; synthetic data, NOT physical validation."""
import argparse,json,time
import numpy as np
from mfsm_model import BasisPack,ModeBatch
from mechanical_modal_classifier import EnergyProjector,classify_modes
from runtime_resources import detect_resources,resolve_policy,configure_threads,process_memory
from modal_batch_executor import select_numpy_backend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dofs',type=int,default=300);p.add_argument('--modes',type=int,default=100)
    p.add_argument('--output',required=True);args=p.parse_args()
    if args.dofs<3 or args.modes<1:p.error('dofs >=3 and modes >=1 required')
    policy=resolve_policy(detect_resources());configure_threads(policy.cpus)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=policy.cpus):
        identity=np.eye(args.dofs);indices=np.array_split(np.arange(args.dofs),3)
        basis=BasisPack(identity,{k:identity[:,v] for k,v in zip(('L','D','G'),indices)}, {},'SYNTHETIC_ONLY')
        batch=ModeBatch(np.random.default_rng(17).normal(size=(args.dofs,args.modes)),tuple(range(args.modes)))
        start=time.perf_counter();projector=EnergyProjector(basis);cold=time.perf_counter()-start
        backend,info=select_numpy_backend(policy,lambda xp:np.concatenate(list(projector.project(batch.vectors,xp)['shares_percent'].values())))
        context=backend.cuda.Device(info['device']) if backend is not np else threadpool_limits(limits=policy.cpus)
        with context:
            start=time.perf_counter();rows=classify_modes(batch,basis,projector=projector,backend=backend);warm=time.perf_counter()-start
    result=dict(kind='SYNTHETIC_NUMERICAL_KERNEL_ONLY',scientifically_eligible=False,
                memory=process_memory(),resources=policy.provenance(),backend=info,dofs=args.dofs,modes=args.modes,
                basis_seconds=cold,projection_seconds=warm,max_closure_error=max(r['closure_error'] for r in rows))
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
