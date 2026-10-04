"""Inventory pending physical evidence. Assertions alone never activate mFSM."""
import argparse,json
from mfsm_model import ALGORITHM_VERSION,ValidationEvidence


def assess(data,model_hash):
    missing=[name for name in ValidationEvidence.REQUIRED if data.get('checks',{}).get(name) is not True]
    compatible=data.get('model_hash')==model_hash and data.get('algorithm')==ALGORITHM_VERSION
    # Automated artifact/physics reproduction is not implemented. This inventory
    # cannot promote a set of manually supplied True flags to scientific evidence.
    return dict(status='PENDING',eligible=False,model_matches=compatible,
                missing_checks=missing,reason='INDEPENDENT_MEASURED_PHYSICS_REPRODUCTION_REQUIRED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest');p.add_argument('--model-hash',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    with open(args.manifest) as f: data=json.load(f)
    result=assess(data,args.model_hash)
    with open(args.output,'w') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
