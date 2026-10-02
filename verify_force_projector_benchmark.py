"""Compare Python force decomposition against native CUFSM benchmark MAT.

Generate MAT using benchmark_force_split_cufsm.m. This is an algorithm test,
not physical validation of any Abaqus model or its DSM family minima.
"""
import argparse
import hashlib
import json
import numpy as np
from scipy.io import loadmat
from abaqus_modal_validation import ForceProjector


def verify(mat_path, output_path):
    data = loadmat(mat_path)
    result = ForceProjector(data['K'], data['J'], data['equilibrium']).project(data['U'])
    report = {'benchmark_schema_version': 1,
              'algorithm_under_test': 'abaqus_modal_validation.ForceProjector',
              'native_reference_kind': 'CUFSM fcFSM MAT export',
              f+'_relative_vector_error': float(np.linalg.norm(result['components'][f]-data['P'+f])/
              np.linalg.norm(data['P'+f])) for f in ('L', 'D', 'G')}
    report['energy_closure_relative'] = result['cross_relative']
    report['maximum_native_reconstruction_error'] = float(np.linalg.norm(data['PL']+data['PD']+data['PG']-data['U'])/
                                                          np.linalg.norm(data['U']))
    report['source'] = str(data['source'].ravel()[0])
    with open(report['source'], 'rb') as stream:
        report['native_source_sha256'] = hashlib.sha256(stream.read()).hexdigest()
    report['scope'] = 'Native CUFSM single-channel algorithm benchmark; NOT four-piece Abaqus physical validation'
    report['external_runtime_required'] = True
    report['synthetic_substitute_allowed'] = False
    report['passed'] = bool(max(report[f+'_relative_vector_error'] for f in ('L', 'D', 'G')) < 1e-7
                            and report['energy_closure_relative'] < 1e-10)
    with open(output_path, 'w') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    if not report['passed']: raise RuntimeError('Force projector benchmark failed; see '+output_path)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.mat, args.output)
