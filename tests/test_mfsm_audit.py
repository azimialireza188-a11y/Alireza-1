import unittest,tempfile,os,json
from types import SimpleNamespace
import numpy as np
from mfsm_audit import load_operator_pack,availability
import abaqus_dsm_modal_audit as audit

class AuditTests(unittest.TestCase):
    def test_no_pack_is_explicitly_unavailable(self):
        r=availability(None)
        self.assertEqual(r['status'],'UNAVAILABLE')
        self.assertFalse(r['scientifically_eligible'])
    def test_explicit_mfsm_cli_requires_pack(self):
        import contextlib,io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            audit.parse_arguments(['--run-dir','x','--classifier','mfsm'])
    def test_pack_provenance_and_rotations_required(self):
        with tempfile.TemporaryDirectory() as root:
            path=os.path.join(root,'pack.npz')
            np.savez(path,metadata=json.dumps({'source_odb_sha256':'wrong','model_signature':'model'}))
            with self.assertRaisesRegex(ValueError,'provenance'):
                load_operator_pack(path,None,'hash','model')
    def test_screening_cli_and_resource_policy(self):
        args=audit.parse_arguments(['--run-dir','x','--classifier','screening','--cpus','2','--gpus','0'])
        self.assertEqual(args.classifier,'screening')
        self.assertEqual(args.resource_policy['cpus'],2)

    def test_raw_constraints_checked_before_reduced_mapping(self):
        from mfsm_audit import evaluate
        node=SimpleNamespace(label=1,coordinates=(0.,0.,0.))
        instance=SimpleNamespace(name='P1',nodes=[node])
        odb=SimpleNamespace(rootAssembly=SimpleNamespace(instances={'P1':instance}))
        def value(data):
            return SimpleNamespace(instance=instance,nodeLabel=1,precision='SINGLE_PRECISION',data=data,localCoordSystem=None)
        frames={1:SimpleNamespace(fieldOutputs={'U':SimpleNamespace(values=[value((1.,0.,0.))]),'UR':SimpleNamespace(values=[value((0.,0.,0.))])}),
                2:SimpleNamespace(fieldOutputs={'U':SimpleNamespace(values=[value((0.,0.,0.))]),'UR':SimpleNamespace(values=[value((0.,0.,1.))])})}
        metadata=dict(source_odb_sha256='hash',model_signature='model',nu_class=0,
            contact_status='INACTIVE_VERIFIED',connection_status='RIGID_REDUCTION_VERIFIED',
            coordinate_space_review=True,constraint_mapping_review=True,
            source=dict(reference='test fixture only',equations='Eq125',metric_definition='nu0'),
            components=['eps_x','eps_y'],spaces={
                'L':dict(zero_components=['eps_x'],equations=['fixture']),
                'D':dict(zero_components=['eps_y'],equations=['fixture'])})
        mapping=np.zeros((2,6));mapping[0,0]=mapping[1,5]=1.
        resources=dict(cpus=2,gpus=0,inventory=dict(logical_cpus=2,physical_cores=2,total_memory_bytes=2**30,available_memory_bytes=2**29,devices=[]))
        summary=dict(source_odb_sha256='hash',model_signature='model',modes=[dict(mode=1,eigenvalue=1.),dict(mode=2,eigenvalue=2.)])
        with tempfile.TemporaryDirectory() as root:
            path=os.path.join(root,'pack.npz')
            np.savez(path,metadata=json.dumps(metadata),instances=['P1']*6,labels=[1]*6,dofs=range(1,7),coordinates=np.zeros((6,3)),
                mapping=mapping,K_system=np.eye(2),K_eps_x=np.diag([0.,1.]),K_eps_y=np.diag([1.,0.]),H_L=np.eye(2),H_D=np.eye(2),raw_constraints=[[0,1,0,0,0,0]])
            result=evaluate(odb,frames,summary,path,resources,os.path.join(root,'cache'))
            self.assertEqual([r['dominant_family'] for r in result['modes']],['LOCAL','DISTORTIONAL'])
            self.assertFalse(result['scientifically_eligible'])
            self.assertTrue(result['clusters'][-1]['spectral_boundary_open'])
            summary['modes'][1]['eigenvalue']=1.5
            frames[1].fieldOutputs['U'].values[0].data=(np.sqrt(.95),0.,0.)
            frames[1].fieldOutputs['UR'].values[0].data=(0.,0.,np.sqrt(.05))
            result=evaluate(odb,frames,summary,path,resources,os.path.join(root,'cache'),
                            thresholds={'dominance':.99,'max_residual':.05,'cluster_tolerance':.75})
            self.assertIsNone(result['modes'][0]['dominant_family'])
            self.assertEqual(result['thresholds']['dominance'],.99)
            self.assertEqual(len(result['clusters']),1)
            frames[1].fieldOutputs['U'].values[0].data=(1e10,0.,0.)
            frames[2].fieldOutputs['U'].values[0].data=(0.,1.,0.)
            with self.assertRaisesRegex(ValueError,'constraints'):
                evaluate(odb,frames,summary,path,resources,os.path.join(root,'cache'))

    def test_availability_does_not_require_scipy(self):
        import subprocess,sys
        script="""
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='scipy' or fullname.startswith('scipy.'):raise ImportError('SciPy blocked')
sys.meta_path.insert(0,Block())
from mfsm_audit import availability
assert availability(None)['status']=='UNAVAILABLE'
"""
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
