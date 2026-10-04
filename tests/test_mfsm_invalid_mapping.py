import json,os,tempfile,unittest
from types import SimpleNamespace as NS
import numpy as np
from mfsm_audit import evaluate

class InvalidMappingTests(unittest.TestCase):
    def fixture(self,root,fields,global_spaces=False,eigenvalues=None):
        node=NS(label=1,coordinates=(0.,0.,0.));instance=NS(name='P1',nodes=[node])
        odb=NS(rootAssembly=NS(instances={'P1':instance}))
        def val(data):return NS(instance=instance,nodeLabel=1,precision='SINGLE_PRECISION',data=tuple(data),localCoordSystem=None)
        frames={i+1:NS(fieldOutputs={'U':NS(values=[val(raw[:3])]),'UR':NS(values=[val(raw[3:])])}) for i,raw in enumerate(fields)}
        mapping=np.zeros((2,6));mapping[0,0]=mapping[1,5]=1.
        source=dict(reference='synthetic fixture',equations='35',metric_definition='nu0')
        meta=dict(source_odb_sha256='hash',model_signature='model',nu_class=0,contact_status='INACTIVE_VERIFIED',
            connection_status='RIGID_REDUCTION_VERIFIED',coordinate_space_review=True,constraint_mapping_review=True,
            source=source,components=['eps_x'],spaces={'G':dict(zero_components=['eps_x'],equations='fixture')})
        arrays=dict(metadata='',instances=['P1']*6,labels=[1]*6,dofs=range(1,7),coordinates=np.zeros((6,3)),
            mapping=mapping,reconstruction=mapping.T,raw_metric_diagonal=np.ones(6),K_system=np.eye(2),K_eps_x=np.zeros((2,2)),H_G=np.eye(2))
        if global_spaces:
            meta.update(global_definition_review=True,global_source=source,global_subspaces=['FLEXURAL','TORSIONAL'])
            arrays.update(G_FLEXURAL=np.eye(2)[:,0:1],G_TORSIONAL=np.eye(2)[:,1:2])
        arrays['metadata']=json.dumps(meta);path=os.path.join(root,'pack.npz');np.savez(path,**arrays)
        ev=eigenvalues or list(range(1,len(fields)+1))
        summary=dict(source_odb_sha256='hash',model_signature='model',modes=[dict(mode=i+1,eigenvalue=float(v)) for i,v in enumerate(ev)])
        resources=dict(cpus=2,gpus=0,inventory=dict(logical_cpus=2,physical_cores=2,total_memory_bytes=2**30,available_memory_bytes=2**29,devices=[]))
        return evaluate(odb,frames,summary,path,resources,os.path.join(root,'cache'))
    def test_fully_discarded_mode_is_unresolved_without_projection_crash(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.fixture(root,[[0,0,0,1,0,0],[1,0,0,0,0,0]])
            self.assertEqual(r['modes'][0]['quality_state'],'UNRESOLVED')
            self.assertEqual(r['modes'][0]['mapping_status'],'FAILED_RECONSTRUCTION')
            self.assertIsNone(r['modes'][0]['dominant_family'])
    def test_rank_collapsed_cluster_is_unresolved_without_abort(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.fixture(root,[[1,0,0,0,0,0],[1,0,0,1,0,0],[0,0,0,0,0,1]],eigenvalues=[1,1.0001,2])
            self.assertFalse(r['clusters'][0]['mapping_accepted'])
            self.assertIsNone(r['clusters'][0]['stable_family'])
    def test_repeated_cluster_clears_basis_dependent_row_subtype(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.fixture(root,[[1,0,0,0,0,0],[0,0,0,0,0,1],[1,0,0,0,0,1]],True,[1,1.0001,2])
            self.assertIsNone(r['clusters'][0]['global_subtype'])
            self.assertEqual([row['global_subtype'] for row in r['modes'][:2]],[None,None])
            self.assertEqual([row['observed_global_subtype'] for row in r['modes'][:2]],['FLEXURAL','TORSIONAL'])
