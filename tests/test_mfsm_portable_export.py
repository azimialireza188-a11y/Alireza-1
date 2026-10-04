import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from tests.test_mfsm_inp_constraints import FIXTURE


class PortableExportTests(unittest.TestCase):
    def setup_odb(self):
        from abaqus_mfsm_input import read_input_text
        model=read_input_text(FIXTURE);instances={}
        for name in model['instances']:
            instances[name]=NS(name=name,nodes=[NS(label=n,coordinates=xyz) for (inst,n),xyz in model['nodes'].items() if inst==name],
                               elements=[NS(label=n,connectivity=[key[1] for key in conn],type='S4R') for (inst,n),conn in model['elements'].items() if inst==name])
        frames=[]
        for mode in range(1,4):
            fields={}
            for field in ('U','UR'):
                fields[field]=NS(values=[NS(instance=instances[name],nodeLabel=label,
                    data=(float(mode),0.,0.),precision='SINGLE_PRECISION',localCoordSystem=None) for name,label in model['nodes']])
            frames.append(NS(mode=mode,description='Mode %d: EigenValue = %s'%(mode,-2.5 if mode==1 else mode),fieldOutputs=fields))
        return model,NS(rootAssembly=NS(instances=instances),steps={'Buckle':NS(frames=frames)})

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('abaqus_mfsm_export'), 'Portable ODB exporter missing')
        import abaqus_mfsm_export
        return abaqus_mfsm_export

    def test_exports_full_uur_shards_without_odb_runtime_on_reader(self):
        m=self.module();model,odb=self.setup_odb()
        with tempfile.TemporaryDirectory() as root:
            output=Path(root,'export')
            r=m.export_modal_data(odb,model,output,'source-hash',modes_per_shard=2)
            self.assertEqual(r['mode_count'],3);self.assertEqual(len(r['shards']),2)
            self.assertEqual(r['modes'][0]['eigenvalue'],-2.5)
            self.assertFalse(r['scientifically_eligible'])
            self.assertEqual(r['source_odb_sha256'],'source-hash')
            with np.load(output/r['shards'][0],allow_pickle=False) as data:
                self.assertEqual(data['vectors'].shape,(48,2))
                np.testing.assert_equal(data['vectors'][0],[1.,2.])
                np.testing.assert_equal(data['vectors'][3],[1.,2.])
            self.assertEqual(json.loads((output/'modal_export.json').read_text())['mode_count'],3)
            with self.assertRaises(ValueError):m.export_modal_data(odb,model,output,'source-hash')

    def test_missing_rotations_aborts_without_inventing_data(self):
        m=self.module();model,odb=self.setup_odb();del odb.steps['Buckle'].frames[1].fieldOutputs['UR']
        with tempfile.TemporaryDirectory() as root:
            output=Path(root,'export')
            with self.assertRaisesRegex(ValueError,'UR'):m.export_modal_data(odb,model,output,'hash')
            self.assertFalse((output/'modal_export.json').exists())

    def test_wrong_topology_or_coordinates_rejected(self):
        m=self.module()
        for kind in ('coordinates','connectivity'):
            model,odb=self.setup_odb()
            inst=odb.rootAssembly.instances['P2']
            if kind=='coordinates':inst.nodes[0].coordinates=np.array([100.,100.,100.])
            else:inst.elements[0].connectivity=[1,2,4,3]
            with tempfile.TemporaryDirectory() as root,self.assertRaisesRegex(ValueError,'ODB/INP'):
                m.export_modal_data(odb,model,Path(root,'export'),'hash')

    def test_duplicate_mode_ids_rejected(self):
        m=self.module();model,odb=self.setup_odb();odb.steps['Buckle'].frames[1].mode=1
        with tempfile.TemporaryDirectory() as root,self.assertRaisesRegex(ValueError,'mode'):
            m.export_modal_data(odb,model,Path(root,'export'),'hash')

    def test_each_odb_field_value_sequence_is_retrieved_once(self):
        m=self.module();model,odb=self.setup_odb();watched=[]
        class Field:
            def __init__(self,values):self.data=values;self.reads=0
            @property
            def values(self):self.reads+=1;return self.data
        for frame in odb.steps['Buckle'].frames:
            for name,field in list(frame.fieldOutputs.items()):
                tracker=Field(field.values);watched.append(tracker);frame.fieldOutputs[name]=tracker
        with tempfile.TemporaryDirectory() as root:
            m.export_modal_data(odb,model,Path(root,'export'),'hash')
        self.assertEqual([field.reads for field in watched],[1]*6)

    def test_cli_rejects_locked_or_changing_source_before_commit(self):
        from unittest.mock import patch
        import sys
        m=self.module()
        for kind in ('lock','odb_change','inp_change'):
            model,odb=self.setup_odb();odb.close=lambda:None
            with tempfile.TemporaryDirectory() as root:
                inp=Path(root,'source.inp');inp.write_text(FIXTURE)
                source=Path(root,'source.odb');source.write_bytes(b'initial source generation')
                output=Path(root,'export')
                if kind=='lock':source.with_suffix('.lck').write_text('active writer')
                else:
                    target=source if kind=='odb_change' else inp
                    values=odb.steps['Buckle'].frames[0].fieldOutputs['U'].values
                    class MutatingField:
                        @property
                        def values(self):
                            target.write_bytes(b'different source generation');return values
                    odb.steps['Buckle'].frames[0].fieldOutputs['U']=MutatingField()
                argv=['export','--odb',str(source),'--inp',str(inp),'--output-dir',str(output)]
                with self.subTest(kind=kind):
                    with patch.object(sys,'argv',argv),patch.dict(sys.modules,{'odbAccess':NS(openOdb=lambda **unused:odb)}):
                        with self.assertRaisesRegex(ValueError,'lock|changed'):m.main()
                    self.assertFalse((output/'modal_export.json').exists())
