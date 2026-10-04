import json,tempfile,unittest
from pathlib import Path
import numpy as np
from tests import test_portable_mfsm_preparation as fixtures
pa=fixtures.pa

@unittest.skipIf(pa is None,'Parquet dependency unavailable')
class PortableAnalysisTests(unittest.TestCase):
    def fixture(self,root):
        from portable_mfsm_preparation import PortableParquetModes
        model,manifest=fixtures.PortablePreparationTests().fixture(root)
        model.update(steps=[],mpcs=[],boundary=[],sets={})
        reader=PortableParquetModes(root,model)
        raw=reader.read_shard(0)['raw'];mapping=raw.T/(raw.T@raw)[0,0]
        keys=[(name,label,dof) for name,label in reader.nodekeys for dof in range(1,7)]
        keys=keys[::-1];mapping=mapping[:,::-1]
        metadata=dict(source_inp_sha256=model['input_sha256'],source_odb_sha256=manifest['source_odb_sha256'],
            model_signature='synthetic-fixture',nu_class=0,contact_status='INACTIVE_VERIFIED',connection_status='RIGID_REDUCTION_VERIFIED',
            coordinate_space_review=True,constraint_mapping_review=True,
            source=dict(reference='synthetic fixture only',equations='fixture',metric_definition='nu0'),
            components=['eps_x','eps_y'],spaces={'L':dict(zero_components=['eps_x'],equations=['fixture'])})
        arrays=dict(metadata=json.dumps(metadata),instances=[k[0] for k in keys],labels=[k[1] for k in keys],dofs=[k[2] for k in keys],
            coordinates=[model['nodes'][k[:2]] for k in keys],mapping=mapping,reconstruction=raw[::-1],raw_metric_diagonal=np.ones(len(keys)),
            K_system=np.eye(1),K_eps_x=np.zeros((1,1)),K_eps_y=np.eye(1),H_L=np.eye(1))
        path=root/'pack.npz';np.savez(path,**arrays)
        return model,reader,path,arrays

    def test_actual_parquet_to_classifier_and_atomic_report(self):
        from portable_mfsm_analysis import analyze
        from mfsm_audit import write_report,load_report
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,reader,path,arrays=self.fixture(root)
            result=analyze(model,reader,path,root/'cache')
            self.assertEqual(result['modes'][0]['dominant_family'],'LOCAL')
            self.assertEqual(result['modes'][0]['mapping_status'],'VERIFIED_RECONSTRUCTION')
            self.assertEqual(result['modes'][0]['eigenvalue'],-2.)
            self.assertTrue(result['clusters'][0]['spectral_boundary_open'])
            self.assertFalse(result['scientifically_eligible'])
            self.assertEqual(result['resources']['memory_reserve_bytes'],0)
            write_report(root/'report',result);self.assertEqual(load_report(root/'report')['source_inp_sha256'],model['input_sha256'])

    def test_pack_input_provenance_is_mandatory(self):
        from portable_mfsm_analysis import analyze
        for value in [None,'c'*64]:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as d:
                root=Path(d);model,reader,path,arrays=self.fixture(root)
                meta=json.loads(arrays['metadata']);meta['source_inp_sha256']=value;arrays['metadata']=json.dumps(meta);np.savez(path,**arrays)
                with self.assertRaisesRegex(ValueError,'INP provenance'):analyze(model,reader,path,root/'cache')

    def test_missing_raw_node_is_rejected(self):
        from portable_mfsm_analysis import analyze
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,reader,path,arrays=self.fixture(root)
            for name in ('instances','labels','dofs','coordinates','raw_metric_diagonal'):arrays[name]=arrays[name][6:]
            arrays['mapping']=arrays['mapping'][:,6:];arrays['reconstruction']=arrays['reconstruction'][6:];np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,'complete portable raw map'):analyze(model,reader,path,root/'cache')

    def test_initial_input_constraints_cannot_be_omitted_by_pack(self):
        from portable_mfsm_analysis import analyze
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,reader,path,arrays=self.fixture(root)
            model['boundary']=[dict(step=None,attrs={},flags=[],rows=[['P1.3','1','1','0']])]
            with self.assertRaisesRegex(ValueError,'initial INP constraints'):analyze(model,reader,path,root/'cache')

    def test_changed_pack_before_publication_is_rejected(self):
        from portable_mfsm_analysis import analyze
        from unittest.mock import patch
        import portable_mfsm_analysis as module
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,reader,path,arrays=self.fixture(root)
            original=module.evaluate_loaded
            def mutate(*args,**kwargs):
                result=original(*args,**kwargs);path.write_bytes(b'changed');return result
            with patch.object(module,'evaluate_loaded',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'Operator pack changed'):analyze(model,reader,path,root/'cache')

    def test_reader_allocation_failure_retries_smaller_mode_batch(self):
        from portable_mfsm_analysis import analyze
        from portable_mfsm_preparation import PortableParquetModes,file_hash
        import pyarrow.parquet as pq
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,reader,path,arrays=self.fixture(root)
            table=pq.read_table(root/'mode_shapes_0001.parquet')
            for name in list(table.column_names):
                if name.startswith('m0001_'):table=table.append_column(name.replace('0001','0002'),table[name])
            meta=dict(table.schema.metadata);meta[b'modes']=b'1,2';table=table.replace_schema_metadata(meta)
            pq.write_table(table,root/'mode_shapes_0001.parquet')
            manifest=reader.manifest;manifest['mode_count']=2;manifest['shards'][0]['modes']=[1,2]
            manifest['modes'].append(dict(mode=2,eigenvalue=-1.,fields=['U','UR']))
            pq.write_table(pa.table(dict(mode=[1,2],eigenvalue=[-2.,-1.])),root/'modes.parquet')
            manifest['artifact_sha256']={name:file_hash(root/name) for name in manifest['artifacts']}
            (root/'modal_export.json').write_text(json.dumps(manifest));reader=PortableParquetModes(root,model)
            original=reader.read_shard;attempts=[]
            def limited(index,modes=None):
                attempts.append(list(modes))
                if len(modes)>1:raise MemoryError('injected allocation failure')
                return original(index,modes)
            with patch.object(reader,'read_shard',side_effect=limited):result=analyze(model,reader,path,root/'cache')
            self.assertEqual([r['mode'] for r in result['modes']],[1,2]);self.assertEqual(attempts,[[1,2],[1],[2]]);self.assertEqual(result['allocation_retries'],1)

    def test_cli_runs_without_abaqus_and_refuses_existing_output(self):
        from portable_mfsm_analysis import main
        from portable_mfsm_preparation import file_hash
        from mfsm_audit import load_report
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,reader,path,arrays=self.fixture(root)
            lines=['*PART, NAME=P','*NODE']
            lines += [','.join(map(str,[label,*xyz])) for (name,label),xyz in model['nodes'].items()]
            lines += ['*ELEMENT, TYPE=S4R']
            lines += [','.join(map(str,[label,*[key[1] for key in nodes]])) for (name,label),nodes in model['elements'].items()]
            lines += ['*END PART','*ASSEMBLY, NAME=A','*INSTANCE, NAME=P1, PART=P','*END INSTANCE','*END ASSEMBLY']
            inp=root/'model.inp';inp.write_text('\n'.join(lines)+'\n');digest=file_hash(inp)
            manifest=reader.manifest;manifest['source_inp_sha256']=digest;(root/'modal_export.json').write_text(json.dumps(manifest))
            meta=json.loads(arrays['metadata']);meta['source_inp_sha256']=digest;arrays['metadata']=json.dumps(meta);np.savez(path,**arrays)
            args=['--portable-dir',str(root),'--inp',str(inp),'--mfsm-pack',str(path),'--output-dir',str(root/'report')]
            main(args);self.assertEqual(load_report(root/'report')['modes'][0]['dominant_family'],'LOCAL')
            with self.assertRaisesRegex(ValueError,'new output'):main(args)
