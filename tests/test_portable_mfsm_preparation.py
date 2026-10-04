import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
try:import pyarrow as pa;import pyarrow.parquet as pq
except ImportError:pa=None
from prismatic_mfsm_operators import reconstruct_canonical

@unittest.skipIf(pa is None,'Parquet dependency pyarrow unavailable')
class PortablePreparationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('portable_mfsm_preparation'),'Portable mechanical preparation missing')
        import portable_mfsm_preparation as m
        return m

    def fixture(self,root):
        z=np.linspace(0,10,7);nodekeys=[('P1',i+1) for i in range(14)]
        coords=np.array([[x,0,zz] for zz in z for x in [0,2]])
        model={'nodes':dict(zip(nodekeys,coords)),'elements':{},'input_sha256':'a'*64,'keywords':[]}
        elems={'instance':[],'label':[],'n1':[],'n2':[],'n3':[],'n4':[]}
        for i in range(6):
            labels=[2*i+1,2*i+2,2*i+4,2*i+3];model['elements']['P1',i+1]=tuple(('P1',n) for n in labels)
            elems['instance'].append('P1');elems['label'].append(i+1)
            for j,n in enumerate(labels,1):elems['n'+str(j)].append(n)
        pq.write_table(pa.table(dict(node_index=range(14),instance=['P1']*14,label=range(1,15),x=coords[:,0],y=coords[:,1],z=coords[:,2])),root/'mesh_nodes.parquet')
        pq.write_table(pa.table(dict(raw_index=range(84),node_index=np.repeat(np.arange(14),6),dof=np.tile(np.arange(1,7),14))),root/'raw_dof_map.parquet')
        pq.write_table(pa.table(elems),root/'elements.parquet')
        pq.write_table(pa.table(dict(mode=[1],eigenvalue=[-2.],description=['Mode1: EigenValue = -2'],fields=['U,UR'],source_precision=['DOUBLE_PRECISION'])),root/'modes.parquet')
        coeff=np.ones((1,2,4,1));f=reconstruct_canonical(z,coeff,[1],10.)
        data=dict(node_index=range(14))
        for j,d in enumerate(['u1','u2','u3','ur1','ur2','ur3']):
            data['m0001_'+d]=f[:,:,{0:0,1:2,2:1,5:3}[j],0].ravel() if j in (0,1,2,5) else np.zeros(14)
        table=pa.table(data).replace_schema_metadata({b'vector_coordinate_system':b'GLOBAL',b'source_odb_sha256':b'b'*64,b'modes':b'1'})
        pq.write_table(table,root/'mode_shapes_0001.parquet')
        files=sorted(p.name for p in root.glob('*.parquet'))
        manifest=dict(kind='PORTABLE_RAW_MODAL_DATA_ONLY',format='parquet',source_inp_sha256='a'*64,source_odb_sha256='b'*64,node_count=14,s4r_element_count=6,raw_dof_count=84,mode_count=1,rotations_available=True,shards=[dict(modes=[1],parquet_file='mode_shapes_0001.parquet')],modes=[dict(mode=1,eigenvalue=-2.,fields=['U','UR'])],artifacts=files,artifact_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
        (root/'modal_export.json').write_text(json.dumps(manifest));return model,manifest

    def test_reader_preserves_signed_modes_and_all_six_dofs(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model,manifest=self.fixture(p);reader=m.PortableParquetModes(p,model)
            batch=reader.read_shard(0);self.assertEqual(batch['eigenvalues'],[-2.]);self.assertEqual(batch['raw'].shape,(84,1))
            self.assertEqual(batch['canonical'].shape,(7,2,4,1));reader.ensure_unchanged()

    def test_corrupt_artifact_or_traversal_fails_before_numeric_work(self):
        m=self.module()
        for which in ['hash','path']:
            with self.subTest(which=which),tempfile.TemporaryDirectory() as d:
                p=Path(d);model,r=self.fixture(p)
                if which=='hash':(p/'modes.parquet').write_bytes(b'bad')
                else:r['artifacts'].append('../outside.parquet');(p/'modal_export.json').write_text(json.dumps(r))
                with self.assertRaises(ValueError):m.PortableParquetModes(p,model)

    def test_missing_rotation_column_and_nonfinite_data_fail_closed(self):
        m=self.module()
        for which in ['missing','nan']:
            with self.subTest(which=which),tempfile.TemporaryDirectory() as d:
                p=Path(d);model,r=self.fixture(p);t=pq.read_table(p/'mode_shapes_0001.parquet')
                if which=='missing':t=t.drop(['m0001_ur2'])
                else:
                    index=t.column_names.index('m0001_u1');t=t.set_column(index,'m0001_u1',pa.array([np.nan]*14))
                pq.write_table(t,p/'mode_shapes_0001.parquet');r['artifact_sha256']['mode_shapes_0001.parquet']=hashlib.sha256((p/'mode_shapes_0001.parquet').read_bytes()).hexdigest();(p/'modal_export.json').write_text(json.dumps(r))
                reader=m.PortableParquetModes(p,model)
                with self.assertRaises(ValueError):reader.read_shard(0)

    def test_source_changed_after_read_fails_recheck(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model,r=self.fixture(p);reader=m.PortableParquetModes(p,model)
            (p/'modal_export.json').write_text('{}')
            with self.assertRaises(ValueError):reader.ensure_unchanged()

    def test_energy_preparation_never_assigns_a_family_or_drops_rotations(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model,r=self.fixture(p);reader=m.PortableParquetModes(p,model)
            energy=m.PreparationKernel(reader.reference,[1,2],young=200.,thickness=.3)
            rows=energy.run(reader.read_shard(0))
            self.assertEqual(len(rows),1);row=rows[0]
            self.assertEqual(row['eigenvalue'],-2.);self.assertIsNone(row['dominant_family'])
            self.assertFalse(row['scientifically_eligible']);self.assertIn('S4R_CPT_MAPPING_UNVERIFIED',row['unavailable_reasons'])
            self.assertLess(row['harmonic_convergence'][0]['canonical_relative_residual'],1e-12)
            self.assertAlmostEqual(row['harmonic_convergence'][0]['auxiliary_energy'],row['harmonic_convergence'][1]['auxiliary_energy'],places=10)

    def test_material_reader_preserves_physical_E_nu_and_rejects_offset(self):
        m=self.module();self.assertTrue(hasattr(m,'physical_material'),'General physical material reader missing')
        model=dict(elastic=[dict(definition={},data=['210000., 0.28'])],sections=[dict(definition={'MATERIAL':'Steel'},data=['3., 5'])],keywords=[])
        self.assertEqual(m.physical_material(model),{'E':210000.,'nu':.28,'thickness':3.})
        model['sections'][0]['definition']['OFFSET']='0.5'
        with self.assertRaises(ValueError):m.physical_material(model)

    def test_shard_subset_reads_only_requested_modes_without_changing_metadata(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model,r=self.fixture(p);reader=m.PortableParquetModes(p,model)
            self.assertEqual(reader.read_shard(0,modes=[1])['modes'],[1])
            with self.assertRaises(ValueError):reader.read_shard(0,modes=[])

    def test_topology_worker_allocation_pressure_falls_back_to_numeric_serial(self):
        import threading
        from unittest.mock import patch
        from runtime_resources import ResourceInventory,resolve_policy
        m=self.module();policy=resolve_policy(ResourceInventory(4,4,10**9,10**9,[]))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model,r=self.fixture(p);reader=m.PortableParquetModes(p,model)
            kernel=m.PreparationKernel(reader.reference,[1,2],200.,.3)
            def run(batch):
                if threading.current_thread() is not threading.main_thread():
                    raise MemoryError('actual worker allocation pressure injected')
                return kernel.run(batch)
            with patch.object(m,'available_memory',return_value=(10**9,10**9)):
                info=m.benchmark_cpu_topology(run,reader.read_shard(0),policy,10000)
            self.assertTrue(info['allocation_fallback'])
            self.assertEqual(info['failed_worker_counts'],[4,2])
            self.assertEqual(info['workers'],1)
            self.assertEqual(info['blas_threads'],4)

    def test_probe_parquet_loading_retries_fewer_columns_without_metadata_change(self):
        from unittest.mock import patch
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model,r=self.fixture(p)
            table=pq.read_table(p/'mode_shapes_0001.parquet')
            for name in list(table.column_names):
                if name.startswith('m0001_'):
                    table=table.append_column(name.replace('m0001_','m0002_'),table.column(name))
            meta=dict(table.schema.metadata);meta[b'modes']=b'1,2'
            table=table.replace_schema_metadata(meta)
            pq.write_table(table,p/'mode_shapes_0001.parquet')
            table=pq.read_table(p/'modes.parquet');second=table.to_pydict()
            second['mode']=[2];second['eigenvalue']=[-3.]
            pq.write_table(pa.concat_tables([table,pa.table(second,schema=table.schema)]),p/'modes.parquet')
            r['mode_count']=2;r['shards'][0]['modes']=[1,2]
            r['modes'].append(dict(mode=2,eigenvalue=-3.,fields=['U','UR']))
            r['artifact_sha256']={name:m.file_hash(p/name) for name in r['artifacts']}
            (p/'modal_export.json').write_text(json.dumps(r))
            reader=m.PortableParquetModes(p,model);read=pq.read_table;seen=[]
            def allocation_fault(path,*args,**kwargs):
                if Path(path).name=='mode_shapes_0001.parquet':
                    columns=kwargs['columns'];seen.append(columns)
                    if any(c.startswith('m0002_') for c in columns):
                        raise MemoryError('full representative shard does not fit')
                return read(path,*args,**kwargs)
            with patch.object(pq,'read_table',side_effect=allocation_fault):
                sample,info=m.load_probe_sample(reader)
            self.assertEqual(sample['modes'],[1]);self.assertEqual(info['allocation_retries'],1)
            self.assertEqual(reader.manifest['shards'][0]['modes'],[1,2])
            self.assertEqual(len(seen),2)
            self.assertFalse(any(c.startswith('m0002_') for c in seen[-1]))

    def test_cli_survives_topology_worker_allocation_pressure_and_commits_report(self):
        import contextlib,io,threading
        from unittest.mock import patch
        from runtime_resources import ResourceInventory,resolve_policy
        m=self.module();policy=resolve_policy(ResourceInventory(4,4,10**9,10**9,[]))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'data';source.mkdir();model,r=self.fixture(source)
            inp=p/'input.inp';inp.write_text('source identity for mocked parsed input')
            model.update(input_sha256=m.file_hash(inp),steps=[],mpcs=[],boundary=[],
                elastic=[dict(definition={},data=['200.,.3'])],
                sections=[dict(definition={'MATERIAL':'Steel'},data=['.3,5'])])
            r['source_inp_sha256']=model['input_sha256']
            (source/'modal_export.json').write_text(json.dumps(r))
            output=p/'result';original=m.PreparationKernel.run
            def guarded(kernel,batch,xp=np):
                if threading.current_thread() is not threading.main_thread():
                    raise MemoryError('topology workers do not fit')
                return original(kernel,batch,xp)
            args=['prepare','--portable-dir',str(source),'--inp',str(inp),
                  '--output-dir',str(output),'--harmonic-counts','1,2']
            with patch('sys.argv',args),patch('abaqus_mfsm_input.read_input',return_value=model),\
                 patch.object(m,'detect_resources',return_value=policy.inventory),\
                 patch.object(m,'available_memory',return_value=(10**9,10**9)),\
                 patch.object(m.PreparationKernel,'run',guarded),contextlib.redirect_stdout(io.StringIO()):
                m.main()
            report=json.loads((output/'mechanical_preparation.json').read_text())
            self.assertEqual(len(report['rows']),1)
            self.assertFalse(report['scientifically_eligible'])
            self.assertTrue(report['cpu_topology_timing']['allocation_fallback'])
            self.assertEqual(report['cpu_topology_timing']['workers'],1)
