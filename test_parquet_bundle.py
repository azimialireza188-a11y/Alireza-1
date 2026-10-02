import json
import os
import tempfile
import unittest
import zipfile
import numpy as np
import pyarrow.parquet as pq
import modal_analysis_parquet as p


class ParquetBundleTests(unittest.TestCase):
    def reference(self):
        return dict(
            length_mm=100.0,
            material=dict(thickness_mm=2.0),
            nodes=[
                dict(id=1,piece='C1',x=0.0,y=0.0),
                dict(id=2,piece='C1',x=10.0,y=0.0),
                dict(id=3,piece='C2',x=0.0,y=10.0),
            ],
            pieces=[
                dict(name='C1',original_name='P1',node_ids=[1,2]),
                dict(name='C2',original_name='P2',node_ids=[3]),
            ],
            definition_hash='refhash')

    def mapped_mode(self):
        z=np.array([0.,25.,50.,75.,100.])
        U=np.zeros((5,3,3),float)
        UR=np.zeros_like(U)
        for i,zz in enumerate(z):
            U[i,:,0]=np.sin(np.pi*zz/100.)*np.array([1.,2.,.5])
            U[i,:,2]=np.cos(np.pi*zz/100.)*np.array([.2,.4,.1])
            UR[i,:,1]=np.cos(np.pi*zz/100.)*np.array([.01,.02,.005])
        return dict(mode=1,eigenvalue=123.4,z=z,U=U,UR=UR)

    def harmonic(self):
        mapped=self.mapped_mode()
        import abaqus_modal_harmonics as h
        return h.decompose_mode(mapped['z'],mapped['U'],mapped['UR'],100.,3)

    def summary(self):
        return dict(
            metadata=dict(processed_modes=1,available_modes=1,odb='demo.odb'),
            basis_metadata=dict(reference_hash='refhash',retained_harmonics=[1],
                                basis_hashes={'1':'basis1'},
                                resource_plan={'cpus':24,'gpus':1}),
            modes=[dict(
                mode=1,eigenvalue=123.4,stress_MPa=123.4,dominant_m=1,
                half_wavelength_mm=100.,L_energy_percent=92.,
                D_energy_percent=5.,G_energy_percent=2.,O_energy_percent=1.,
                L_vector_percent=90.,D_vector_percent=6.,G_vector_percent=3.,
                O_vector_percent=1.,assembly_percent=4.,
                seam_normal_opening_index=.2,seam_transverse_slip_index=.1,
                seam_longitudinal_slip_index=.05,final_family='LOCAL',
                quality_state='OK',flags=['METRIC_SENSITIVE'],cluster_id=1,
                harmonic_residual=.01,mechanical_residual_percent=1.0,
                metric_sensitivity_pp=2.0,geometric_screening_family='D')],
            eigenspace_clusters=[dict(
                cluster_id=1,modes=[1],eigenvalue_range=[123.4,123.4],
                family='LOCAL',flags=[],bounds=dict(
                    min_percent=[91.,4.,1.],max_percent=[93.,6.,3.]))])

    def test_compact_tables_capture_summary_harmonics_and_peak_section(self):
        tables=p.analysis_tables(
            self.summary(),self.reference(),[self.mapped_mode()],[self.harmonic()],
            harmonic_section_min_share=.001)
        self.assertTrue({'modal_summary','harmonic_summary','peak_sections',
                         'harmonic_sections','clusters','provenance'}.issubset(tables))
        self.assertEqual(tables['modal_summary'].num_rows,1)
        self.assertEqual(tables['peak_sections'].num_rows,3)
        self.assertGreaterEqual(tables['harmonic_summary'].num_rows,3)
        self.assertEqual(tables['modal_summary'].column('final_family')[0].as_py(),'LOCAL')
        peak=tables['peak_sections'].to_pydict()
        self.assertEqual(set(peak['piece']),{'C1','C2'})
        self.assertIn('U1_normalized',peak)
        hs=tables['harmonic_sections'].to_pydict()
        self.assertTrue(all(x>=.001 for x in hs['harmonic_share']))

    def test_bundle_writes_zstd_parquet_manifest_and_single_upload_zip(self):
        tables=p.analysis_tables(
            self.summary(),self.reference(),[self.mapped_mode()],[self.harmonic()],
            harmonic_section_min_share=.001)
        with tempfile.TemporaryDirectory() as folder:
            result=p.write_bundle(tables,folder,bundle_name='demo_analysis_bundle')
            self.assertTrue(os.path.isfile(result['zip_path']))
            self.assertTrue(os.path.isfile(result['manifest_path']))
            with open(result['manifest_path']) as stream:
                manifest=json.load(stream)
            self.assertEqual(manifest['format'],'parquet_analysis_bundle_v1')
            self.assertIn('modal_summary.parquet',manifest['files'])
            table=pq.read_table(os.path.join(result['directory'],'modal_summary.parquet'))
            self.assertEqual(table.num_rows,1)
            with zipfile.ZipFile(result['zip_path']) as zf:
                names=set(zf.namelist())
            self.assertIn('modal_summary.parquet',names)
            self.assertIn('manifest.json',names)

    def test_parquet_runtime_probe_is_explicit(self):
        probe=p.runtime_probe()
        self.assertTrue(probe['available'])
        self.assertIn('pyarrow_version',probe)


if __name__=='__main__':
    unittest.main()
