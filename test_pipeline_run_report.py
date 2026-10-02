import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pipeline_run_report as r


class PipelineRunReportTests(unittest.TestCase):
    def fake_tracker(self):
        return SimpleNamespace(
            stage_names=['BUILD','SOLVE','MODAL_AUDIT','PARQUET_EXPORT'],
            weights=[.10,.60,.25,.05],
            completed={0,1,2},
            stage='PARQUET_EXPORT',
            stage_index=3,
            stage_started=140.0,
            created_at=100.0,
            stage_fraction=.4,
            stage_durations={'BUILD':10.0,'SOLVE':25.0,'MODAL_AUDIT':5.0},
            stage_started_epochs={'BUILD':100.0,'SOLVE':110.0,'MODAL_AUDIT':135.0,
                                  'PARQUET_EXPORT':140.0},
            stage_finished_epochs={'BUILD':110.0,'SOLVE':135.0,'MODAL_AUDIT':140.0},
            summary=lambda:dict(overall_percent=97.0,elapsed_seconds=42.0,
                                stage_durations_seconds={'BUILD':10.0,'SOLVE':25.0,
                                                        'MODAL_AUDIT':5.0}))

    def test_capture_invocation_records_reproducible_normalized_command_and_context(self):
        with mock.patch.object(r,'_git_head',return_value='abc123'):
            item=r.capture_invocation(
                r'C:\\repo\\abaqus_complete_model_m20.py',
                ['--mesh-mm','5','--modal-audit'],cwd=r'D:\\run')
        self.assertIn('abaqus_complete_model_m20.py',item['normalized_command'])
        self.assertIn('--mesh-mm',item['normalized_command'])
        self.assertEqual(item['effective_args'],['--mesh-mm','5','--modal-audit'])
        self.assertEqual(item['git_commit'],'abc123')
        self.assertEqual(item['launch_cwd'],r'D:\\run')
        self.assertIn('python_version',item)

    def test_stage_rows_include_top_level_and_nested_audit_timings(self):
        audit=dict(basis_metadata=dict(progress_timing=dict(
            stage_durations_seconds={'MAP_HARMONICS':2.0,'BASIS':1.0,
                                     'CLASSIFY':1.5,'EIGENSPACE':.5})))
        rows=r.stage_rows(self.fake_tracker(),audit_summary=audit,now_epoch=142.0)
        top={x['stage']:x for x in rows if x['scope']=='PIPELINE'}
        self.assertEqual(top['BUILD']['duration_seconds'],10.0)
        self.assertEqual(top['PARQUET_EXPORT']['status'],'RUNNING')
        self.assertAlmostEqual(top['PARQUET_EXPORT']['duration_seconds'],2.0)
        nested={x['stage']:x for x in rows if x['scope']=='MODAL_AUDIT'}
        self.assertEqual(nested['CLASSIFY']['duration_seconds'],1.5)
        self.assertEqual(nested['EIGENSPACE']['status'],'COMPLETED')

    def test_write_report_creates_json_and_csv_with_stage_durations_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder,'Job.odb'),'wb') as stream:
                stream.write(b'12345')
            audit_dir=os.path.join(folder,'modal_dsm_audit')
            os.makedirs(audit_dir)
            audit=dict(basis_metadata=dict(progress_timing=dict(
                stage_durations_seconds={'MAP_HARMONICS':2.0})))
            with open(os.path.join(audit_dir,'modal_audit.json'),'w') as stream:
                json.dump(audit,stream)
            result=r.write_report(
                folder,self.fake_tracker(),
                invocation={'normalized_command':'abaqus ...','effective_args':['--x'],
                            'git_commit':'abc'},
                state={'status':'POSTPROCESSING','settings':{'resource_plan':{'cpus':24,'gpus':1}}},
                started_epoch=100.0,status='POSTPROCESSING',now_epoch=142.0)
            self.assertTrue(os.path.isfile(result['json_path']))
            self.assertTrue(os.path.isfile(result['csv_path']))
            with open(result['json_path']) as stream:
                report=json.load(stream)
            self.assertEqual(report['status'],'POSTPROCESSING')
            self.assertEqual(report['resource_plan']['cpus'],24)
            self.assertEqual(report['outputs']['Job.odb']['bytes'],5)
            self.assertTrue(any(x['stage']=='SOLVE' for x in report['stages']))
            with open(result['csv_path'],newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(any(x['stage']=='MAP_HARMONICS' for x in rows))

    def test_append_final_report_puts_normal_timing_files_into_upload_zip(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            zpath=os.path.join(folder,'bundle.zip')
            with zipfile.ZipFile(zpath,'w') as zf:
                zf.writestr('modal_summary.parquet',b'x')
            j=os.path.join(folder,'pipeline_run_report.json')
            c=os.path.join(folder,'pipeline_stage_timings.csv')
            open(j,'w').write('{}')
            open(c,'w').write('stage,duration_seconds\n')
            r.append_final_report_to_zip(zpath,j,c)
            with zipfile.ZipFile(zpath) as zf:
                names=set(zf.namelist())
            self.assertIn('pipeline_run_report.json',names)
            self.assertIn('pipeline_stage_timings.csv',names)


if __name__=='__main__':
    unittest.main()
