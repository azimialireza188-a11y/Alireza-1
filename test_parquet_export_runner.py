import json
import os
import tempfile
import unittest
from unittest import mock
import parquet_export_runner as r


class ParquetExportRunnerTests(unittest.TestCase):
    def test_current_pyarrow_runtime_is_preferred(self):
        with mock.patch.object(r.bundle,'runtime_probe',
                               return_value={'available':True,'pyarrow_version':'20','python':'x'}):
            found=r.find_runtime()
        self.assertTrue(found['available'])
        self.assertEqual(found['kind'],'in_process')

    def test_external_python_is_used_when_abaqus_runtime_has_no_pyarrow(self):
        completed=mock.Mock(returncode=0,stdout='21.0.0\n',stderr='')
        with mock.patch.object(r.bundle,'runtime_probe',
                               return_value={'available':False,'error':'missing'}), \
             mock.patch.object(r.shutil,'which',side_effect=lambda name: 'C:/Python/python.exe' if name=='python' else None), \
             mock.patch.object(r.subprocess,'run',return_value=completed):
            found=r.find_runtime()
        self.assertTrue(found['available'])
        self.assertEqual(found['kind'],'external')
        self.assertEqual(found['command'],['C:/Python/python.exe'])

    def test_required_mode_fails_before_expensive_run_when_no_runtime_exists(self):
        unavailable={'available':False,'status':'PARQUET_RUNTIME_UNAVAILABLE','attempts':[]}
        with mock.patch.object(r,'find_runtime',return_value=unavailable):
            with self.assertRaisesRegex(RuntimeError,'PyArrow'):
                r.prepare_runtime('required')

    def test_auto_mode_is_nonfatal_when_no_runtime_exists(self):
        unavailable={'available':False,'status':'PARQUET_RUNTIME_UNAVAILABLE','attempts':[]}
        with mock.patch.object(r,'find_runtime',return_value=unavailable):
            result=r.prepare_runtime('auto')
        self.assertFalse(result['available'])
        self.assertEqual(result['policy'],'auto')

    def test_in_process_export_calls_bundle_exporter(self):
        runtime={'available':True,'kind':'in_process','policy':'required'}
        expected={'zip_path':'x.zip','bytes':123}
        with mock.patch.object(r.bundle,'export_run',return_value=expected) as export:
            result=r.export_run('RUN','AUDIT',runtime=runtime)
        self.assertEqual(result['zip_path'],'x.zip')
        export.assert_called_once()

    def test_external_export_runs_same_repository_script_and_parses_json(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime={'available':True,'kind':'external','command':['python'],'policy':'required'}
            expected={'zip_path':'bundle.zip','bytes':456}
            completed=mock.Mock(returncode=0,stdout=json.dumps(expected),stderr='')
            with mock.patch.object(r.subprocess,'run',return_value=completed) as run:
                result=r.export_run(folder,os.path.join(folder,'audit'),runtime=runtime)
            self.assertEqual(result['bytes'],456)
            command=run.call_args.args[0]
            self.assertIn('modal_analysis_parquet.py',command[1])
            self.assertIn('--run-dir',command)


if __name__=='__main__':
    unittest.main()
