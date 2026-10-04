import json,os,tempfile,unittest
from unittest import mock
from mfsm_audit import write_report

class AtomicReportTests(unittest.TestCase):
    def test_invalid_json_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as root:
            write_report(root,{'status':'OLD'})
            with self.assertRaises(ValueError):write_report(root,{'status':'NEW','bad':float('nan')})
            with open(os.path.join(root,'mfsm_audit.json')) as f:self.assertEqual(json.load(f),{'status':'OLD','report_artifacts':{}})
            self.assertEqual(os.listdir(root),['mfsm_audit.json'])
    def test_replace_failure_preserves_old_json_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as root:
            write_report(root,{'status':'OLD'})
            with mock.patch('mfsm_audit.os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):write_report(root,{'status':'NEW'})
            with open(os.path.join(root,'mfsm_audit.json')) as f:self.assertEqual(json.load(f)['status'],'OLD')
            self.assertEqual(os.listdir(root),['mfsm_audit.json'])
    def test_artifact_hash_check_rejects_interrupted_bundle(self):
        from mfsm_audit import load_report
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root,'data.csv'),'w') as f:f.write('original')
            with open(os.path.join(root,'mfsm_audit.json'),'w') as f:json.dump({'report_artifacts':{'data.csv':hashlib.sha256(b'original').hexdigest()}},f)
            self.assertIn('report_artifacts',load_report(root))
            with open(os.path.join(root,'data.csv'),'w') as f:f.write('interrupted replacement')
            with self.assertRaisesRegex(ValueError,'artifact'):load_report(root)
