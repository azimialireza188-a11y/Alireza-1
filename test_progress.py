import io
import unittest
from unittest import mock
import abaqus_progress as p


class ProgressTests(unittest.TestCase):
    def test_exact_counter_reports_stage_percent_rate_eta_and_remaining(self):
        clock = mock.Mock(side_effect=[0.0, 0.0, 5.0, 5.0])
        out = []
        tracker = p.ProgressTracker(
            ['build', 'solve', 'report'], [0.2, 0.6, 0.2],
            emit=out.append, clock=clock)
        tracker.start('build')
        row = tracker.update(done=5, total=10, note='meshing')
        self.assertEqual(row['stage'], 'build')
        self.assertFalse(row['estimated'])
        self.assertEqual((row['done'], row['total']), (5, 10))
        self.assertAlmostEqual(row['stage_percent'], 50.0)
        self.assertAlmostEqual(row['overall_percent'], 10.0)
        self.assertAlmostEqual(row['rate_per_second'], 1.0)
        self.assertAlmostEqual(row['eta_seconds'], 5.0)
        self.assertEqual(row['remaining_stages'], ['solve', 'report'])
        line = out[-1]
        self.assertIn('build', line)
        self.assertIn('5/10', line)
        self.assertIn('10.0%', line)
        self.assertIn('ETA', line)
        self.assertIn('remaining=solve,report', line)

    def test_estimated_stage_is_labeled_monotonic_and_never_finishes_early(self):
        now = [0.0]
        out = []
        tracker = p.ProgressTracker(['solve'], [1.0], emit=out.append, clock=lambda: now[0])
        tracker.start('solve')
        now[0] = 10.0
        a = tracker.update(estimate_fraction=.4)
        now[0] = 20.0
        b = tracker.update(estimate_fraction=.2)
        now[0] = 30.0
        c = tracker.update(estimate_fraction=1.0)
        self.assertTrue(a['estimated'])
        self.assertGreaterEqual(b['stage_percent'], a['stage_percent'])
        self.assertLess(c['stage_percent'], 100.0)
        self.assertIn('ESTIMATED', out[-1])
        now[0] = 31.0
        final = tracker.finish('solve')
        self.assertEqual(final['overall_percent'], 100.0)

    def test_summary_contains_elapsed_current_and_remaining_work(self):
        now = [0.0]
        tracker = p.ProgressTracker(['build', 'solve'], [1, 3], emit=lambda unused: None, clock=lambda: now[0])
        tracker.start('build')
        now[0] = 2.0
        tracker.finish('build')
        tracker.start('solve')
        now[0] = 5.0
        tracker.update(estimate_fraction=.25)
        s = tracker.summary()
        self.assertEqual(s['stage'], 'solve')
        self.assertEqual(s['remaining_stages'], [])
        self.assertAlmostEqual(s['overall_percent'], 43.75)
        self.assertAlmostEqual(s['elapsed_seconds'], 5.0)


    def test_summary_records_completed_stage_durations(self):
        now=[0.0]
        tracker=p.ProgressTracker(['extract','classify'],[1,1],
                                  emit=lambda unused:None,clock=lambda:now[0])
        tracker.start('extract')
        now[0]=3.5
        tracker.finish('extract')
        tracker.start('classify')
        now[0]=8.0
        tracker.finish('classify')
        summary=tracker.summary()
        self.assertEqual(summary['stage_durations_seconds'],
                         {'extract':3.5,'classify':4.5})
        self.assertAlmostEqual(summary['elapsed_seconds'],8.0)



    def test_tracker_records_stage_start_and_finish_epochs(self):
        now=[100.0]
        tracker=p.ProgressTracker(['build','solve'],[1,1],
                                  emit=lambda unused:None,clock=lambda:now[0])
        tracker.start('build')
        now[0]=112.5
        tracker.finish('build')
        self.assertEqual(tracker.stage_started_epochs['build'],100.0)
        self.assertEqual(tracker.stage_finished_epochs['build'],112.5)
        self.assertAlmostEqual(tracker.stage_durations['build'],12.5)



    def test_observer_receives_structured_monotonic_rows_without_parsing_stdout(self):
        now=[0.0]
        observed=[]
        tracker=p.ProgressTracker(
            ['extract','classify'],[1,1],emit=lambda unused:None,
            observer=observed.append,clock=lambda:now[0])
        tracker.start('extract')
        now[0]=1.0
        tracker.update(done=1,total=2,note='one of two')
        now[0]=2.0
        tracker.finish('extract')
        tracker.start('classify')
        now[0]=3.0
        tracker.update(done=1,total=4)
        overall=[row['overall_percent'] for row in observed]
        self.assertTrue(all(b>=a for a,b in zip(overall,overall[1:])))
        self.assertEqual(observed[1]['done'],1)
        self.assertEqual(observed[1]['total'],2)
        self.assertEqual(observed[1]['note'],'one of two')



if __name__ == '__main__':
    unittest.main()
