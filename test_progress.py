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


if __name__ == '__main__':
    unittest.main()
