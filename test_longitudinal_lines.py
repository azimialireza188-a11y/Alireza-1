"""Geometry selection checks; Abaqus builds are checked separately."""
import contextlib
import io
import math
import unittest

import abaqus_complete_model_m20 as builder


class LongitudinalLinesTests(unittest.TestCase):
    def setUp(self):
        # Two quarter circles with tangent flats and intervening sharp corners.
        arc = [(math.cos(i*math.pi/24), math.sin(i*math.pi/24))
               for i in range(13)]
        self.arc1 = arc
        self.arc2 = [(x-4, y+1) for x, y in arc]
        self.points = [(1., -2.), (1., -1.)] + arc + [(-1., 1.),
            (-2., 1.), (-2., -2.), (-3., -2.), (-3., -1.),
            (-3., 0.)] + self.arc2 + [(-5., 2.), (-6., 2.)]
        self.segments = [a+b for a, b in zip(self.points, self.points[1:])]
        self.legacy = {builder.xykey(self.points[0]), builder.xykey(self.points[-1])}

    def select(self, count, keep=None):
        self.assertTrue(hasattr(builder, 'select_longitudinal_lines'),
                        'The longitudinal geometry selector is missing')
        return builder.select_longitudinal_lines(
            self.segments, self.legacy if keep is None else keep, count)

    def test_zero_preserves_the_existing_selection_exactly(self):
        self.assertEqual(self.select(0), self.legacy)

    def test_maximum_retains_every_source_boundary_and_mandatory_bolt_line(self):
        keep = self.legacy | {(1., -1.5)}
        expected = {builder.xykey(p) for p in self.points} | keep
        self.assertEqual(self.select(100, keep), expected)

    def test_two_retains_two_internal_lines_per_arc_and_no_flat_subdivision(self):
        selected = self.select(2)
        for arc in (self.arc1, self.arc2):
            interior = {builder.xykey(p) for p in arc[1:-1]}
            self.assertEqual(len(selected & interior), 2)
        self.assertNotIn((1., -1.), selected)
        self.assertNotIn((-3., -1.), selected)

    def test_increasing_count_adds_lines_without_removing_earlier_ones(self):
        previous = self.select(2)
        for count in range(3, 13):
            current = self.select(count)
            self.assertTrue(previous <= current)
            previous = current

    def test_mandatory_lines_survive_and_count_towards_arc_budget(self):
        bolt = (1., -1.5)
        anchor = builder.xykey(self.arc1[5])
        selected = self.select(2, self.legacy | {bolt, anchor})
        self.assertTrue({bolt, anchor} <= selected)
        self.assertEqual(len(selected & {builder.xykey(p) for p in self.arc1[1:-1]}), 2)

    def test_refinement_concentrates_lines_where_curvature_is_greater(self):
        headings = [0.] + [0.5*i for i in range(1, 17)] + [8.+4*i for i in range(1, 17)]
        points = [(0., 0.)]
        for heading in headings:
            x, y = points[-1]
            angle = math.radians(heading)
            points.append((x+math.cos(angle), y+math.sin(angle)))
        self.segments = [a+b for a, b in zip(points, points[1:])]
        self.legacy = {builder.xykey(points[0]), builder.xykey(points[-1])}
        selected = self.select(8)
        low = {builder.xykey(p) for p in points[2:17]}
        high = {builder.xykey(p) for p in points[17:-2]}
        self.assertGreater(len(selected & high), len(selected & low))

    def test_curve_at_either_chain_end_has_no_extra_internal_boundary(self):
        for points in (self.arc1, list(reversed(self.arc1))):
            self.segments = [a+b for a, b in zip(points, points[1:])]
            self.legacy = {builder.xykey(points[0]), builder.xykey(points[-1])}
            selected = self.select(2)
            self.assertEqual(len(selected - self.legacy), 2)

    def test_cli_defaults_and_valid_range(self):
        args = builder.parse_arguments([])
        self.assertEqual(getattr(args, 'longitudinal_lines', None), 0)
        for count in (0, 2, 10, 99, 100):
            self.assertEqual(builder.parse_arguments(
                ['--longitudinal-lines', str(count)]).longitudinal_lines, count)
        for value in ('-1', '1', '101', '2.5'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.parse_arguments(['--longitudinal-lines', value])


if __name__ == '__main__':
    unittest.main()
