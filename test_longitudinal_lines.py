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

    def select(self, count, keep=None, spacing=0.0):
        self.assertTrue(hasattr(builder, 'select_longitudinal_lines'),
                        'The longitudinal geometry selector is missing')
        return builder.select_longitudinal_lines(
            self.segments, self.legacy if keep is None else keep, count, spacing)

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

    def test_optional_lines_respect_minimum_section_path_spacing(self):
        radius = 20.0
        points = [(radius*math.cos(i*math.pi/40.), radius*math.sin(i*math.pi/40.))
                  for i in range(21)]
        self.segments = [a+b for a, b in zip(points, points[1:])]
        self.legacy = {builder.xykey(points[0]), builder.xykey(points[-1])}
        selected = self.select(20, spacing=5.0)
        ds = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:])]
        arc = [0.0]
        for value in ds:
            arc.append(arc[-1]+value)
        ids = [i for i, point in enumerate(points) if builder.xykey(point) in selected]
        self.assertLess(len(ids)-2, 20)  # spacing saturates before the requested budget
        self.assertGreater(len(ids), 2)
        for a, b in zip(ids, ids[1:]):
            self.assertGreaterEqual(arc[b]-arc[a]+1e-9, 5.0)

    def test_mandatory_close_lines_are_preserved_but_do_not_allow_close_optional_lines(self):
        radius = 20.0
        points = [(radius*math.cos(i*math.pi/40.), radius*math.sin(i*math.pi/40.))
                  for i in range(21)]
        self.segments = [a+b for a, b in zip(points, points[1:])]
        mandatory = {builder.xykey(points[0]), builder.xykey(points[-1]), builder.xykey(points[2])}
        self.legacy = mandatory
        selected = self.select(20, spacing=5.0)
        self.assertTrue(mandatory <= selected)
        ds = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:])]
        arc = [0.0]
        for value in ds:
            arc.append(arc[-1]+value)
        selected_ids = [i for i, point in enumerate(points) if builder.xykey(point) in selected]
        mandatory_ids = {0, 2, len(points)-1}
        for i in selected_ids:
            if i in mandatory_ids:
                continue
            self.assertTrue(all(abs(arc[i]-arc[j])+1e-9 >= 5.0 for j in selected_ids if j != i))

    def test_spacing_zero_is_exact_backward_compatibility(self):
        for count in (2, 5, 12):
            self.assertEqual(self.select(count), self.select(count, spacing=0.0))

    def test_100_keeps_all_source_lines_even_when_spacing_is_requested(self):
        expected = {builder.xykey(p) for p in self.points} | self.legacy
        self.assertEqual(self.select(100, spacing=5.0), expected)

    def test_curve_at_either_chain_end_has_no_extra_internal_boundary(self):
        for points in (self.arc1, list(reversed(self.arc1))):
            self.segments = [a+b for a, b in zip(points, points[1:])]
            self.legacy = {builder.xykey(points[0]), builder.xykey(points[-1])}
            selected = self.select(2)
            self.assertEqual(len(selected - self.legacy), 2)

    def test_cli_defaults_and_valid_range(self):
        args = builder.parse_arguments([])
        self.assertEqual(getattr(args, 'longitudinal_lines', None), 0)
        self.assertEqual(getattr(args, 'longitudinal_line_min_spacing_mm', None), 0.0)
        for count in (0, 2, 10, 99, 100):
            self.assertEqual(builder.parse_arguments(
                ['--longitudinal-lines', str(count)]).longitudinal_lines, count)
        self.assertEqual(builder.parse_arguments(
            ['--longitudinal-line-min-spacing-mm', '5']).longitudinal_line_min_spacing_mm, 5.0)
        for value in ('-1', '1', '101', '2.5'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.parse_arguments(['--longitudinal-lines', value])
        for value in ('-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.parse_arguments(['--longitudinal-line-min-spacing-mm', value])


if __name__ == '__main__':
    unittest.main()
