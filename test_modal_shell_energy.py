import unittest
import numpy as np
import abaqus_modal_shell_energy as energy


class ShellEnergyTests(unittest.TestCase):
    def test_report_serializes_numpy_location_indices_without_partial_files(self):
        import json
        report = {'modes': [{'peak_section_point': np.int64(5), 'value': np.float64(1.25)}]}
        self.assertEqual(json.loads(energy.serialize_result(report)), {'modes': [{'peak_section_point': 5, 'value': 1.25}]})

    def data(self, membrane, curvature, thickness=3.):
        z = np.linspace(-thickness/2, thickness/2, 5)
        e = np.asarray(membrane)[None, None, :]+z[:, None, None]*np.asarray(curvature)[None, None, :]
        c = energy.plane_stress_matrix(200000., .3)
        s = e @ c.T
        return s, e

    def test_pure_membrane_and_engineering_shear(self):
        s, e = self.data([0, 0, .01], [0, 0, 0])
        r = energy.inplane_partition(s, e, 3.)
        self.assertAlmostEqual(r['membrane'][0], .5*(200000/2.6)*.01**2*3)
        self.assertAlmostEqual(r['bending'][0], 0.)

    def test_pure_bending_and_quadratic_scale(self):
        s, e = self.data([0, 0, 0], [.01, -.003, .002])
        r = energy.inplane_partition(s, e, 3.)
        expected = 3.**3/24*np.array([.01, -.003, .002]) @ energy.plane_stress_matrix(200000., .3) @ np.array([.01, -.003, .002])
        self.assertAlmostEqual(r['bending'][0], expected)
        self.assertAlmostEqual(r['membrane'][0], 0.)
        rr = energy.inplane_partition(s*-4, e*-4, 3.)
        np.testing.assert_allclose(rr['total'], r['total']*16)

    def test_mixed_membrane_bending_close(self):
        s, e = self.data([.001, .002, .003], [.004, .005, .006])
        r = energy.inplane_partition(s, e, 3.)
        np.testing.assert_allclose(r['total'], r['membrane']+r['bending'], rtol=1e-12)

    def test_negative_work_is_rejected(self):
        s, e = self.data([.01, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError): energy.inplane_partition(-s, e, 3.)

    def test_labels_not_positions_control_components(self):
        labels = ['SF6', 'SF3', 'SF1', 'SF5', 'SF2', 'SF4']
        a = np.array([[0, 3, 1, 5, 2, 4]])
        np.testing.assert_array_equal(energy.components(a, labels, ['SF1', 'SF2', 'SF3']), [[1, 2, 3]])
        with self.assertRaises(ValueError): energy.components(a, labels, ['SF7'])

    def test_weighted_concentration_and_mesh_refinement_invariance(self):
        # Half of the area has twice the density of the other half.
        a = energy.concentration(np.array([2., 2.]), np.array([4., 2.]))
        b = energy.concentration(np.ones(4), np.array([2., 2., 1., 1.]))
        self.assertAlmostEqual(a['effective_area_percent'], 90.)
        self.assertAlmostEqual(a['top_10pct_area_energy_percent'], 100*0.8/6)
        self.assertEqual(a,b)


if __name__ == '__main__': unittest.main()
