import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_physical_walls import wavy_wall


class ArchiveTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('abaqus_reclassify_archive'),
                             'Archive-only reclassification command is missing')
        import abaqus_reclassify_archive
        return abaqus_reclassify_archive

    def test_streaming_archive_preserves_mode_station_node_component_order(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'shapes.npz'
            xy = np.array([[0., 0.], [1., 0.], [2., 0.]])
            values = np.arange(54., dtype=float).reshape(3, 2, 3, 3)
            np.savez_compressed(path, metadata=json.dumps(dict(names=['P'], transverse=[0, 1],
                source_odb_sha256='a', model_signature='b')),
                modes=[2, 4, 9], eigenvalues=[3., 5., 8.], z=[0., 10.], p0_xy=xy,
                p0_u=values, p0_reference_xy=xy)
            with m.ShapeArchive(path) as archive:
                for i in range(3):
                    np.testing.assert_array_equal(archive.read_next(), values[i, :, :, :2])
                with self.assertRaises(EOFError):
                    archive.read_next()

    def test_sparse_application_matches_dense_projector(self):
        m = self.module(); xy = wavy_wall(21)
        kwargs = dict(xy=xy, edges=[(i, i+1) for i in range(20)], pieces=['P']*21,
            weights=np.linspace(.5, 2., 21),
            physical_segments={'P': np.column_stack((xy[:-1], xy[1:])).tolist()})
        sparse = m.SparseProjector(**kwargs)
        import abaqus_modal_report as report
        dense = report.SectionProjector(**kwargs)
        v = np.random.RandomState(25).normal(size=126)
        for a,b in zip(sparse.audit_components(v), dense.audit_components(v)):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)

    def test_archive_provenance_mismatch_is_rejected(self):
        m = self.module()
        with self.assertRaisesRegex(ValueError, 'provenance'):
            m.validate_provenance(dict(source_odb_sha256='wrong',model_signature='b'),
                dict(source_odb_sha256='a',model_signature='b'))

    def test_selected_plot_accepts_json_geometry(self):
        m = self.module()
        xy = wavy_wall(21)
        geometry = dict(xy=xy.tolist(), edges=[(i, i+1) for i in range(20)], pieces=['P']*21)
        u = np.column_stack((np.zeros(21), np.sin(np.linspace(0., np.pi, 21))))
        selected = [dict(mode=2, z=5., U=u, L=u, D=u*0., old=[1.,99.,0.], new=[100.,0.,0.])]
        with tempfile.TemporaryDirectory() as tmp:
            m.selected_plot(Path(tmp), selected, geometry)
            self.assertGreater((Path(tmp)/'selected_mode_components.png').stat().st_size, 1000)

    def test_reclassification_recomputes_cross_term_instead_of_copying_old_zero(self):
        m = self.module(); xy = wavy_wall(21)
        fit = m.SparseProjector(xy, [(i, i+1) for i in range(20)], ['P']*21, np.ones(21),
            physical_segments={'P': np.column_stack((xy[:-1], xy[1:])).tolist()})
        vector = np.random.RandomState(9).normal(size=42)
        old = dict(mode=1, flags=[], displacement_cross_percent=0., condition=99.)
        settings = dict(dominance=.9, max_sensitivity_pp=5., max_assembly_percent=25., max_other_percent=25.)
        row = m.classified_row(old, vector, [fit]*3, settings)
        parts = fit.audit_components(vector)
        cross = sum(2.*np.dot(a,b) for i,a in enumerate(parts) for b in parts[i+1:])
        self.assertAlmostEqual(row['displacement_cross_percent'], 100.*cross/np.dot(vector,vector), places=10)
        self.assertNotAlmostEqual(row['displacement_cross_percent'], 0., places=3)
        self.assertIsNone(row['condition'])


if __name__ == '__main__':
    unittest.main()
