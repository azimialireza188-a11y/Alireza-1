import json
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
import numpy as np
import abaqus_modal_archive as a


def field(values):
    return NS(values=values)


def value(instance, label, data, local=None):
    return NS(instance=NS(name=instance), nodeLabel=label, data=tuple(data),
              precision='SINGLE_PRECISION', localCoordSystem=local)


class ModalArchiveTests(unittest.TestCase):
    def fake_odb(self, include_ur=True, local=False):
        nodes = [NS(label=1, coordinates=(0., 0., 0.)),
                 NS(label=2, coordinates=(1., 0., 0.))]
        inst = NS(name='P1', nodes=nodes)
        frames = [NS(description='Initial', frameValue=0., fieldOutputs={})]
        outputs = {'U': field([
            value('P1', 1, (1., 2., 3.), np.eye(3) if local else None),
            value('P1', 2, (4., 5., 6.))])}
        if include_ur:
            outputs['UR'] = field([
                value('P1', 1, (.1, .2, .3)),
                value('P1', 2, (.4, .5, .6))])
        frames.append(NS(description='Mode 1: Eigen Value = 10.0', frameValue=10., fieldOutputs=outputs))
        return NS(rootAssembly=NS(instances={'P1': inst}),
                  steps={'Buckle': NS(frames=frames)})

    def test_round_trip_preserves_u_ur_node_order_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'modes.npz')
            meta = dict(step='Buckle', source_odb_sha256='abc', model_signature='xyz',
                        resource_plan={'cpus': 24})
            summary = a.extract_modal_archive(self.fake_odb(), meta, path)
            self.assertEqual(summary['modes'], 1)
            with a.open_modal_archive(path) as archive:
                self.assertEqual(archive.metadata['source_odb_sha256'], 'abc')
                self.assertEqual(archive.node_keys, [('P1', 1), ('P1', 2)])
                row = archive.read_mode(0)
                self.assertEqual((row['mode'], row['eigenvalue']), (1, 10.0))
                np.testing.assert_allclose(row['U'], [[1,2,3],[4,5,6]])
                np.testing.assert_allclose(row['UR'], [[.1,.2,.3],[.4,.5,.6]])

    def test_u_only_odb_is_explicitly_unavailable_for_mechanical_classification(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(a.MechanicalClassificationUnavailable,
                                        'MISSING_UR'):
                a.extract_modal_archive(self.fake_odb(include_ur=False), {'step':'Buckle'},
                                        os.path.join(folder, 'modes.npz'))

    def test_local_coordinate_nodal_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'global'):
                a.extract_modal_archive(self.fake_odb(local=True), {'step':'Buckle'},
                                        os.path.join(folder, 'modes.npz'))

    def test_builder_source_requests_u_and_ur_but_not_shell_fields(self):
        import abaqus_complete_model_m20 as builder
        with open(builder.__file__) as stream:
            source = stream.read()
        self.assertIn("variables=('U', 'UR')", source)
        self.assertNotIn("variables=('S', 'E', 'SF', 'SE')", source)


if __name__ == '__main__':
    unittest.main()
