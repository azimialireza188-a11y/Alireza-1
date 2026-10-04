import unittest
import numpy as np
from assembly_projector import assembly_diagnostics

class AssemblyTests(unittest.TestCase):
    def test_common_rigid_motion_no_false_slip(self):
        xyz=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]])
        u=np.cross(np.tile([0,0,1.],(6,1)),xyz)+[1,2,3]
        result=assembly_diagnostics(xyz,u,[1,1,1,2,2,2],[(1,3)])
        self.assertLess(result['relative_piece_norm'],1e-12)
        self.assertLess(result['seam_relative_norm'],1e-12)
    def test_relative_piece_motion_is_diagnostic(self):
        xyz=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]])
        u=np.zeros((6,3));u[3:,0]=1.
        result=assembly_diagnostics(xyz,u,[1,1,1,2,2,2],[(1,3)])
        self.assertGreater(result['relative_piece_norm'],0.)
        self.assertNotIn('family',result)
    def test_oriented_seam_opening_and_slip(self):
        xyz=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]])
        u=np.zeros((6,3));u[3:,0]=1.
        r=assembly_diagnostics(xyz,u,[1,1,1,2,2,2],[(1,3)],seam_axes=[np.eye(3)])
        self.assertEqual(len(r['seam_components']),1)
        self.assertAlmostEqual(r['seam_components'][0]['opening'],r['seam_vectors'][0][0])
        self.assertEqual(r['seam_components'][0]['labels'],['opening','transverse_slip','longitudinal_slip'])
    def test_invalid_seam_indices_or_axes_rejected(self):
        with self.assertRaises(ValueError):assembly_diagnostics(np.zeros((3,3)),np.zeros((3,3)),[1]*3,[(0,3)])
        with self.assertRaises(ValueError):assembly_diagnostics(np.zeros((3,3)),np.zeros((3,3)),[1]*3,[(0,1)],seam_axes=[np.zeros((3,3))])
