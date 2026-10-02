import os
import tempfile
import unittest
from unittest import mock
import numpy as np
import builtup_reference_section as section
import fcfsm_reference_basis as f


class FcFSMBasisTests(unittest.TestCase):
    def test_nullspace_safe_solver_matches_pseudoinverse(self):
        k = np.diag([4., 2., 0., 0.])
        rhs = np.array([[8.], [2.], [3.], [0.]])
        solver = f.EnergeticSolver(k)
        np.testing.assert_allclose(solver.solve(rhs), np.linalg.pinv(k) @ rhs, atol=1e-12)
        self.assertEqual(solver.rank, 2)
        self.assertEqual(solver.nullity, 2)
        self.assertEqual(solver.negative_count, 0)

    def test_analytic_force_spaces_are_k_orthogonal(self):
        k = np.diag([2., 3., 4., 5.])
        j = np.eye(4)[:, :2]
        equilibrium = np.array([[1., 0.]])
        basis = f.force_family_basis(k, j, equilibrium)
        self.assertEqual(basis.C_G.shape[1], 1)
        self.assertEqual(basis.C_D.shape[1], 1)
        self.assertEqual(basis.C_L.shape[1], 2)
        self.assertLess(np.linalg.norm(basis.C_L.T @ k @ basis.C_D), 1e-11)
        self.assertLess(np.linalg.norm(basis.C_L.T @ k @ basis.C_G), 1e-11)
        self.assertLess(np.linalg.norm(basis.C_D.T @ k @ basis.C_G), 1e-11)
        r = basis.project(np.ones(4))
        self.assertEqual(r['family'], 'L')
        np.testing.assert_allclose(r['energy_percent'], [100.*9/14, 100.*3/14, 100.*2/14],
                                   atol=1e-10)

    def test_wall_load_units_do_not_change_projected_family_energy(self):
        k = np.diag([2., 3., 4., 5.])
        j = np.eye(4)[:, :2]
        equilibrium = np.array([[1., 0.]])
        a = f.force_family_basis(k, j, equilibrium).project(np.ones(4))
        scale = np.diag([3., 7.])
        b = f.force_family_basis(k, j @ scale, equilibrium @ scale).project(np.ones(4))
        np.testing.assert_allclose(a['energy_percent'], b['energy_percent'], atol=1e-10)

    def reference(self):
        pieces = {
            'P1': [[10., 10., 30., 10.], [30., 10., 30., 30.]],
            'P2': [[-10., 10., -10., 30.], [-10., 30., -30., 30.]],
            'P3': [[-10., -10., -30., -10.], [-30., -10., -30., -30.]],
            'P4': [[10., -10., 10., -30.], [10., -30., 30., -30.]],
        }
        return section.build_reference_section(pieces, 3., 200000., .3, 3600.)

    def test_k0_contains_no_cross_piece_regularization(self):
        ref = self.reference()
        basis = f.build_fcfsm_basis(ref, 1)
        dof_piece = []
        by_node = {n['id']: n['piece'] for n in ref['nodes']}
        n = len(ref['nodes'])
        # Global CUFSM order is [u,v] for all nodes then [w,theta] for all nodes.
        for node in ref['nodes']:
            dof_piece.extend([node['piece'], node['piece']])
        for node in ref['nodes']:
            dof_piece.extend([node['piece'], node['piece']])
        k = basis.K0
        for i in range(k.shape[0]):
            for j in np.flatnonzero(np.abs(k[i]) > 1e-10):
                self.assertEqual(dof_piece[i], dof_piece[j])

    def test_corner_elements_are_in_k0_but_not_wall_load_groups(self):
        ref = self.reference()
        # Mark one element as a corner and remove it from any flat plate group.
        corner = ref['elements'][0]['id']
        ref['corner_elements'] = sorted(set(ref['corner_elements'] + [corner]))
        for group in ref['plate_groups']:
            group['element_ids'] = [eid for eid in group['element_ids'] if eid != corner]
        ref['plate_groups'] = [g for g in ref['plate_groups'] if g['element_ids']]
        basis = f.build_fcfsm_basis(ref, 2)
        self.assertTrue(np.any(np.abs(basis.K0) > 0))
        self.assertEqual(basis.J_GD.shape[1], len(ref['plate_groups']))

    def test_cache_key_ignores_connection_metadata(self):
        ref = self.reference()
        a = f.basis_cache_key(ref['definition_hash'], 7, 'S-S')
        changed = dict(ref); changed['connection_metadata'] = {'bolts':[25.,100.,175.]}
        b = f.basis_cache_key(changed['definition_hash'], 7, 'S-S')
        self.assertEqual(a, b)
        self.assertNotEqual(a, f.basis_cache_key(ref['definition_hash'], 8, 'S-S'))


    def test_disk_cache_reuses_exact_physical_definition_hash(self):
        ref=self.reference()
        with tempfile.TemporaryDirectory() as folder:
            first,status1,path1=f.get_fcfsm_basis(ref,3,'S-S',cache_dir=folder)
            self.assertEqual(status1,'BUILT')
            self.assertTrue(os.path.isfile(path1))
            with mock.patch.object(f,'build_fcfsm_basis',
                                   side_effect=AssertionError('cache miss rebuilt basis')):
                second,status2,path2=f.get_fcfsm_basis(ref,3,'S-S',cache_dir=folder)
            self.assertEqual(status2,'HIT')
            self.assertEqual(path1,path2)
            self.assertEqual(first.definition_hash,second.definition_hash)
            np.testing.assert_allclose(first.K0,second.K0)
            changed=dict(ref)
            changed['definition_hash']='different-physical-hash'
            with mock.patch.object(f,'build_fcfsm_basis',
                                   wraps=f.build_fcfsm_basis) as rebuilt:
                unused,status3,path3=f.get_fcfsm_basis(changed,3,'S-S',cache_dir=folder)
            self.assertEqual(status3,'BUILT')
            self.assertNotEqual(path1,path3)
            rebuilt.assert_called_once()



if __name__ == '__main__':
    unittest.main()
