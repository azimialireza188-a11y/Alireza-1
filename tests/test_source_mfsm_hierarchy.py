import unittest
import tempfile
import os
import numpy as np
from mfsm_model import OperatorPack, SourceDefinition, ModeBatch


def diagonal_fixture():
    # Analytic coordinate directions: GA, GB, GT, D, L, SBt, STt, SDt,
    # complementary transverse shear, TES, TEP, warping shear.
    entries = dict(eps_x=[9, 10], eps_y=[0, 1, 2, 3, 11],
                   gamma_xy=[5, 6, 7, 8, 11], kappa_x=[3, 4, 7, 8],
                   kappa_y=[1, 2, 3, 5, 10], kappa_xy=[2, 6])
    components = {}
    for name, indices in entries.items():
        d = np.zeros(12); d[indices] = 1.
        components[name] = np.diag(d)
    pack = OperatorPack(sum(components.values()) + np.eye(12)*.1,
                        components, SourceDefinition('synthetic, not physical', '2019', 'nu=0'))
    c = np.zeros((2, 12)); c[0, 4] = c[1, 8] = 1.
    definition = dict(coordinate_definition='complete synthetic coordinate space',
                      equilibrium_definition='explicit synthetic Zte, not Abaqus',
                      transverse_equilibrium=c, closed_cells=False,
                      open_shear_selection='REVIEWED_AGGREGATE',
                      open_shear_equations='synthetic disjoint auxiliary space, no physical source claim',
                      open_shear_basis=np.eye(12)[:,[5,6,7,8,11]])
    return pack, definition


class SourceHierarchyTests(unittest.TestCase):
    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('source_mfsm_hierarchy'),
                             'Source-derived hierarchy is not implemented')
        import source_mfsm_hierarchy
        return source_mfsm_hierarchy

    def test_automatic_hierarchy_separates_extension_shear_and_gdl(self):
        m = self.module(); pack, definition = diagonal_fixture()
        basis, sub = m.build_source_hierarchy(pack, definition)
        expected = dict(GA=[0], G=[1, 2], D=[3], L=[4], TE=[9, 10], S=[5, 6, 7, 8, 11])
        for name, indices in expected.items():
            q = basis.spaces[name]
            self.assertEqual(q.shape[1], len(indices), name)
            # Compare spans; eigenvector amplitudes/order are arbitrary.
            from scipy.linalg import orth
            u = orth(q)
            np.testing.assert_allclose(u@u.T, np.eye(12)[:, indices]@np.eye(12)[indices], atol=1e-12)
        np.testing.assert_allclose(definition['transverse_equilibrium']@basis.spaces['D'], 0, atol=1e-12)
        self.assertEqual(sub.spaces['FLEXURAL'].shape[1], 1)
        self.assertEqual(sub.spaces['TORSIONAL'].shape[1], 1)
        from mechanical_modal_classifier import classify_modes
        rows = classify_modes(ModeBatch(np.eye(12)[:, [0, 1, 2, 3, 4]], tuple(range(5))), basis,
                              global_projector=__import__('mechanical_modal_classifier').EnergyProjector(sub))
        self.assertEqual([r['dominant_family'] for r in rows], [None, 'GLOBAL', 'GLOBAL', 'DISTORTIONAL', 'LOCAL'])
        self.assertEqual(rows[1]['global_subtype'], 'FLEXURAL')
        self.assertEqual(rows[2]['global_subtype'], 'TORSIONAL')
        self.assertFalse(rows[1]['scientifically_eligible'])

    def test_appendix_rows_selected_before_constraint_reduction(self):
        m = self.module()
        k = np.diag([2., 3., 4.]); q = np.array([[1., 0.], [0., 1.], [1., 1.]])
        c = m.transverse_equilibrium_rows(k, [2], q)
        np.testing.assert_equal(c, [[4., 4.]])
        with self.assertRaises(ValueError): m.transverse_equilibrium_rows(k, [2, 2], q)
        with self.assertRaises(ValueError): m.transverse_equilibrium_rows(k, [-1], q)

    def test_coordinate_change_preserves_spaces_with_pulled_back_euclidean_metric(self):
        m = self.module(); pack, d = diagonal_fixture()
        base, unused = m.build_source_hierarchy(pack, d)
        rng = np.random.default_rng(4)
        t = np.eye(12)+rng.normal(size=(12, 12))*.03
        changed = OperatorPack(t.T@pack.system@t, {k: t.T@v@t for k, v in pack.components.items()}, pack.source)
        transformed = dict(d, transverse_equilibrium=d['transverse_equilibrium']@t, coordinate_metric=t.T@t,
                           open_shear_basis=np.linalg.solve(t,d['open_shear_basis']))
        other, unused = m.build_source_hierarchy(changed, transformed)
        from scipy.linalg import orth
        for name in base.spaces:
            a = orth(base.spaces[name]); b = orth(t@other.spaces[name])
            np.testing.assert_allclose(a@a.T, b@b.T, atol=1e-10, err_msg=name)

    def test_missing_equilibrium_and_components_rejected(self):
        m = self.module(); pack, d = diagonal_fixture()
        with self.assertRaisesRegex(ValueError, 'equilibrium'):
            m.build_source_hierarchy(pack, {k:v for k,v in d.items() if k!='transverse_equilibrium'})
        pack.components.pop('kappa_xy')
        with self.assertRaisesRegex(ValueError, 'six'):
            m.build_source_hierarchy(pack, d)

    def test_negative_component_rejected_even_if_sum_is_positive(self):
        m = self.module(); pack, d = diagonal_fixture()
        pack.components['eps_x'][0, 0] = -.01
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            m.build_source_hierarchy(pack, d)

    def test_closed_torsion_excludes_warping_only_shear_under_rotated_candidates(self):
        m = self.module()
        components = {k:np.zeros((3, 3)) for k in ('eps_x','eps_y','gamma_xy','kappa_x','kappa_y','kappa_xy')}
        components['eps_y'] = np.eye(3)
        components['gamma_xy'] = np.diag([0., 1., 1.])
        components['kappa_xy'] = np.diag([0., 1., 0.])
        pack = OperatorPack(sum(components.values()), components, SourceDefinition('synthetic', 'Part2 4/12', 'nu0'))
        d = dict(coordinate_definition='complete synthetic space', equilibrium_definition='empty synthetic rows',
                 transverse_equilibrium=np.empty((0, 3)), closed_cells=True,
                 closed_loop_review='synthetic strip split; no physical validation', gamma_open=np.zeros((3, 3)))
        basis, sub = m.build_source_hierarchy(pack, d)
        self.assertEqual(sub.spaces['TORSIONAL'].shape[1], 1)
        from scipy.linalg import orth
        np.testing.assert_allclose(orth(sub.spaces['TORSIONAL'])@orth(sub.spaces['TORSIONAL']).T,
                                   np.diag([0., 1., 0.]), atol=1e-12)
        candidates = np.eye(3)[:, 1:]
        rotation = np.array([[1., 1.], [-1., 1.]])/np.sqrt(2)
        a = m.positive_curvature_subspace(candidates, pack.system, components['kappa_xy'])
        b = m.positive_curvature_subspace(candidates@rotation, pack.system, components['kappa_xy'])
        np.testing.assert_allclose(orth(a)@orth(a).T, orth(b)@orth(b).T, atol=1e-12)
        self.assertEqual(a.shape[1], 1)
        self.assertEqual(basis.spaces['S'].shape[1], 1)
        with self.assertRaisesRegex(ValueError, 'closed'):
            m.build_source_hierarchy(pack, {k:v for k,v in d.items() if k!='gamma_open'})
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            m.build_source_hierarchy(pack, dict(d, gamma_open=np.eye(3)*2))

    def test_cache_binds_equilibrium_tolerance_and_subtype_identity(self):
        m = self.module(); pack, d = diagonal_fixture()
        with tempfile.TemporaryDirectory() as root:
            a, sub, first = m.cached_source_hierarchy(pack, d, root)
            b, sub2, second = m.cached_source_hierarchy(pack, d, root)
            self.assertFalse(first['hit']); self.assertTrue(second['hit'])
            self.assertEqual(sub.definition_id, sub2.definition_id)
            np.testing.assert_equal(a.spaces['G'], b.spaces['G'])
            unused, unused, changed = m.cached_source_hierarchy(pack, d, root, tolerance=1e-9)
            self.assertNotEqual(first['key'], changed['key'])
            d2 = dict(d, transverse_equilibrium=np.zeros((0,12)))
            self.assertNotEqual(m.hierarchy_cache_key(pack, d), m.hierarchy_cache_key(pack, d2))
            with open(first['path'], 'wb') as f: f.write(b'broken')
            with self.assertRaisesRegex(ValueError, 'cache'): m.cached_source_hierarchy(pack, d, root)

    def test_audit_builds_hierarchy_without_supplied_family_bases(self):
        import json
        from types import SimpleNamespace
        from mfsm_audit import evaluate
        pack, d = diagonal_fixture()
        nodes = [SimpleNamespace(label=i+1,coordinates=(float(i),0.,0.)) for i in range(2)]
        instance = SimpleNamespace(name='P',nodes=nodes)
        odb = SimpleNamespace(rootAssembly=SimpleNamespace(instances={'P':instance}))
        frames = {}
        for mode, column in enumerate([0,1,2,3,4], 1):
            x = np.eye(12)[:,column]
            def values(offset):
                return [SimpleNamespace(instance=instance,nodeLabel=i+1,precision='SINGLE_PRECISION',
                            data=x[6*i+offset:6*i+offset+3],localCoordSystem=None) for i in range(2)]
            frames[mode] = SimpleNamespace(fieldOutputs={
                'U':SimpleNamespace(values=values(0)), 'UR':SimpleNamespace(values=values(3))})
        hierarchy = {k:v for k,v in d.items() if k not in ('transverse_equilibrium','open_shear_basis')}
        meta = dict(source_odb_sha256='hash',model_signature='fixture',nu_class=0,
                    contact_status='INACTIVE_VERIFIED',connection_status='ELASTIC_VERIFIED',
                    coordinate_space_review=True,constraint_mapping_review=True,
                    source=dict(reference=pack.source.reference,equations=pack.source.equations,metric_definition=pack.source.metric_definition),
                    components=list(pack.components),basis_construction='KHEZRI_2019_HIERARCHY',hierarchy=hierarchy)
        resources = dict(cpus=2,gpus=0,inventory=dict(logical_cpus=2,physical_cores=2,total_memory_bytes=2**30,available_memory_bytes=2**29,devices=[]))
        summary = dict(source_odb_sha256='hash',model_signature='fixture',
                       modes=[dict(mode=i,eigenvalue=float(i)) for i in frames])
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root,'pack.npz')
            np.savez(path,metadata=json.dumps(meta),instances=['P']*12,labels=[1]*6+[2]*6,
                     dofs=list(range(1,7))*2,coordinates=np.repeat([n.coordinates for n in nodes],6,axis=0),
                     mapping=np.eye(12),reconstruction=np.eye(12),raw_metric_diagonal=np.ones(12),
                     K_system=pack.system,Z_te=d['transverse_equilibrium'],S_open=d['open_shear_basis'],
                     **{'K_'+k:v for k,v in pack.components.items()})
            r = evaluate(odb,frames,summary,path,resources,os.path.join(root,'cache'))
            self.assertEqual(r['source_method'],'MFSM_KHEZRI_2019_OPERATOR_HIERARCHY')
            self.assertEqual(r['modes'][1]['source_method'],r['source_method'])
            self.assertEqual(r['modes'][1]['global_subtype'],'FLEXURAL')
            self.assertEqual(r['modes'][2]['global_subtype'],'TORSIONAL')
            self.assertEqual(r['modes'][3]['dominant_family'],'DISTORTIONAL')
            self.assertFalse(r['scientifically_eligible'])
            other = evaluate(odb,frames,summary,path,resources,os.path.join(root,'cache'))
            self.assertTrue(other['cache']['hit'])
            # Appendix construction uses canonical Kx before Q reduction.
            with np.load(path, allow_pickle=False) as saved:
                payload = {k:saved[k].copy() for k in saved.files if k!='Z_te'}
            meta['hierarchy']['equilibrium_construction'] = 'APPENDIX_A1_ROWS'
            payload.update(metadata=json.dumps(meta), K_kappa_x_canonical=pack.components['kappa_x'],
                           canonical_reduction=np.eye(12), unprescribed_rows=np.array([4,8]))
            np.savez(path, **payload)
            derived = evaluate(odb,frames,summary,path,resources,os.path.join(root,'cache'))
            self.assertEqual(derived['modes'][3]['dominant_family'], 'DISTORTIONAL')

    def test_fsm_node_rows_do_not_prescribe_internal_displacements(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'fsm_unprescribed_rows'), 'Canonical FSM row construction missing')
        # Node-major U,V,W,theta: main nodes 0/2; internal node 1.
        rows = m.fsm_unprescribed_rows(3, [0,2], 2)
        np.testing.assert_equal(rows, [3,4,6,7,11,15,16,18,19,23])
        with self.assertRaises(ValueError): m.fsm_unprescribed_rows(3, [3], 1)

    def test_closed_modified_torsion_preserves_part1_distortional_space(self):
        m=self.module()
        k={name:np.zeros((3,3)) for name in ('eps_x','eps_y','gamma_xy','kappa_x','kappa_y','kappa_xy')}
        k['eps_y']=np.array([[1.,0,0],[0,1,.1],[0,.1,1.]])
        k['gamma_xy']=np.diag([0.,1.,0.]);k['kappa_x']=np.diag([0.,0.,1.]);k['kappa_y']=np.diag([0.,1.,0.])
        p=OperatorPack(sum(k.values()),k,SourceDefinition('synthetic review regression','Part1 126; Part2 Table1','nu0'))
        d=dict(coordinate_definition='complete synthetic space',equilibrium_definition='empty synthetic rows',
               transverse_equilibrium=np.empty((0,3)),closed_cells=True,closed_loop_review='synthetic',gamma_open=np.zeros((3,3)))
        b,unused=m.build_source_hierarchy(p,d)
        self.assertEqual(b.spaces['D'].shape[1],1,'Modified GT must not delete classical Part1 D')
        from mechanical_modal_classifier import classify_modes
        row=classify_modes(ModeBatch(np.eye(3)[:,2:3],(1,)),b)[0]
        self.assertEqual(row['dominant_family'],'DISTORTIONAL')

    def test_open_source_warping_shear_selection_is_not_closed_complement(self):
        m=self.module()
        k={name:np.zeros((2,2)) for name in ('eps_x','eps_y','gamma_xy','kappa_x','kappa_y','kappa_xy')}
        k['eps_y']=np.diag([1.,0.]);k['gamma_xy']=np.array([[1.,-1.],[-1.,1.]])
        k['kappa_y']=np.diag([0.,1.])
        p=OperatorPack(sum(k.values()),k,SourceDefinition('synthetic review regression','Part2 Table1 open','nu0'))
        d=dict(coordinate_definition='V,U synthetic ordering',equilibrium_definition='empty synthetic rows',
               transverse_equilibrium=np.empty((0,2)),closed_cells=False,open_shear_selection='WARPING',
               warping_projection=np.diag([1.,0.]).tolist(),warping_projection_review='canonical V-only projection')
        b,unused=m.build_source_hierarchy(p,d)
        from mechanical_modal_classifier import EnergyProjector
        shares=EnergyProjector(b).project(np.eye(2)[:,0:1])['shares_percent']
        self.assertAlmostEqual(shares['S'][0],100.)
        self.assertAlmostEqual(shares['G'][0],0.)
        with self.assertRaisesRegex(ValueError,'open.*shear'):
            m.build_source_hierarchy(p,{key:value for key,value in d.items() if key!='open_shear_selection'})
        with self.assertRaisesRegex(ValueError,'warping'):
            m.build_source_hierarchy(p,dict(d,warping_projection=np.eye(2)))
