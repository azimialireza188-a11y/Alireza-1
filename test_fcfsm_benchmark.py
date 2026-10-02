import copy
import json
import os
import tempfile
import unittest
import numpy as np
import verify_fcfsm_classifier_benchmark as b
import fcfsm_reference_basis as fb


def fixture():
    common=dict(
        schema_version=1,
        source=dict(program='CUFSM',version='5.70',method='fcFSM'),
        geometry=dict(name='simple_channel',nodes=[[0.,0.],[50.,0.],[50.,100.]],thickness_mm=2.0),
        material=dict(E_MPa=200000.,nu=.3),
        boundary_conditions=dict(longitudinal='S-S'),
        harmonic=dict(m=2,length_mm=1000.))
    reference=copy.deepcopy(common)
    reference['families']={
        'L':dict(share_percent=70.,basis=[[0.],[0.],[0.],[1.]]),
        'D':dict(share_percent=20.,basis=[[0.],[1.],[0.],[0.]]),
        'G':dict(share_percent=10.,basis=[[1.],[0.],[0.],[0.]])}
    classifier=copy.deepcopy(common)
    classifier['source']=dict(program='Python Stage A',version='test',method='FCFSM_K0_ENERGY')
    classifier['families']={
        'L':dict(share_percent=70.2,basis=[[0.],[0.],[0.],[2.]]),
        'D':dict(share_percent=19.9,basis=[[0.],[3.],[0.],[0.]]),
        'G':dict(share_percent=9.9,basis=[[-4.],[0.],[0.],[0.]])}
    return reference,classifier


class FcfsmBenchmarkTests(unittest.TestCase):
    def test_analytic_synthetic_ldg_basis_recovers_all_families(self):
        result=b.run_synthetic_benchmarks()
        self.assertTrue(result['passed'])
        self.assertGreater(result['cases']['pure_L']['L_percent'],99.999999)
        self.assertGreater(result['cases']['pure_D']['D_percent'],99.999999)
        self.assertGreater(result['cases']['pure_G']['G_percent'],99.999999)
        self.assertLess(result['maximum_cross_family_energy_percent'],1e-8)

    def test_reference_fixture_requires_complete_cufsm_provenance(self):
        reference,classifier=fixture()
        checked=b.validate_reference_fixture(reference)
        self.assertEqual(checked['source']['program'],'CUFSM')
        broken=copy.deepcopy(reference); del broken['source']['version']
        with self.assertRaisesRegex(ValueError,'source.version'):
            b.validate_reference_fixture(broken)

    def test_share_and_subspace_agreement_are_both_reported(self):
        reference,classifier=fixture()
        result=b.compare_with_cufsm_reference(reference,classifier,
                                               share_tolerance_pp=.5,
                                               minimum_cosine_squared=.999)
        self.assertTrue(result['compatible'])
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['families']['L']['share_difference_pp'],.2)
        for family in 'LDG':
            self.assertGreater(result['families'][family]['minimum_cosine_squared'],.999999999)

    def test_incompatible_geometry_material_bc_or_harmonic_is_rejected(self):
        keys=[
            ('geometry','thickness_mm',3.0),
            ('material','E_MPa',199000.),
            ('boundary_conditions','longitudinal','C-F'),
            ('harmonic','m',3)]
        for group,key,value in keys:
            with self.subTest(group=group,key=key):
                reference,classifier=fixture()
                classifier[group][key]=value
                with self.assertRaisesRegex(ValueError,'incompatible'):
                    b.compare_with_cufsm_reference(reference,classifier)

    def test_json_paths_are_supported_and_report_contains_source_provenance(self):
        reference,classifier=fixture()
        with tempfile.TemporaryDirectory() as folder:
            rp=os.path.join(folder,'cufsm.json'); cp=os.path.join(folder,'classifier.json')
            with open(rp,'w') as f: json.dump(reference,f)
            with open(cp,'w') as f: json.dump(classifier,f)
            result=b.compare_with_cufsm_reference(rp,cp)
        self.assertEqual(result['reference_source']['version'],'5.70')
        self.assertEqual(result['harmonic']['m'],2)


    def test_native_reference_can_generate_stage_a_fixture_from_actual_implementation(self):
        nodes=[[0.,0.],[50.,0.],[50.,100.],[0.,100.]]
        thickness=2.0; E=200000.; nu=.3; length=1000.; harmonic=2
        reference_model=b.reference_model_from_fixture(dict(
            geometry=dict(name='open_channel',nodes=nodes,thickness_mm=thickness,
                          elements=[[1,2],[2,3],[3,4]],corner_element_ids=[]),
            material=dict(E_MPa=E,nu=nu),
            boundary_conditions=dict(longitudinal='S-S'),
            harmonic=dict(m=harmonic,length_mm=length)))
        basis=fb.build_fcfsm_basis(reference_model,harmonic)
        probe=np.linspace(.1,1.6,4*len(nodes))
        projected=basis.project(probe)
        native=dict(
            schema_version=2,
            source=dict(program='CUFSM',version='5.70',method='native test fixture'),
            geometry=dict(name='open_channel',nodes=nodes,thickness_mm=thickness,
                          elements=[[1,2],[2,3],[3,4]],corner_element_ids=[]),
            material=dict(E_MPa=E,nu=nu),
            boundary_conditions=dict(longitudinal='S-S'),
            harmonic=dict(m=harmonic,length_mm=length),
            probe_vector=probe.tolist(),
            K0=basis.K0.tolist(),
            families={
                'L':dict(share_percent=projected['energy_percent'][0],basis=basis.C_L.tolist()),
                'D':dict(share_percent=projected['energy_percent'][1],basis=basis.C_D.tolist()),
                'G':dict(share_percent=projected['energy_percent'][2],basis=basis.C_G.tolist())})
        classifier=b.classifier_fixture_from_cufsm_reference(native)
        result=b.compare_with_cufsm_reference(
            native,classifier,share_tolerance_pp=1e-8,
            minimum_cosine_squared=.999999999,k0_relative_tolerance=1e-10)
        self.assertTrue(result['passed'])
        self.assertLess(result['k0_relative_frobenius_error'],1e-12)
        self.assertEqual(classifier['source']['method'],'FCFSM_K0_ENERGY')

    def test_end_to_end_reference_requires_probe_k0_and_explicit_connectivity(self):
        reference,classifier=fixture()
        reference['schema_version']=2
        for missing in ('probe_vector','K0'):
            broken=copy.deepcopy(reference)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError,missing):
                b.classifier_fixture_from_cufsm_reference(broken)
        broken=copy.deepcopy(reference)
        broken['probe_vector']=[0.]*4
        broken['K0']=np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError,'geometry.elements'):
            b.classifier_fixture_from_cufsm_reference(broken)



    def test_repository_contains_native_cufsm_exporter_using_real_570_functions(self):
        root=os.path.dirname(os.path.abspath(__file__))
        path=os.path.join(root,'benchmark_fcfsm_classifier_cufsm.m')
        self.assertTrue(os.path.isfile(path))
        with open(path,encoding='utf-8') as stream:
            source=stream.read()
        for token in ('SecAnal_fcFSM','klocal(','trans(','assemble(','jsonencode'):
            self.assertIn(token,source)
        self.assertIn("cufsm-git-5.70",source)
        self.assertIn("'schema_version',2",source.replace(' ',''))



if __name__=='__main__':
    unittest.main()
