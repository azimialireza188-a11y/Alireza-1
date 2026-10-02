import copy
import unittest
import numpy as np
import mechanical_modal_classifier as m


class StaticBasis(object):
    def __init__(self, diag=(1.,1.,1.,1.)):
        self.K0=np.diag(diag)
        self.metadata={'thickness_mm':3.0}
        self.condition_report={'energetic_condition':1.0,'rank':4,'nullity':0}
    def project(self,q):
        q=np.asarray(q,float)
        parts={
            'L':np.array([q[0],0,0,0.]),
            'D':np.array([0,q[1],0,0.]),
            'G':np.array([0,0,q[2],0.]),
        }
        residual=np.array([0,0,0,q[3]])
        return {'components':parts,'residual':residual}


def harmonic(q,residual=0.0):
    return dict(components={1:{'reference_vector':np.asarray(q,float)}},
                dominant_m=1,half_wavelength_mm=100.,
                relative_residual=float(residual),shares=[1.0])


def diagnostics(assembly=0.,normal=0.,transverse=0.,longitudinal=0.,interaction=0.):
    return dict(assembly_percent=float(assembly),
                normal_opening_index=float(normal),
                transverse_slip_index=float(transverse),
                longitudinal_slip_index=float(longitudinal),
                interpiece_interaction_percent=float(interaction))


class MechanicalClassifierTests(unittest.TestCase):
    def classify(self,q,basis=None,harmonic_residual=0.,diag=None,settings=None):
        basis=basis or StaticBasis()
        return m.classify_mode({'mode':1,'eigenvalue':123.},harmonic(q,harmonic_residual),
                               {1:basis},diag or diagnostics(),settings or {})

    def test_pure_synthetic_families(self):
        for q,expected in [([1,0,0,0],'LOCAL'),([0,1,0,0],'DISTORTIONAL'),
                           ([0,0,1,0],'GLOBAL')]:
            with self.subTest(expected=expected):
                r=self.classify(q)
                self.assertEqual(r['family'],expected)
                self.assertGreater(max(r['energy_percent'][:3]),99.999999)

    def test_known_mixture_below_dominance_is_mixed_with_exact_energy_shares(self):
        q=np.sqrt([.6,.3,.1,0.])
        r=self.classify(q)
        self.assertEqual(r['family'],'MIXED')
        np.testing.assert_allclose(r['energy_percent'],[60,30,10,0],atol=1e-10)

    def test_energy_is_primary_and_metric_disagreement_sets_flag(self):
        r=self.classify([.5,.5,0,0],basis=StaticBasis((100.,1.,1.,1.)))
        self.assertEqual(r['family'],'LOCAL')
        self.assertGreater(r['energy_percent'][0],98.)
        self.assertAlmostEqual(r['vector_percent'][0],50.,places=9)
        self.assertIn('METRIC_SENSITIVE',r['flags'])

    def test_mechanical_residual_over_five_percent_is_unresolved(self):
        r=self.classify(np.sqrt([.94,0,0,.06]))
        self.assertEqual(r['family'],'UNRESOLVED')
        self.assertGreater(r['mechanical_residual_percent'],5.)
        self.assertIn('HIGH_MECHANICAL_RESIDUAL',r['flags'])

    def test_harmonic_fit_poor_is_unresolved(self):
        r=self.classify([1,0,0,0],harmonic_residual=.051)
        self.assertEqual(r['family'],'UNRESOLVED')
        self.assertIn('HARMONIC_FIT_POOR',r['flags'])

    def test_assembly_warning_and_unresolved_gate_do_not_renormalize_families(self):
        warn=self.classify([1,0,0,0],diag=diagnostics(assembly=16.))
        self.assertEqual(warn['family'],'LOCAL')
        self.assertIn('HIGH_ASSEMBLY',warn['flags'])
        self.assertGreater(warn['energy_percent'][0],99.999)
        fail=self.classify([1,0,0,0],diag=diagnostics(assembly=26.))
        self.assertEqual(fail['family'],'UNRESOLVED')
        self.assertIn('HIGH_ASSEMBLY',fail['flags'])
        self.assertGreater(fail['energy_percent'][0],99.999)

    def test_seam_and_interpiece_flags_do_not_reassign_ldg(self):
        r=self.classify([1,0,0,0],diag=diagnostics(normal=20.,interaction=18.))
        self.assertEqual(r['family'],'LOCAL')
        self.assertIn('HIGH_SEAM_RELATIVE_MOTION',r['flags'])
        self.assertIn('INTERPIECE_INTERACTION_HIGH',r['flags'])

    def test_repeated_mode_eigenspace_bounds_are_rotation_scale_invariant(self):
        L=np.array([[1.,0.],[0.,1.],[0.,0.],[0.,0.]])
        Z=np.zeros_like(L)
        rows=[{'mode':1,'eigenvalue':100.,'spectral_isolated':True},
              {'mode':2,'eigenvalue':100.02,'spectral_isolated':True}]
        a=m.classify_eigenspace(rows,{'L':L,'D':Z,'G':Z,'O':Z},{})
        change=np.array([[3.,1.],[0.,2.]])
        b=m.classify_eigenspace(rows,{k:v@change for k,v in {'L':L,'D':Z,'G':Z,'O':Z}.items()},{})
        np.testing.assert_allclose(a['min_percent'],b['min_percent'],atol=1e-10)
        np.testing.assert_allclose(a['max_percent'],b['max_percent'],atol=1e-10)
        self.assertEqual(a['family'],'LOCAL')
        open_rows=copy.deepcopy(rows); open_rows[0]['spectral_isolated']=False
        c=m.classify_eigenspace(open_rows,{'L':L,'D':Z,'G':Z,'O':Z},{})
        self.assertEqual(c['family'],'UNRESOLVED')
        self.assertIn('EIGENSPACE_NOT_ISOLATED',c['flags'])

    def test_parallel_and_serial_are_equivalent(self):
        records=[]
        for i,q in enumerate(([1,0,0,0],[0,1,0,0],[0,0,1,0]),1):
            records.append(dict(mode_record={'mode':i,'eigenvalue':100.+i},
                                harmonic_result=harmonic(q),basis_provider={1:StaticBasis()},
                                diagnostics=diagnostics()))
        serial=m.classify_modes_parallel(records,{}, {'worker_layout':{'processes':1,'blas_threads':1},'gpus':0})
        parallel=m.classify_modes_parallel(records,{}, {'worker_layout':{'processes':3,'blas_threads':1},'gpus':0})
        self.assertEqual([r['family'] for r in serial],[r['family'] for r in parallel])
        np.testing.assert_allclose([r['energy_percent'] for r in serial],
                                   [r['energy_percent'] for r in parallel],atol=1e-12)

    def test_mock_gpu_qc_backend_must_match_cpu_before_acceptance(self):
        class FakeGPU:
            @staticmethod
            def asarray(x): return np.asarray(x)
            @staticmethod
            def asnumpy(x): return np.asarray(x)
            @staticmethod
            def abs(x): return np.abs(x)
            @staticmethod
            def max(x,axis=None): return np.max(x,axis=axis)
        self.assertTrue(m.validate_gpu_backend(FakeGPU))
        records=[dict(mode_record={'mode':1,'eigenvalue':100.},
                      harmonic_result=harmonic([1,0,0,0]),basis_provider={1:StaticBasis()},
                      diagnostics=diagnostics())]
        rows=m.classify_modes_parallel(records,{'gpu_module':FakeGPU},
                                       {'worker_layout':{'processes':1,'blas_threads':1},'gpus':1})
        self.assertEqual(rows[0]['numeric_backend'],'gpu_qc_validated')
        self.assertEqual(rows[0]['family'],'LOCAL')


    def test_unavailable_interpiece_diagnostic_is_not_falsely_reported_as_zero(self):
        diag=diagnostics()
        diag.pop('interpiece_interaction_percent')
        r=self.classify([1,0,0,0],diag=diag)
        self.assertIsNone(r['interpiece_interaction_percent'])
        self.assertNotIn('INTERPIECE_INTERACTION_HIGH',r['flags'])

    def test_zero_energy_secondary_harmonic_is_ignored_not_rejected(self):
        h=harmonic([1,0,0,0])
        h['components'][2]={'reference_vector':np.zeros(4)}
        r=m.classify_mode({'mode':1,'eigenvalue':100.},h,
                          {1:StaticBasis(),2:StaticBasis()},diagnostics(),{})
        self.assertEqual(r['family'],'LOCAL')
        self.assertGreater(r['L_energy_percent'],99.999)

    def test_parallel_classifier_reports_aggregate_done_total_progress(self):
        records=[]
        for i,q in enumerate(([1,0,0,0],[0,1,0,0],[0,0,1,0]),1):
            records.append(dict(mode_record={'mode':i,'eigenvalue':100.+i},
                                harmonic_result=harmonic(q),basis_provider={1:StaticBasis()},
                                diagnostics=diagnostics()))
        seen=[]
        m.classify_modes_parallel(
            records,{}, {'worker_layout':{'processes':2,'blas_threads':1},'gpus':0},
            progress=lambda done,total: seen.append((done,total)))
        self.assertEqual(seen[-1],(3,3))
        self.assertEqual(sorted(done for done,total in seen),[1,2,3])
        self.assertTrue(all(total==3 for done,total in seen))



if __name__=='__main__':
    unittest.main()
