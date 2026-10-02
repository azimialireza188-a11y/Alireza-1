import unittest
import numpy as np
import abaqus_modal_harmonics as h


class ModalHarmonicTests(unittest.TestCase):
    def synthetic(self, z, length, terms, nodes=3):
        u = np.zeros((len(z), nodes, 3))
        ur = np.zeros((len(z), nodes, 3))
        shape = np.arange(1., nodes+1.)
        for m, amp in terms:
            s = np.sin(m*np.pi*z/length)
            c = np.cos(m*np.pi*z/length)
            u[:,:,0] += amp*s[:,None]*shape[None,:]
            u[:,:,1] += .4*amp*s[:,None]*shape[None,:]
            u[:,:,2] += .2*amp*c[:,None]*shape[None,:]
            ur[:,:,0] += .1*amp*c[:,None]*shape[None,:]
            ur[:,:,1] += .15*amp*c[:,None]*shape[None,:]
            ur[:,:,2] += .3*amp*s[:,None]*shape[None,:]
        return u, ur

    def test_pure_ss_harmonic_is_recovered_from_u_and_ur(self):
        length=3600.; z=np.linspace(0,length,81)
        u,ur=self.synthetic(z,length,[(7,2.5)])
        r=h.decompose_mode(z,u,ur,length,12)
        self.assertEqual(r['dominant_m'],7)
        self.assertAlmostEqual(r['half_wavelength_mm'],length/7.)
        self.assertLess(r['relative_residual'],1e-11)
        np.testing.assert_allclose(r['components'][7]['U'][:,0],2.5*np.arange(1.,4.),atol=1e-10)
        np.testing.assert_allclose(r['components'][7]['UR'][:,2],.75*np.arange(1.,4.),atol=1e-10)

    def test_known_multi_harmonic_mixture_and_irregular_grid(self):
        length=1000.
        z=np.unique(np.r_[0., np.linspace(13.,987.,119)**1.001/987.**.001, length])
        u,ur=self.synthetic(z,length,[(7,2.),(9,1.)],nodes=2)
        r=h.decompose_mode(z,u,ur,length,12)
        self.assertEqual(r['dominant_m'],7)
        self.assertLess(r['relative_residual'],1e-9)
        ratio=r['translation_power'][6]/r['translation_power'][8]
        self.assertAlmostEqual(ratio,4.,places=7)

    def test_sign_and_scale_do_not_change_harmonic_shares(self):
        z=np.linspace(0,100.,51); u,ur=self.synthetic(z,100.,[(2,1.),(5,.5)])
        a=h.decompose_mode(z,u,ur,100.,8)
        b=h.decompose_mode(z,-7*u,-7*ur,100.,8)
        np.testing.assert_allclose(a['shares'],b['shares'],atol=1e-12)

    def test_ss_dof_convention_uses_sine_for_transverse_and_cosine_for_axial(self):
        z=np.array([0.,25.,50.,75.,100.])
        functions=h.ss_functions(z,100.,1)
        np.testing.assert_allclose(functions['transverse'],np.sin(np.pi*z/100.))
        np.testing.assert_allclose(functions['axial'],np.cos(np.pi*z/100.))
        self.assertAlmostEqual(functions['transverse'][0],0.)
        self.assertAlmostEqual(functions['transverse'][-1],0.)
        self.assertAlmostEqual(functions['axial'][0],1.)
        self.assertAlmostEqual(functions['axial'][-1],-1.)

    def test_invalid_or_underresolved_grid_is_rejected(self):
        u=np.zeros((3,2,3)); ur=np.zeros_like(u)
        with self.assertRaises(ValueError):
            h.decompose_mode([0,50,100],u,ur,100.,10)


if __name__ == '__main__':
    unittest.main()
