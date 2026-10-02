import unittest
import numpy as np
import builtup_reference_section as section
import assembly_projector as a


class AssemblyProjectorTests(unittest.TestCase):
    def reference(self):
        pieces={
            'P1': [[10.,10.,20.,10.],[20.,10.,20.,20.]],
            'P2': [[-10.,10.,-10.,20.],[-10.,20.,-20.,20.]],
            'P3': [[-10.,-10.,-20.,-10.],[-20.,-10.,-20.,-20.]],
            'P4': [[10.,-10.,10.,-20.],[10.,-20.,20.,-20.]],
        }
        seams=[
            [1,1,2,10,10,-10,10],
            [2,2,3,-10,10,-10,-10],
            [3,3,4,-10,-10,10,-10],
            [4,4,1,10,-10,10,10],
        ]
        return section.build_reference_section(pieces,3.,200000.,.3,3600.,seams=seams)

    def coords(self, ref):
        return np.array([[n['x'],n['y'],0.] for n in ref['nodes']])

    def rigid(self, coords, trans=(0,0,0), omega=(0,0,0)):
        t=np.asarray(trans,float); w=np.asarray(omega,float)
        U=t[None,:]+np.cross(np.broadcast_to(w,coords.shape),coords)
        UR=np.broadcast_to(w,coords.shape).copy()
        return U,UR

    def test_common_rigid_translation_and_rotation_have_zero_assembly(self):
        ref=self.reference(); xyz=self.coords(ref)
        for trans,omega in [((3,-2,1),(0,0,0)),((0,0,0),(.02,-.01,.03))]:
            U,UR=self.rigid(xyz,trans,omega)
            r=a.assembly_diagnostics(dict(U=U,UR=UR),ref)
            self.assertLess(r['assembly_percent'],1e-8)
            seam=a.seam_relative_diagnostics(dict(U=U,UR=UR),ref)
            self.assertLess(seam['normal_opening_index'],1e-8)
            self.assertLess(seam['transverse_slip_index'],1e-8)
            self.assertLess(seam['longitudinal_slip_index'],1e-8)

    def test_equal_opposite_piece_rigid_motion_is_high_assembly(self):
        ref=self.reference(); n=len(ref['nodes'])
        U=np.zeros((n,3)); UR=np.zeros((n,3))
        for i,node in enumerate(ref['nodes']):
            U[i,0]=1. if node['piece'] in ('C1','C3') else -1.
        r=a.assembly_diagnostics(dict(U=U,UR=UR),ref)
        self.assertGreater(r['assembly_percent'],99.999)
        self.assertLess(r['within_piece_deformation_percent'],1e-8)

    def test_within_piece_bending_is_not_redefined_as_assembly(self):
        ref=self.reference(); n=len(ref['nodes'])
        U=np.zeros((n,3)); UR=np.zeros((n,3))
        # zero-mean non-rigid pattern in each piece
        for piece in {x['piece'] for x in ref['nodes']}:
            ids=[i for i,node in enumerate(ref['nodes']) if node['piece']==piece]
            values=np.linspace(-1.,1.,len(ids))
            U[ids,0]=values*values-np.mean(values*values)
        r=a.assembly_diagnostics(dict(U=U,UR=UR),ref)
        self.assertLess(r['assembly_percent'],10.)
        self.assertGreater(r['within_piece_deformation_percent'],90.)

    def test_seam_components_are_resolved_separately(self):
        ref=self.reference(); n=len(ref['nodes'])
        for component,key in [('normal','normal_opening_index'),
                              ('tangent','transverse_slip_index'),
                              ('longitudinal','longitudinal_slip_index')]:
            U=np.zeros((n,3)); UR=np.zeros((n,3))
            a.impose_test_seam_motion(U,ref,seam_id=1,component=component,magnitude=1.)
            r=a.seam_relative_diagnostics(dict(U=U,UR=UR),ref)
            self.assertGreater(r[key],0.)
            others={'normal_opening_index','transverse_slip_index','longitudinal_slip_index'}-{key}
            for other in others:
                self.assertLess(r[other],1e-8)

    def test_diagnostics_are_sign_and_scale_invariant(self):
        ref=self.reference(); n=len(ref['nodes'])
        U=np.zeros((n,3)); UR=np.zeros((n,3))
        a.impose_test_seam_motion(U,ref,1,'normal',2.)
        x=a.assembly_diagnostics(dict(U=U,UR=UR),ref)
        y=a.assembly_diagnostics(dict(U=-7*U,UR=-7*UR),ref)
        self.assertAlmostEqual(x['assembly_percent'],y['assembly_percent'],places=10)


if __name__=='__main__':
    unittest.main()
