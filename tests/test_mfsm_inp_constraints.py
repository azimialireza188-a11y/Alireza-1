import importlib.util
import unittest
import numpy as np


FIXTURE = '''*Part, name=Shell
*Node
1, 0,0,0
2, 1,0,0
3, 1,1,0
4, 0,1,0
*Element, type=S4R
1,1,2,3,4
*Shell Section, elset=ALL, material=Steel
2,5
*End Part
*Assembly, name=A
*Instance, name=P1, part=Shell
*End Instance
*Instance, name=P2, part=Shell
2,0,0
0,0,0,0,0,1,90
*End Instance
*Nset,nset=A_NODE,instance=P1
1
*Nset,nset=B_NODE,instance=P2
1
*MPC
BEAM,B_NODE,A_NODE
*End Assembly
*Boundary
A_NODE,3,3
*Contact
*Contact Inclusions, ALL EXTERIOR
*Step,name=Buckle,perturbation,nlgeom=NO
*Buckle
250,,500,1250
*Boundary,op=NEW,load case=1
A_NODE,1,1
*Boundary,op=NEW,load case=2
A_NODE,2,2
*Output,field,variable=PRESELECT
*End Step
'''


class InputConstraintTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('abaqus_mfsm_input'), 'INP constraint extraction missing')
        import abaqus_mfsm_input
        return abaqus_mfsm_input

    def test_instance_transforms_and_sparse_beam_offsets(self):
        m=self.module(); model=m.read_input_text(FIXTURE)
        np.testing.assert_allclose(model['nodes'][('P2',1)],[0,2,0],atol=1e-14)
        c,keys,meta=m.compile_initial_constraints(model)
        self.assertEqual(c.shape,(7,48))
        self.assertEqual(meta['beam_rows'],6)
        self.assertEqual(meta['boundary_rows'],1)
        # Global infinitesimal rigid rotation satisfies offset-aware BEAM rows.
        omega=np.array([.4,.2,-.1]);u=[]
        for name,label in meta['node_keys']:
            u.extend(np.cross(omega,model['nodes'][(name,label)]));u.extend(omega)
        np.testing.assert_allclose(c[:6]@u,0,atol=1e-14)
        boundary=c[-1].toarray().ravel()
        self.assertEqual(np.flatnonzero(boundary).tolist(),[1])
        self.assertFalse(meta['active_base_state_verified'])

    def test_contact_is_unknown_and_preselect_is_not_missing_rotations(self):
        m=self.module(); model=m.read_input_text(FIXTURE)
        r=m.input_summary(model)
        self.assertEqual(r['contact_state'],'UNKNOWN_REQUIRES_BASE_STATE_EVIDENCE')
        self.assertEqual(r['rotation_output_status'],'REQUIRES_ODB_FIELD_INSPECTION')
        self.assertEqual(r['requested_modes'],250)
        self.assertFalse(r['scientifically_eligible'])

    def test_rejects_includes_local_axes_and_unsupported_mpc(self):
        m=self.module()
        for text in (FIXTURE+'*Include,input=other.inp\n',FIXTURE+'*Transform,nset=A_NODE\n1,0,0,0,1,0\n',
                     FIXTURE.replace('BEAM,B_NODE,A_NODE','PIN,B_NODE,A_NODE')):
            with self.subTest(text=text[-50:]),self.assertRaises(ValueError):
                model=m.read_input_text(text);m.compile_initial_constraints(model)

    def test_duplicate_dependent_nodes_and_multinode_sets_rejected(self):
        m=self.module()
        for text in (FIXTURE.replace('*End Assembly','*MPC\nBEAM,B_NODE,A_NODE\n*End Assembly'),
                     FIXTURE.replace('*Nset,nset=B_NODE,instance=P2\n1','*Nset,nset=B_NODE,instance=P2\n1,2')):
            with self.assertRaises(ValueError):m.compile_initial_constraints(m.read_input_text(text))

    def test_nonzero_mode_bcs_and_missing_labels_rejected(self):
        m=self.module()
        for text in (FIXTURE.replace('A_NODE,2,2','A_NODE,2,2,1.'),
                     FIXTURE.replace('*Nset,nset=B_NODE,instance=P2\n1','*Nset,nset=B_NODE,instance=P2\n99')):
            with self.assertRaises(ValueError):m.compile_initial_constraints(m.read_input_text(text))

    def test_generated_sets_and_no_loadcase2_preserve_initial_bcs(self):
        m=self.module()
        text=FIXTURE[:FIXTURE.index('*Step')]
        text=text.replace('*Nset,nset=A_NODE,instance=P1\n1','*Nset,nset=A_NODE,instance=P1,internal,generate\n1,1,1')
        c,keys,meta=m.compile_initial_constraints(m.read_input_text(text))
        self.assertEqual(c.shape,(7,48))
        self.assertEqual(np.flatnonzero(c[-1].toarray()).tolist(),[2])

    def test_unsupported_coordinate_options_and_geometry_keywords_fail_closed(self):
        m=self.module()
        for suffix in ('SYSTEM=C','SYSTEM=S','INPUT=nodes.txt'):
            with self.subTest(option=suffix),self.assertRaises(ValueError):
                m.read_input_text(FIXTURE.replace('*Node\n','*Node,'+suffix+'\n'))
        for keyword in ('NMAP','NGEN','NCOPY','NFILL','ELGEN','UNKNOWN GEOMETRY'):
            with self.subTest(keyword=keyword),self.assertRaises(ValueError):
                m.read_input_text(FIXTURE+'*'+keyword+'\n')
