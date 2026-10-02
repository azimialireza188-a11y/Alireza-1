import copy
import unittest
import builtup_reference_section as r


class ReferenceSectionTests(unittest.TestCase):
    def square_pieces(self):
        return {
            'P1': [[10., 10., 20., 10.], [20., 10., 20., 20.]],
            'P2': [[-10., 10., -10., 20.], [-10., 20., -20., 20.]],
            'P3': [[-10., -10., -20., -10.], [-20., -10., -20., -20.]],
            'P4': [[10., -10., 10., -20.], [10., -20., 20., -20.]],
        }

    def test_reference_has_no_cross_gap_elements(self):
        ref = r.build_reference_section(self.square_pieces(), 3., 200000., .3, 3600.)
        node_piece = {n['id']: n['piece'] for n in ref['nodes']}
        self.assertEqual(len(ref['pieces']), 4)
        self.assertTrue(ref['elements'])
        for e in ref['elements']:
            self.assertEqual(node_piece[e['n1']], node_piece[e['n2']])
            self.assertEqual(node_piece[e['n1']], e['piece'])

    def test_piece_numbering_does_not_change_physical_definition_hash(self):
        pieces = self.square_pieces()
        a = r.build_reference_section(pieces, 3., 200000., .3, 3600.)
        renamed = {'X4': pieces['P2'], 'X1': pieces['P4'], 'X9': pieces['P1'], 'X2': pieces['P3']}
        b = r.build_reference_section(renamed, 3., 200000., .3, 3600.)
        self.assertEqual(a['definition_hash'], b['definition_hash'])

    def test_material_thickness_or_geometry_changes_hash(self):
        pieces = self.square_pieces()
        base = r.build_reference_section(pieces, 3., 200000., .3, 3600.)['definition_hash']
        self.assertNotEqual(base, r.build_reference_section(pieces, 2., 200000., .3, 3600.)['definition_hash'])
        self.assertNotEqual(base, r.build_reference_section(pieces, 3., 210000., .3, 3600.)['definition_hash'])
        changed = copy.deepcopy(pieces); changed['P1'][0][0] += .25
        self.assertNotEqual(base, r.build_reference_section(changed, 3., 200000., .3, 3600.)['definition_hash'])

    def test_curved_corner_segments_remain_in_stiffness_mesh_but_are_flagged(self):
        pieces = self.square_pieces()
        pieces['P1'] = [
            [10.,0.,30.,0.], [30.,0.,31.,.2], [31.,.2,31.8,.8],
            [31.8,.8,32.,1.8], [32.,1.8,32.,22.]]
        ref = r.build_reference_section(pieces, 3., 200000., .3, 3600.)
        corners = set(ref['corner_elements'])
        self.assertTrue(corners)
        self.assertTrue(corners.issubset({e['id'] for e in ref['elements']}))
        self.assertTrue(any(e['id'] in corners for e in ref['elements']))
        self.assertTrue(ref['plate_groups'])

    def test_connection_metadata_does_not_change_family_definition_hash(self):
        pieces = self.square_pieces()
        a = r.build_reference_section(pieces, 3., 200000., .3, 3600.,
                                      connection_metadata={'bolt_positions_mm':[25,225]})
        b = r.build_reference_section(pieces, 3., 200000., .3, 3600.,
                                      connection_metadata={'bolt_positions_mm':[25,125,225]})
        self.assertEqual(a['definition_hash'], b['definition_hash'])


    def test_fcfsm_plate_groups_split_nonparallel_smooth_wall_segments(self):
        pieces=self.square_pieces()
        # Gentle, non-corner waviness: physical-wall visualization may keep this
        # as one wall, but fcFSM SecAnal defines a plate only from parallel strips.
        pieces['P1']=[
            [10.,10.,20.,10.],
            [20.,10.,30.,10.5],
            [30.,10.5,40.,11.5],
            [40.,11.5,50.,13.0],
        ]
        ref=r.build_reference_section(pieces,3.,200000.,.3,3600.)
        p1=ref['original_to_canonical']['P1']
        groups=[g for g in ref['plate_groups'] if g['piece']==p1]
        self.assertEqual(len(groups),4)
        self.assertTrue(all(len(g['element_ids'])==1 for g in groups))
        self.assertEqual(ref['plate_definition'],
                         'fcFSM parallel-adjacent flat strips excluding curved-corner strips')



if __name__ == '__main__':
    unittest.main()
