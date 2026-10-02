# -*- coding: utf-8 -*-
"""Canonical, bolt-independent four-piece reference-section geometry."""
from __future__ import print_function
import hashlib
import json
import math
import numpy as np
from abaqus_physical_walls import physical_wall_layout


def _points_from_segments(segments):
    rows = [tuple(float(v) for v in row[:4]) for row in segments]
    if not rows:
        raise ValueError('Each piece needs at least one segment')
    points = [rows[0][:2]]
    for i, row in enumerate(rows):
        if i and math.hypot(points[-1][0]-row[0], points[-1][1]-row[1]) > 1e-6:
            raise ValueError('Disconnected piece source polyline')
        if math.hypot(row[2]-row[0], row[3]-row[1]) <= 1e-12:
            raise ValueError('Zero-length reference segment')
        points.append(row[2:4])
    a = tuple(round(x, 12) for p in points for x in p)
    bpoints = list(reversed(points))
    b = tuple(round(x, 12) for p in bpoints for x in p)
    return bpoints if b < a else points


def _canonical_piece_records(section_segments):
    records = []
    for original, segments in section_segments.items():
        points = _points_from_segments(segments)
        xy = np.asarray(points, dtype=float)
        centroid = np.mean(xy, axis=0)
        angle = math.atan2(float(centroid[1]), float(centroid[0]))
        if angle < 0:
            angle += 2*math.pi
        records.append(dict(original=str(original), points=points,
                            sort_key=(round(angle, 12), round(float(np.linalg.norm(centroid)), 12),
                                      tuple(round(x, 9) for p in points for x in p))))
    records.sort(key=lambda r: r['sort_key'])
    for i, record in enumerate(records, 1):
        record['piece'] = 'C%d' % i
    return records


PLATE_DEFINITION = 'fcFSM parallel-adjacent flat strips excluding curved-corner strips'
PLATE_DEFINITION_VERSION = 'fcfsm_parallel_adjacent_v1'
CORNER_DEFINITION_VERSION = 'circular_same_sign_bend_v1'


def _parallel(a, b, tolerance=1e-4):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na<=0 or nb<=0:
        return False
    ua=a/na; ub=b/nb
    return bool(np.max(np.abs(ua-ub)) < tolerance or
                np.max(np.abs(ua+ub)) < tolerance)


def _circular_corner_element_indices(points, minimum_total_turn_deg=30.0,
                                     radial_tolerance=2e-3):
    """Identify discretized circular bends independent of bend radius.

    A CUFSM cornerStrip is a strip belonging to a curved corner. Source bends
    are circular, while optimized leg waviness is not assumed to be a corner.
    Candidate same-sign turning runs are therefore accepted only when their
    interior strip nodes fit one circle to a tight relative radial tolerance.
    """
    p=np.asarray(points,dtype=float)
    if len(p)<4:
        return set()
    vectors=np.diff(p,axis=0)
    lengths=np.linalg.norm(vectors,axis=1)
    if np.any(lengths<=0):
        raise ValueError('Corner detection requires nonzero source segments')
    turn=np.arctan2(vectors[:-1,0]*vectors[1:,1]-vectors[:-1,1]*vectors[1:,0],
                    np.sum(vectors[:-1]*vectors[1:],axis=1))
    active=np.abs(turn)>1e-7
    runs=[]; current=[]
    for j,value in enumerate(turn):
        if not active[j]:
            if current:
                runs.append(current); current=[]
            continue
        if current and value*turn[current[-1]]<=0:
            runs.append(current); current=[]
        current.append(j)
    if current:
        runs.append(current)

    corner=set()
    minimum=math.radians(float(minimum_total_turn_deg))
    for run in runs:
        if abs(float(np.sum(turn[run]))) < minimum:
            continue
        first=run[0]+1
        last=run[-1]  # inclusive element index; excludes tangent straight strips
        if last < first:
            continue  # a single sharp vertex has no finite-radius corner strip
        arc_points=p[first:last+2]
        if len(arc_points)<3:
            continue
        x=arc_points[:,0]; y=arc_points[:,1]
        A=np.column_stack((x,y,np.ones(len(arc_points))))
        rhs=-(x*x+y*y)
        coef,unused_resid,rank,unused_s=np.linalg.lstsq(A,rhs,rcond=None)
        if rank<3:
            continue
        center=-.5*coef[:2]
        radius2=float(center@center-coef[2])
        if radius2<=0 or not math.isfinite(radius2):
            continue
        radius=math.sqrt(radius2)
        radial=np.linalg.norm(arc_points-center[None,:],axis=1)
        error=float(np.max(np.abs(radial-radius)))/max(radius,1e-250)
        if error <= float(radial_tolerance):
            corner.update(range(first,last+1))
    return corner


def _fcfsm_plate_groups(points, element_ids, corner_ids, piece):
    """Match SecAnal_fcFSM: connected parallel strips form one flat plate."""
    points=np.asarray(points,dtype=float)
    groups=[]; current=[]
    for i,eid in enumerate(element_ids):
        if eid in corner_ids:
            if current:
                groups.append(dict(piece=piece,element_ids=current)); current=[]
            continue
        if not current:
            current=[eid]; continue
        prev_index=i-1
        prev_eid=element_ids[prev_index]
        if prev_eid in corner_ids:
            groups.append(dict(piece=piece,element_ids=current)); current=[eid]; continue
        prev=points[prev_index+1]-points[prev_index]
        this=points[i+1]-points[i]
        if _parallel(prev,this):
            current.append(eid)
        else:
            groups.append(dict(piece=piece,element_ids=current)); current=[eid]
    if current:
        groups.append(dict(piece=piece,element_ids=current))
    return groups


def _hash_payload(records, thickness_mm, E_MPa, nu, length_mm):
    payload = dict(
        version='builtup_reference_v2',
        plate_definition_version=PLATE_DEFINITION_VERSION,
        corner_definition_version=CORNER_DEFINITION_VERSION,
        pieces=[[[round(float(x), 12) for x in p] for p in r['points']] for r in records],
        thickness_mm=round(float(thickness_mm), 12),
        E_MPa=round(float(E_MPa), 8),
        nu=round(float(nu), 12),
        length_mm=round(float(length_mm), 8),
        cross_gap_constraints='none')
    text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest(), payload


def build_reference_section(section_segments, thickness_mm, E_MPa, nu, length_mm,
                            seams=None, connection_metadata=None):
    if len(section_segments) != 4:
        raise ValueError('Whole built-up reference requires exactly four physical pieces')
    for name, value in (('thickness_mm', thickness_mm), ('E_MPa', E_MPa),
                        ('length_mm', length_mm)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError('%s must be positive and finite' % name)
    if not math.isfinite(float(nu)) or not -1.0 < float(nu) < .5:
        raise ValueError('nu must be finite and physically admissible')

    records = _canonical_piece_records(section_segments)
    definition_hash, payload = _hash_payload(records, thickness_mm, E_MPa, nu, length_mm)
    nodes, elements, plate_groups, corner_ids = [], [], [], set()
    original_to_canonical = {}
    node_id = 1
    element_id = 1
    piece_info = []

    for record in records:
        piece = record['piece']
        original_to_canonical[record['original']] = piece
        points = np.asarray(record['points'], dtype=float)
        local_node_ids = []
        for x, y in points:
            local_node_ids.append(node_id)
            nodes.append(dict(id=node_id, piece=piece, x=float(x), y=float(y)))
            node_id += 1
        layout = physical_wall_layout(points)
        local_element_ids = []
        for i in range(len(points)-1):
            eid = element_id
            local_element_ids.append(eid)
            elements.append(dict(id=eid, piece=piece, n1=local_node_ids[i],
                                 n2=local_node_ids[i+1],
                                 length_mm=float(np.linalg.norm(points[i+1]-points[i])),
                                 thickness_mm=float(thickness_mm), corner=False))
            element_id += 1
        corner_local=set()
        for lo, hi in layout['bends']:
            corner_local.update(i for i in range(int(lo),int(hi))
                                if 0 <= i < len(local_element_ids))
        corner_local.update(_circular_corner_element_indices(points))
        for i in sorted(corner_local):
            if 0 <= i < len(local_element_ids):
                corner_ids.add(local_element_ids[i])
        plate_groups.extend(_fcfsm_plate_groups(
            points,local_element_ids,corner_ids,piece))
        piece_info.append(dict(name=piece, original_name=record['original'],
                               node_ids=local_node_ids, element_ids=local_element_ids,
                               points=points.tolist()))

    for element in elements:
        if element['id'] in corner_ids:
            element['corner'] = True

    seam_pairs = []
    for row in seams or []:
        if len(row) < 7:
            raise ValueError('Seam rows require id,pieceA,pieceB,xA,yA,xB,yB')
        sid, pa, pb, xa, ya, xb, yb = row[:7]
        pa_key, pb_key = str(int(pa)) if float(pa).is_integer() else str(pa), str(int(pb)) if float(pb).is_integer() else str(pb)
        # Match common P1-style source names as well as raw numeric keys.
        ca = original_to_canonical.get(pa_key, original_to_canonical.get('P'+pa_key))
        cb = original_to_canonical.get(pb_key, original_to_canonical.get('P'+pb_key))
        seam_pairs.append(dict(id=int(sid), piece_a=ca, piece_b=cb,
                               point_a=[float(xa), float(ya)], point_b=[float(xb), float(yb)]))

    return dict(
        version='builtup_reference_v2',
        plate_definition=PLATE_DEFINITION,
        plate_definition_version=PLATE_DEFINITION_VERSION,
        corner_definition_version=CORNER_DEFINITION_VERSION,
        pieces=piece_info,
        nodes=nodes,
        elements=elements,
        plate_groups=plate_groups,
        corner_elements=sorted(corner_ids),
        seam_pairs=seam_pairs,
        material=dict(thickness_mm=float(thickness_mm), E_MPa=float(E_MPa), nu=float(nu)),
        length_mm=float(length_mm),
        definition_hash=definition_hash,
        definition_payload=payload,
        connection_metadata=dict(connection_metadata or {}),
        cross_gap_constraints=[],
        original_to_canonical=original_to_canonical)
