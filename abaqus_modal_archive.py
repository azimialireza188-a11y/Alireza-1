# -*- coding: utf-8 -*-
"""Single-pass extraction of Abaqus eigenmode U/UR data into numeric NPZ."""
from __future__ import print_function
import json
import math
import re
import numpy as np

_EIGEN = re.compile(r'eigen\s*value\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)', re.I)
_MODE = re.compile(r'\bmode\s*[:=]?\s*(\d+)', re.I)


class MechanicalClassificationUnavailable(RuntimeError):
    pass


def _frame_mode(frame):
    desc = str(getattr(frame, 'description', ''))
    mm = _MODE.search(desc)
    em = _EIGEN.search(desc)
    if not mm or not em:
        return None
    token=em.group(1).replace('D','E').replace('d','e')
    value=float(token)
    mantissa,sep,exponent_text=token.upper().partition('E')
    decimals=len(mantissa.split('.',1)[1]) if '.' in mantissa else 0
    exponent=int(exponent_text) if sep else 0
    resolution=.5*10.**(exponent-decimals)
    try:
        exact=float(frame.frameValue)
        if math.isfinite(exact) and abs(exact-value) <= 1.01*resolution:
            value=exact
    except (AttributeError,TypeError,ValueError):
        pass
    return int(mm.group(1)),value


def _vector(value):
    double = str(getattr(value, 'precision', '')) == 'DOUBLE_PRECISION'
    if double:
        data = getattr(value, 'dataDouble', None)
        system = getattr(value, 'localCoordSystemDouble', None)
    else:
        data = getattr(value, 'data', None)
        system = getattr(value, 'localCoordSystem', None)
    if system is not None:
        try:
            if np.size(system):
                raise ValueError('Modal archive requires global nodal U/UR; local coordinate output is unsupported')
        except TypeError:
            raise ValueError('Modal archive requires global nodal U/UR; local coordinate output is unsupported')
    result = np.asarray(data, dtype=float).ravel()
    if result.size < 3 or not np.all(np.isfinite(result[:3])):
        raise ValueError('Incomplete/nonfinite nodal vector output')
    return result[:3]


def _read_field(field, lookup, count):
    data = np.full((count, 3), np.nan, dtype=float)
    for item in field.values:
        inst = getattr(item, 'instance', None)
        if inst is None:
            continue
        index = lookup.get((inst.name, int(item.nodeLabel)))
        if index is not None:
            data[index] = _vector(item)
    if not np.all(np.isfinite(data)):
        raise ValueError('Incomplete modal nodal output for required mapped nodes')
    return data


def extract_modal_archive(odb, metadata, output_path):
    meta = dict(metadata or {})
    step_name = meta.get('step', 'Buckle')
    if step_name not in odb.steps:
        raise ValueError('ODB is missing step: '+step_name)
    # Abaqus odbAccess.Repository exposes keys()/getitem but is not iterable.
    names = sorted(list(odb.rootAssembly.instances.keys()))
    node_keys, coordinates = [], []
    for name in names:
        for node in sorted(odb.rootAssembly.instances[name].nodes, key=lambda n: int(n.label)):
            node_keys.append((str(name), int(node.label)))
            coordinates.append(tuple(float(x) for x in node.coordinates[:3]))
    if not node_keys:
        raise ValueError('No assembly nodes found for modal archive')
    lookup = {key: i for i, key in enumerate(node_keys)}
    modes, eigenvalues, all_u, all_ur = [], [], [], []
    for frame in odb.steps[step_name].frames:
        parsed = _frame_mode(frame)
        if parsed is None:
            continue
        if 'U' not in frame.fieldOutputs:
            raise MechanicalClassificationUnavailable('MECHANICAL_CLASSIFICATION_UNAVAILABLE_MISSING_U')
        if 'UR' not in frame.fieldOutputs:
            raise MechanicalClassificationUnavailable('MECHANICAL_CLASSIFICATION_UNAVAILABLE_MISSING_UR')
        modes.append(parsed[0]); eigenvalues.append(parsed[1])
        all_u.append(_read_field(frame.fieldOutputs['U'], lookup, len(node_keys)))
        all_ur.append(_read_field(frame.fieldOutputs['UR'], lookup, len(node_keys)))
    if not modes:
        raise ValueError('No eigenmode frames found in step '+step_name)
    meta.update(format='abaqus_modal_archive_v1', step=step_name, instances=names,
                fields=['U', 'UR'], mode_count=len(modes), node_count=len(node_keys))
    np.savez(output_path,
             metadata=np.asarray(json.dumps(meta, sort_keys=True)),
             modes=np.asarray(modes, dtype=np.int64),
             eigenvalues=np.asarray(eigenvalues, dtype=float),
             instances=np.asarray([k[0] for k in node_keys]),
             labels=np.asarray([k[1] for k in node_keys], dtype=np.int64),
             coordinates=np.asarray(coordinates, dtype=float),
             U=np.asarray(all_u, dtype=float),
             UR=np.asarray(all_ur, dtype=float))
    return dict(path=output_path, modes=len(modes), nodes=len(node_keys), fields=['U', 'UR'])


class ModalArchive(object):
    def __init__(self, path):
        self.path = path
        self.data = np.load(path, allow_pickle=False)
        self.metadata = json.loads(str(self.data['metadata'].item()))
        self.instances = np.asarray(self.data['instances']).astype(str)
        self.labels = np.asarray(self.data['labels'], dtype=np.int64)
        self.coordinates = np.asarray(self.data['coordinates'], dtype=float)
        self.node_keys = [(str(n), int(label))
                          for n, label in zip(self.instances, self.labels)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        self.data.close()

    def __len__(self):
        return len(self.data['modes'])

    def read_mode(self, index):
        return dict(mode=int(self.data['modes'][index]),
                    eigenvalue=float(self.data['eigenvalues'][index]),
                    U=np.asarray(self.data['U'][index], dtype=float),
                    UR=np.asarray(self.data['UR'][index], dtype=float))


def _piece_arclength(points, xy):
    """Project one section point onto a source polyline and return (s,distance)."""
    p=np.asarray(points,dtype=float)
    q=np.asarray(xy,dtype=float)
    seg=p[1:]-p[:-1]
    lengths=np.linalg.norm(seg,axis=1)
    if len(seg)==0 or np.any(lengths<=0):
        raise ValueError('Reference piece polyline is invalid')
    cumulative=np.r_[0.,np.cumsum(lengths)]
    best=None
    for i,d in enumerate(seg):
        t=float(np.dot(q-p[i],d)/np.dot(d,d))
        t=min(1.0,max(0.0,t))
        projection=p[i]+t*d
        distance=float(np.linalg.norm(q-projection))
        candidate=(distance,float(cumulative[i]+t*lengths[i]))
        if best is None or candidate<best:
            best=candidate
    return best[1],best[0]


def _cluster_levels(values,tolerance):
    result=[]
    for value in sorted(float(x) for x in values):
        if not result or abs(value-result[-1][-1])>tolerance:
            result.append([value])
        else:
            result[-1].append(value)
    return [float(sum(group)/len(group)) for group in result]


def _interp_section(s_nodes, values, targets, tolerance):
    order=np.argsort(s_nodes)
    s=np.asarray(s_nodes,float)[order]
    v=np.asarray(values,float)[order]
    unique_s=[]; unique_v=[]
    for si,vi in zip(s,v):
        if unique_s and abs(si-unique_s[-1])<=tolerance:
            unique_v[-1]=(unique_v[-1]+vi)/2.0
            unique_s[-1]=(unique_s[-1]+si)/2.0
        else:
            unique_s.append(float(si)); unique_v.append(np.asarray(vi,float))
    s=np.asarray(unique_s,float); v=np.asarray(unique_v,float)
    if len(s)<2:
        raise ValueError('A section station needs at least two mapped shell nodes')
    result=np.empty((len(targets),v.shape[1]),float)
    for j,target in enumerate(targets):
        if target<s[0]-tolerance or target>s[-1]+tolerance:
            raise ValueError('Canonical reference point lies outside available shell section path')
        target=min(max(float(target),float(s[0])),float(s[-1]))
        k=int(np.searchsorted(s,target))
        if k==0:
            result[j]=v[0]
        elif k==len(s):
            result[j]=v[-1]
        elif abs(target-s[k])<=tolerance:
            result[j]=v[k]
        else:
            a,b=k-1,k
            fraction=(target-s[a])/(s[b]-s[a])
            result[j]=(1.0-fraction)*v[a]+fraction*v[b]
    return result


def map_mode_to_reference(archive, mode_index, reference):
    """Map one archived Abaqus mode to canonical-section nodes at every z station.

    Mapping is by each physical piece's source-polyline arclength, not Euclidean
    nearest-neighbour distance, so nearby opposing lips cannot be confused.
    """
    row=archive.read_mode(mode_index)
    coords=np.asarray(archive.coordinates,float)
    if coords.ndim!=2 or coords.shape[1]!=3 or len(coords)!=len(archive.node_keys):
        raise ValueError('Modal archive coordinates are invalid')
    u=np.asarray(row['U'],float); ur=np.asarray(row['UR'],float)
    pieces={p['name']:p for p in reference['pieces']}
    ref_nodes=reference['nodes']
    if not ref_nodes:
        raise ValueError('Canonical reference has no nodes')
    scale=max(1.0,float(np.max(np.abs(coords))))
    ztol=max(1e-7,1e-9*scale)
    stol=max(1e-7,1e-9*scale)

    # All four dependent instances originate from the same longitudinal mesh;
    # use their union and require every physical piece at every retained level.
    zlevels=_cluster_levels(coords[:,2],ztol)
    mapped_u=np.empty((len(zlevels),len(ref_nodes),3),float)
    mapped_ur=np.empty_like(mapped_u)

    ref_index={node['id']:i for i,node in enumerate(ref_nodes)}
    for canonical,piece in pieces.items():
        original=str(piece['original_name'])
        indices=np.where(archive.instances==original)[0]
        if not len(indices):
            raise ValueError('Archive is missing physical piece instance '+original)
        points=np.asarray(piece['points'],float)
        perimeter=float(np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1)))
        section_tol=max(1e-5,1e-6*max(perimeter,1.0))
        snode={}
        for idx in indices:
            sval,dist=_piece_arclength(points,coords[idx,:2])
            if dist>section_tol:
                raise ValueError('ODB node lies off canonical source polyline for '+original)
            snode[int(idx)]=sval
        target_ids=list(piece['node_ids'])
        targets=[]
        for node_id in target_ids:
            node=ref_nodes[ref_index[node_id]]
            sval,dist=_piece_arclength(points,[node['x'],node['y']])
            if dist>section_tol:
                raise ValueError('Canonical node lies off its piece source polyline')
            targets.append(sval)
        for iz,z in enumerate(zlevels):
            at=[int(i) for i in indices if abs(coords[int(i),2]-z)<=ztol]
            if len(at)<2:
                raise ValueError('Physical piece lacks a complete shell section at z=%g' % z)
            svals=[snode[i] for i in at]
            out_u=_interp_section(svals,u[at],targets,stol)
            out_ur=_interp_section(svals,ur[at],targets,stol)
            for local,node_id in enumerate(target_ids):
                j=ref_index[node_id]
                mapped_u[iz,j]=out_u[local]
                mapped_ur[iz,j]=out_ur[local]
    if not np.all(np.isfinite(mapped_u)) or not np.all(np.isfinite(mapped_ur)):
        raise ValueError('Canonical modal mapping produced incomplete values')
    return dict(mode=row['mode'],eigenvalue=row['eigenvalue'],
                z=np.asarray(zlevels,float),U=mapped_u,UR=mapped_ur)




def open_modal_archive(path):
    return ModalArchive(path)
