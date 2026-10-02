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
    value = float(em.group(1).replace('D', 'E').replace('d', 'e'))
    try:
        exact = float(frame.frameValue)
        if math.isfinite(exact):
            value = exact
    except Exception:
        pass
    return int(mm.group(1)), value


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
    names = sorted(odb.rootAssembly.instances)
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
        self.node_keys = [(str(n), int(label))
                          for n, label in zip(self.data['instances'], self.data['labels'])]

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


def open_modal_archive(path):
    return ModalArchive(path)
