"""Canonical physical-wall topology; nodes are never merged between pieces."""
from dataclasses import dataclass
import numpy as np
from mfsm_model import content_hash


@dataclass
class ReferenceSection:
    nodes: np.ndarray
    edges: tuple
    pieces: tuple
    definition_id: str


def build_reference(source_inputs, discretization=1.):
    if not np.isfinite(discretization) or discretization<=0:
        raise ValueError('Positive section discretization required')
    nodes=[]; edges=[]; pieces=[]
    for piece,segments in sorted(source_inputs.items()):
        lookup={}
        for segment in segments:
            a,b=np.asarray(segment,dtype=float).reshape(2,2)
            length=float(np.linalg.norm(b-a))
            if not np.isfinite(length) or length<=0:
                raise ValueError('Nonzero finite physical wall required')
            points=np.linspace(a,b,int(np.ceil(length/discretization))+1)
            ids=[]
            for point in points:
                key=tuple(point)
                if key not in lookup:
                    lookup[key]=len(nodes); nodes.append(point); pieces.append(piece)
                ids.append(lookup[key])
            edges.extend(zip(ids[:-1],ids[1:]))
    if not nodes: raise ValueError('Empty physical section')
    nodes=np.asarray(nodes)
    return ReferenceSection(nodes,tuple(edges),tuple(pieces),content_hash(
        {'pieces':pieces,'edges':edges,'discretization':discretization},[nodes]))
