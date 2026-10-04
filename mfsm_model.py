"""Auditable contracts for auxiliary-energy modal decomposition (FP64, mm/N)."""
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np

ALGORITHM_VERSION = 'mfsm-energy-operators-v1'


def matrix(value, rows=None):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)) or (rows is not None and a.shape[0] != rows):
        raise ValueError('Matrix shape/finite-data mismatch')
    return a


def symmetric(value):
    a = matrix(value)
    if a.shape[0] != a.shape[1] or not np.allclose(a, a.T, rtol=1e-10, atol=max(np.max(np.abs(a)), 1e-250)*1e-12):
        raise ValueError('Operator must be square and symmetric')
    return a


def content_hash(metadata, arrays=()):
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class SourceDefinition:
    reference: str
    equations: str
    metric_definition: str

    def __post_init__(self):
        if not all(isinstance(x, str) and x.strip() for x in (self.reference, self.equations, self.metric_definition)):
            raise ValueError('Source reference, equations and metric are required')


@dataclass
class OperatorPack:
    system: np.ndarray
    components: dict
    source: SourceDefinition
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.system = symmetric(self.system)
        n = len(self.system)
        if not n or not self.components or not isinstance(self.source, SourceDefinition):
            raise ValueError('Nonempty operators and source definition required')
        self.components = {k: symmetric(v) for k, v in self.components.items()}
        if any(v.shape != (n,n) for v in self.components.values()):
            raise ValueError('Component/system DOF mismatch')


@dataclass
class BasisPack:
    metric: np.ndarray
    spaces: dict
    metadata: dict
    definition_id: str

    def __post_init__(self):
        self.metric = symmetric(self.metric)
        self.spaces = {k: matrix(v, len(self.metric)) for k,v in self.spaces.items()}
        if not self.definition_id or not self.spaces:
            raise ValueError('Basis identity and spaces required')


@dataclass
class ValidationEvidence:
    model_hash: str
    algorithm: str
    checks: dict
    REQUIRED = ('source_derivation', 'published_benchmark', 'connection_limit',
                'active_contact', 'abaqus_mesh', 'harmonic_convergence', 'eigenspace_match')

    def eligible(self, model_hash, algorithm):
        return (bool(model_hash) and self.model_hash == model_hash and self.algorithm == algorithm
                and all(self.checks.get(k) is True for k in self.REQUIRED))


@dataclass
class ModeBatch:
    vectors: np.ndarray
    ids: tuple

    def __post_init__(self):
        self.vectors = matrix(self.vectors)
        if self.vectors.shape[1] != len(self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError('Unique mode IDs must match vector columns')


def normalized_columns(value):
    """Remove arbitrary nonzero column amplitude before numerical rank tests."""
    a=matrix(value)
    scales=np.max(np.abs(a),axis=0) if a.shape[0] else np.zeros(a.shape[1])
    nonzero=scales>0
    a=a[:,nonzero]/scales[nonzero]
    return a/np.linalg.norm(a,axis=0) if a.shape[1] else a


def check_constraints_per_mode(vectors,constraints,tolerance=1e-8):
    x=matrix(vectors);c=matrix(constraints)
    if c.shape[1]!=x.shape[0]: raise ValueError('Constraint DOF mismatch')
    # Normalize each mode first: another mode's amplitude cannot hide a failure.
    scales=np.max(np.abs(x),axis=0)
    normalized=x/np.maximum(scales,1e-250)
    residual=np.linalg.norm(c@normalized,axis=0)/np.maximum(
        np.linalg.norm(c)*np.linalg.norm(normalized,axis=0),1e-250)
    if np.any(residual>tolerance): raise ValueError('Modes violate documented constraints')
    return residual
