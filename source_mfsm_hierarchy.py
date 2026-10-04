"""2019 source hierarchy on a complete, explicitly reviewed coordinate space.

Part 1: GA 98–103, GB 105–113, GT 114–122, D 123–131, L 132–137,
SBt 162–167, TE 193–203 and Appendix A1. Part 2: modified GT 2–12,
Table 1 shear complement. Components exclude fasteners; the metric retains
their supported effects (SSRC 2023, 33–35). This is not an S4R extractor.

Numerical adaptations (reported in metadata): diagonalize the Part 2 Eq12
curvature ratio inside its degenerate candidate space; replace the union of
transverse-only shear subspaces in Eq200 by its aggregate strain-null space.
The latter avoids interpreting the SDw/SDt ambiguity in printed Eq188–189.
Neither adaptation is a claim of reproduced physical paper benchmarks.
"""
from dataclasses import asdict
import json
import os
import tempfile
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh, null_space, orth
from mfsm_model import BasisPack, content_hash, matrix, symmetric, normalized_columns

VERSION = 'khezri-2019-hierarchy-v2'
COMPONENTS = ('eps_x', 'eps_y', 'gamma_xy', 'kappa_x', 'kappa_y', 'kappa_xy')
METHOD = 'MFSM_KHEZRI_2019_OPERATOR_HIERARCHY'


def fsm_unprescribed_rows(node_count, main_nodes, harmonics=1):
    """Appendix A1 rows for harmonic-major, node-major U,V,W,theta ordering.

    Main/corner/end nodes are supplied by the reviewed cross-section topology.
    V is longitudinal warping and is never selected as a transverse DOF.
    """
    if (type(node_count) is not int or node_count < 1 or type(harmonics) is not int
            or harmonics < 1): raise ValueError('Positive integer node/harmonic counts required')
    main = np.asarray(main_nodes)
    if (main.ndim != 1 or (main.size and (main.dtype.kind not in 'iu' or
            np.any(main < 0) or np.any(main >= node_count))) or
            len(set(main.tolist())) != len(main)): raise ValueError('Invalid main nodes')
    main = set(main.tolist())
    return np.array([4*node_count*h+4*i+dof for h in range(harmonics)
                     for i in range(node_count) for dof in ((3,) if i in main else (0,2,3))], dtype=int)


def transverse_equilibrium_rows(kappa_x, unprescribed_rows, reduction=None):
    """Appendix A1: select canonical K^kappa_x rows, THEN right multiply Q.

    Caller identifies unprescribed theta at main nodes, U/W/theta at internal
    nodes in its documented canonical ordering; this does not infer corners.
    """
    k = symmetric(kappa_x)
    rows = np.asarray(unprescribed_rows)
    if rows.ndim != 1 or (rows.size and (rows.dtype.kind not in 'iu' or
            np.any(rows < 0) or np.any(rows >= len(k)))) or len(set(rows.tolist())) != len(rows):
        raise ValueError('Invalid or duplicate canonical equilibrium rows')
    c = k[rows.astype(int), :]
    return c if reduction is None else c@matrix(reduction, len(k))


def _psd(a, tolerance, name):
    a = symmetric(a)
    v = eigh(a, eigvals_only=True)
    if v[0] < -tolerance*max(float(np.max(np.abs(v))), 1e-250):
        raise ValueError(name+' must be positive semidefinite')
    return a


def positive_curvature_subspace(candidates, system, curvature, tolerance=1e-10):
    """Part 2 Eq12 positive spectral subspace, invariant to candidate rotation."""
    q = orth(normalized_columns(matrix(candidates, len(system))), rcond=tolerance)
    if not q.shape[1]: return q
    a = q.T@curvature@q; b = q.T@system@q
    vals, vectors = eigh((a+a.T)/2, (b+b.T)/2)
    if np.min(vals) < -tolerance*max(1., float(np.max(np.abs(vals)))):
        raise ValueError('Indefinite curvature ratio')
    return q@vectors[:, vals > tolerance*max(1., float(np.max(np.abs(vals))))]


def hierarchy_cache_key(operators, definition, tolerance=1e-10):
    array_keys = ('transverse_equilibrium', 'gamma_open', 'coordinate_metric',
                  'warping_projection', 'open_shear_basis')
    metadata = {k:v for k,v in definition.items() if k not in array_keys}
    present = [k for k in array_keys if k in definition]
    arrays = [operators.system]+[operators.components[k] for k in sorted(operators.components)]
    arrays += [definition[k] for k in present]
    return content_hash(dict(algorithm=VERSION, tolerance=tolerance, source=asdict(operators.source),
        operator_metadata=operators.metadata, component_names=sorted(operators.components),
        definition=metadata, definition_array_names=present), arrays)


def build_source_hierarchy(operators, definition, tolerance=1e-10):
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('Invalid hierarchy tolerance')
    if set(operators.components) != set(COMPONENTS):
        raise ValueError('Exactly six documented auxiliary strain components required')
    for name in ('coordinate_definition', 'equilibrium_definition'):
        if not isinstance(definition.get(name), str) or not definition[name].strip():
            raise ValueError('Documented '+name+' required')
    if 'transverse_equilibrium' not in definition:
        raise ValueError('Explicit canonical transverse equilibrium required')
    if type(definition.get('closed_cells')) is not bool:
        raise ValueError('Explicit closed_cells topology decision required')
    n = len(operators.system)
    c = matrix(definition['transverse_equilibrium'])
    if c.shape[1] != n: raise ValueError('Transverse equilibrium DOF mismatch')
    e = symmetric(definition.get('coordinate_metric', np.eye(n)))
    if e.shape != (n, n): raise ValueError('Coordinate metric DOF mismatch')
    cholesky(e, lower=True)  # Euclidean pullback must be positive definite.
    k = {name:_psd(operators.components[name], tolerance, name) for name in COMPONENTS}
    # One system whitening/factorization reused through the entire hierarchy.
    # Null energetic coordinates must already be explicitly reduced by caller.
    lt = cholesky(operators.system, lower=True).T
    w = solve_triangular(lt, np.eye(n))
    reduced = {name:w.T@a@w for name,a in k.items()}
    ratios = {}; searches = {}

    def join(*blocks):
        return np.column_stack(blocks) if blocks else np.empty((n, 0))

    def pure(name, components, z=None, custom=None):
        # Z columns encode source orthogonality / equilibrium covectors.
        h = np.eye(n) if z is None or not z.shape[1] else null_space(
            normalized_columns(w.T@z).T, rcond=tolerance)
        searches[name] = h.shape[1]
        if not h.shape[1]:
            ratios[name] = []; return np.empty((n, 0))
        km = sum(reduced[key] for key in components)
        if custom is not None: km = km+w.T@custom@w
        a = h.T@km@h
        vals, vectors = eigh((a+a.T)/2)
        scale = max(1., float(np.max(np.abs(vals))))
        if vals[0] < -tolerance*scale: raise ValueError('Indefinite '+name+' strain ratio')
        ratios[name] = vals.tolist()
        return w@h@vectors[:, np.abs(vals) <= tolerance*scale]

    ga = pure('GA', ('eps_x','gamma_xy','kappa_x','kappa_y','kappa_xy'))
    gb = pure('GB', ('eps_x','gamma_xy','kappa_x','kappa_xy'), k['eps_y']@ga)
    # Part 2 Table 1 retains Part 1 RD even when G uses modified torsion.
    classical_gt = pure('GT_classical', ('eps_x','gamma_xy','kappa_x'), k['eps_y']@join(ga,gb))
    sbt = pure('SBt', ('eps_x','eps_y','kappa_x','kappa_xy'))
    curvature_ratios = None
    if definition['closed_cells']:
        if not definition.get('closed_loop_review') or 'gamma_open' not in definition:
            raise ValueError('Verified closed-loop strip split and gamma_open required')
        gamma_open = _psd(definition['gamma_open'], tolerance, 'gamma_open')
        if gamma_open.shape != (n, n): raise ValueError('Closed-loop operator DOF mismatch')
        _psd(k['gamma_xy']-gamma_open, tolerance, 'closed-strip shear')
        initial = pure('GT_modified_candidates', ('eps_x','kappa_x'), e@join(ga,sbt), gamma_open)
        gt = positive_curvature_subspace(initial, operators.system,
                                        k['kappa_y']+k['kappa_xy'], tolerance)
        curvature_ratios = dict(candidates=initial.shape[1], retained=gt.shape[1])
    else:
        if 'gamma_open' in definition: raise ValueError('gamma_open requires closed_cells=True')
        gt = classical_gt
    g = join(gb,gt)
    d = pure('D', ('eps_x','gamma_xy'), join(k['eps_y']@join(ga,gb,classical_gt),c.T))
    local = pure('L', ('eps_x','eps_y','gamma_xy'))
    # All transverse-only shear: membrane-strain-null space excluding L.
    # Its aggregate replaces SBt+STt+SDt+SCt in Eq200, not a new DSM family.
    st = pure('St_aggregate', ('eps_x','eps_y'), e@local)
    tes = pure('TES', ('eps_y','kappa_x','kappa_xy','kappa_y'))
    tep = pure('TEP', ('eps_y',), e@join(local,st,tes))
    te = join(tes,tep)
    if definition['closed_cells']:
        # Part 2 Table 1 CLOSED-section row only.
        shear = w@null_space(normalized_columns(w.T@e@join(ga,g,d,local,te)).T, rcond=tolerance)
        shear_equations = 'Part2 Table1 CLOSED row'
    else:
        selection = definition.get('open_shear_selection')
        if selection == 'WARPING':
            if not definition.get('warping_projection_review') or 'warping_projection' not in definition:
                raise ValueError('Reviewed canonical warping projection required for open shear')
            p = matrix(definition['warping_projection'], n)
            if p.shape != (n,n): raise ValueError('Warping projection DOF mismatch')
            def zero_product(a,b):
                return np.linalg.norm(a@b) <= tolerance*max(np.linalg.norm(a)*np.linalg.norm(b),1e-250)
            if (np.linalg.norm(p@p-p) > tolerance*max(np.linalg.norm(p),1e-250) or
                    np.linalg.norm(p.T@e-e@p) > tolerance*max(np.linalg.norm(e)*np.linalg.norm(p),1e-250) or
                    any(not zero_product(k[key],p) for key in ('eps_x','kappa_x','kappa_y','kappa_xy')) or
                    not zero_product(k['eps_y'],np.eye(n)-p)):
                raise ValueError('Invalid canonical warping projection/strain split')
            # Sw is the complete aggregate of SBw/STw/SDw/SCw/SSw.
            sw = pure('Sw', ('eps_x','kappa_x','kappa_y','kappa_xy'), k['eps_y']@ga)
            ns = normalized_columns(sw)
            if np.linalg.norm(p@ns-ns) > tolerance*max(np.linalg.norm(ns),1e-250):
                raise ValueError('Sw is not contained in reviewed warping coordinates')
            stt = pure('STt', ('eps_x','eps_y','kappa_x'), k['gamma_xy']@sbt)
            sdw = p@d
            # Literal Eq188/189 uses SDw. Since gamma(D)=0, its shear
            # covectors equal minus those of the transverse part of D.
            sct = pure('SCt', ('eps_x','eps_y'), join(k['gamma_xy']@join(sbt,stt,sdw),e@local))
            shear = join(sw,sct)
            shear_equations = 'Part1 139-146,169-173,186-192; Part2 Table1 OPEN warping selection'
        elif selection == 'REVIEWED_AGGREGATE':
            if not definition.get('open_shear_equations') or 'open_shear_basis' not in definition:
                raise ValueError('Reviewed source open shear aggregate and equations required')
            shear = orth(normalized_columns(matrix(definition['open_shear_basis'],n)),rcond=tolerance)
            if np.linalg.norm(k['eps_x']@shear) > tolerance*max(np.linalg.norm(k['eps_x'])*np.linalg.norm(shear),1e-250):
                raise ValueError('Open shear aggregate violates zero eps_x criterion')
            shear_equations = definition['open_shear_equations']
        else:
            raise ValueError('Explicit source open shear selection WARPING or REVIEWED_AGGREGATE required')
    spaces = dict(GA=ga,G=g,D=d,L=local,TE=te,S=shear)
    key = hierarchy_cache_key(operators, definition, tolerance)
    meta = dict(operators.metadata, algorithm=VERSION, source=asdict(operators.source),
        source_method=METHOD, scientific_validation='PENDING_PHYSICAL_BENCHMARKS',
        scientifically_eligible=False, ratio_tolerance=tolerance, energy_ratios=ratios,
        search_dimensions=searches, closed_cells=definition['closed_cells'],
        coordinate_definition=definition['coordinate_definition'], equilibrium_definition=definition['equilibrium_definition'],
        modified_torsion_curvature=curvature_ratios,
        open_shear_selection=definition.get('open_shear_selection') if not definition['closed_cells'] else None,
        adaptations=['EQ12_CURVATURE_SPECTRAL_SPLIT','EQ200_AGGREGATE_TRANSVERSE_SHEAR',
                     'GA_INTERNAL_EXTENSION_EXCLUDED_FROM_GLOBAL_BUCKLING'],
        derivation_equations=dict(GA='Part1 98-103',GB='Part1 105-113',
            GT='Part2 2-12' if definition['closed_cells'] else 'Part1 114-122',
            D='Part1 123-131; Appendix A1', L='Part1 132-137',
            TE='Part1 193-203 (aggregate shear adaptation)',S=shear_equations),
        space_dimensions={name:space.shape[1] for name,space in spaces.items()})
    basis = BasisPack(operators.system,spaces,meta,key)
    subspaces = dict(FLEXURAL=gb,TORSIONAL=gt)
    subkey = content_hash(dict(algorithm=VERSION, main_basis_identity=key,
                               names=list(subspaces)),[operators.system,gb,gt])
    sub = BasisPack(operators.system,subspaces,
        dict(mechanical_definition_review=True, source_method=METHOD,
             scientifically_eligible=False, source=meta['derivation_equations']), subkey)
    return basis,sub


def cached_source_hierarchy(operators, definition, directory, tolerance=1e-10):
    key = hierarchy_cache_key(operators, definition, tolerance)
    os.makedirs(directory,exist_ok=True)
    path = os.path.join(directory,key+'.npz')
    if os.path.exists(path):
        try:
            with np.load(path,allow_pickle=False) as data:
                meta = json.loads(str(data['metadata'].item()))
                names = meta['names']; subnames = meta['subnames']
                metric = data['metric']; spaces = {k:data['space_'+k] for k in names}
                subs = {k:data['sub_'+k] for k in subnames}
                arrays = [metric]+[spaces[k] for k in names]+[subs[k] for k in subnames]
                if meta['key'] != key or content_hash(meta, arrays) != str(data['checksum'].item()):
                    raise ValueError('Hierarchy cache identity/checksum mismatch')
                return (BasisPack(metric,spaces,meta['basis_metadata'],key),
                        BasisPack(metric,subs,meta['sub_metadata'],meta['sub_identity']),
                        dict(hit=True,key=key,path=path))
        except (OSError,EOFError,KeyError,ValueError) as exc:
            raise ValueError('Invalid hierarchy cache: '+str(exc)) from exc
    basis,sub = build_source_hierarchy(operators,definition,tolerance)
    names = list(basis.spaces); subnames = list(sub.spaces)
    meta = dict(key=key,names=names,subnames=subnames,basis_metadata=basis.metadata,
                sub_metadata=sub.metadata,sub_identity=sub.definition_id)
    arrays = [basis.metric]+[basis.spaces[k] for k in names]+[sub.spaces[k] for k in subnames]
    payload = dict(metric=basis.metric,metadata=json.dumps(meta,sort_keys=True),checksum=content_hash(meta,arrays))
    payload.update({'space_'+k:v for k,v in basis.spaces.items()})
    payload.update({'sub_'+k:v for k,v in sub.spaces.items()})
    fd,temp = tempfile.mkstemp(prefix='hierarchy-',suffix='.npz',dir=directory)
    try:
        with os.fdopen(fd,'wb') as stream: np.savez(stream,**payload)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return basis,sub,dict(hit=False,key=key,path=path)
