# -*- coding: utf-8 -*-
"""Reclassify saved modal_shapes.npz; never open an ODB or run Abaqus.

python abaqus_reclassify_archive.py --run-dir RUN --output-dir NEW_DIRECTORY
Add --modes 2,124,197 for a limited review. Source results are never overwritten.
"""
import argparse
import copy
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import zipfile
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
import abaqus_modal_report as enhanced
import abaqus_modal_validation as validation
import abaqus_modal_visuals as visuals


class SparseProjector(enhanced.SectionProjector):
    """Same kinematics as the regular pipeline; sparse application only."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operators = {k: csr_matrix(op) for k, op in dict(
            G=self.pglobal, A=self.passembly, O=self.pother, D=self.pdist, L=self.plocal).items()}

    def _weighted_components(self, coefficients):
        y = np.asarray(coefficients, dtype=float)
        x = y.reshape(-1, len(self.sqrtw))/self.sqrtw
        return {k: ((op@x.T).T*self.sqrtw).reshape(y.shape) for k, op in self.operators.items()}


def validate_provenance(metadata, audit):
    for key in ('source_odb_sha256', 'model_signature'):
        if not metadata.get(key) or metadata[key] != audit.get(key):
            raise ValueError('Archive/audit provenance mismatch: '+key)


class ShapeArchive:
    """Read compressed arrays sequentially, one real mode at a time."""
    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data['metadata']))
            self.modes = data['modes'].copy()
            self.eigenvalues = data['eigenvalues'].copy()
            self.z = data['z'].copy()
            self.xy_parts = [data['p%d_xy' % i].copy() for i in range(len(self.metadata['names']))]
            self.references = [data['p%d_reference_xy' % i].copy() for i in range(len(self.xy_parts))]
        if len(self.z) < 2 or np.any(np.diff(self.z) <= 0) or not np.all(np.isfinite(self.z)):
            raise ValueError('Archive stations must increase strictly')
        if len(set(self.modes)) != len(self.modes) or len(self.eigenvalues) != len(self.modes):
            raise ValueError('Invalid archive mode/eigenvalue table')
        self.xy = np.vstack(self.xy_parts)
        self.edges = []; self.pieces = []; self.weights = []; offset = 0
        for name, xy in zip(self.metadata['names'], self.xy_parts):
            ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            if not len(ds) or np.any(ds <= 0) or not np.all(np.isfinite(xy)):
                raise ValueError('Invalid section polyline in archive')
            self.weights.extend(np.r_[ds[0]/2., (ds[:-1]+ds[1:])/2., ds[-1]/2.])
            self.edges.extend((offset+j, offset+j+1) for j in range(len(xy)-1))
            self.pieces.extend([name]*len(xy)); offset += len(xy)
        dz = np.diff(self.z)
        self.zweights = np.r_[dz[0]/2., (dz[:-1]+dz[1:])/2., dz[-1]/2.]/(self.z[-1]-self.z[0])
        self.streams = []; self.index = 0

    def __enter__(self):
        self.zip = zipfile.ZipFile(self.path)
        try:
            for i, xy in enumerate(self.xy_parts):
                stream = self.zip.open('p%d_u.npy' % i)
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    stream.close(); raise ValueError('Unsupported NPY header version')
                shape, fortran, dtype = reader(stream)
                if shape != (len(self.modes), len(self.z), len(xy), 3) or fortran or dtype.kind != 'f':
                    stream.close(); raise ValueError('Invalid archived displacement array')
                self.streams.append((stream, shape[1:], dtype))
        except Exception:
            self.__exit__(None, None, None); raise
        return self

    def read_next(self):
        if self.index >= len(self.modes):
            raise EOFError('No further archived modes')
        parts = []
        for stream, shape, dtype in self.streams:
            size = int(np.prod(shape))*dtype.itemsize
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError('Truncated displacement archive')
            values = np.frombuffer(raw, dtype=dtype).reshape(shape)
            if not np.all(np.isfinite(values)):
                raise ValueError('Nonfinite archived displacements')
            parts.append(values[:, :, self.metadata['transverse']])
        self.index += 1
        return np.concatenate(parts, axis=1)

    def __exit__(self, *unused):
        for stream, shape, dtype in self.streams:
            stream.close()
        self.zip.close()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)


def validate_source_geometry(archive, build):
    segments = build['source_inputs']['section_segments']
    for name, reference in zip(archive.metadata['names'], archive.references):
        source = np.asarray(segments[name], float)
        points = np.vstack((source[0, :2], source[:, 2:]))
        if points.shape != reference.shape or not (np.allclose(points, reference, rtol=0, atol=1e-6) or
                                                   np.allclose(points, reference[::-1], rtol=0, atol=1e-6)):
            raise ValueError('Build/archive source geometry mismatch for '+name)
    return segments


def synthetic_controls():
    """Fixed analytical cases, independent of the desired labels of real modes."""
    x = np.linspace(0., 180., 81)
    xy = np.column_stack((x, 6.*np.sin(2.*np.pi*x/180.)))
    segments = np.column_stack((xy[:-1], xy[1:])).tolist()
    fit = enhanced.SectionProjector(xy, [(i, i+1) for i in range(80)], ['P']*81,
        np.ones(81), physical_segments={'P': segments})
    tangent = np.gradient(xy, axis=0); tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    rigid = np.array([2., -3.])+.17*np.column_stack((-xy[:, 1], xy[:, 0]))
    local = np.sin(np.linspace(0., np.pi, len(xy)))[:, None]*normal
    rigid_error = float(np.max(np.abs(fit.plocal@rigid.ravel())))
    invariance_error = float(np.max(np.abs(fit.plocal@(local+rigid).ravel()-fit.plocal@local.ravel())))
    local_percent = fit.component_diagnostics(local)['local_percent']
    from abaqus_physical_walls import wall_kinematics
    left, right, unused, unused_dist = wall_kinematics(xy)
    coarse = np.einsum('nij,j->ni', left, [1., -2.])+np.einsum('nij,j->ni', right, [1., 3.])
    coarse_error = float(np.max(np.abs(fit.plocal@coarse.ravel())))
    # Independent analytical channel: symmetric flange rotations about fixed
    # web junctions, with zero within-wall bending and zero chord extension.
    left_wall = np.column_stack((np.zeros(9), np.linspace(40., 0., 9)))
    web = np.column_stack((np.linspace(0., 100., 21)[1:], np.zeros(20)))
    right_wall = np.column_stack((np.full(8, 100.), np.linspace(0., 40., 9)[1:]))
    channel = np.vstack((left_wall, web, right_wall))
    opening = np.zeros_like(channel)
    opening[:9, 0] = -left_wall[:, 1]/40.
    opening[-8:, 0] = right_wall[:, 1]/40.
    channel_fit = enhanced.SectionProjector(channel,
        [(i, i+1) for i in range(len(channel)-1)], ['P']*len(channel), np.ones(len(channel)),
        physical_segments={'P': np.column_stack((channel[:-1], channel[1:])).tolist()})
    channel_diagnostics = channel_fit.component_diagnostics(opening)
    result = dict(smooth_wall_count=fit.metadata['physical_wall_count'],
        rigid_local_max_abs=rigid_error, local_under_rigid_motion_max_abs=invariance_error,
        fixed_end_bending_L_percent=local_percent, coarse_motion_local_max_abs=coarse_error,
        channel_flange_rotation_D_percent=channel_diagnostics['distortional_percent'],
        channel_flange_rotation_L_percent=channel_diagnostics['local_percent'],
        scope='Analytical kinematic controls, not mechanical family certification')
    result['passed'] = bool(result['smooth_wall_count'] == 1 and
        max(rigid_error, invariance_error, coarse_error) < 1e-10 and local_percent > 99.999999 and
        channel_diagnostics['distortional_percent'] > 99.999999 and channel_diagnostics['local_percent'] < 1e-15)
    if not result['passed']:
        raise ValueError('Synthetic controls failed: '+str(result))
    return result


def mode_preview(values, archive):
    norms = np.linalg.norm(values, axis=2)
    peak, track = np.unravel_index(int(np.argmax(norms)), norms.shape)
    indices = np.unique(np.r_[np.linspace(0, len(archive.z)-1, 25).astype(int), peak])
    return dict(z_mm=archive.z[indices].tolist(), peak_z_mm=float(archive.z[peak]),
        peak_index=int(np.where(indices == peak)[0][0]), sections=values[indices],
        track_z_mm=archive.z.tolist(), track_u=values[:, track, :].tolist(),
        convention='Real saved stations only; all stations/nodes used numerically')


def classified_row(old, vector, projectors, settings):
    row = copy.deepcopy(old)
    fit = projectors[1]
    percentages = [p.audit_shares(vector) for p in projectors]
    sensitivity = visuals.sensitivity(percentages, settings['dominance'])
    diagnostics = visuals.rigid_shares(vector, fit)
    shares = percentages[1]
    label = 'LDG'[int(np.argmax(shares))] if max(shares) >= 100*settings['dominance'] else 'Mixed'
    flags = [flag for flag in old['flags'] if flag in (
        'POOR_SINE_FIT_WAVELENGTH_UNCERTAIN', 'WAVELENGTH_NEAR_MESH_RESOLUTION_LIMIT',
        'INCOMPLETE_TRACK_COVERAGE', 'TRANSVERSE_PROJECTION_HAS_LOW_PARTICIPATION')]
    flags.append('GEOMETRIC_PROXY_NOT_MECHANICAL_IDENTIFICATION')
    if sensitivity['max_range_pp'] > settings['max_sensitivity_pp'] or len(set(sensitivity['labels'])) > 1:
        label = 'Unresolved'; flags.append('PANEL_DEFINITION_SENSITIVE')
    if diagnostics['assembly_percent'] >= settings['max_assembly_percent']:
        label = 'Assembly'; flags.append('ASSEMBLY_RIGID_MOTION_DOMINANT')
    elif diagnostics['other_percent'] >= settings['max_other_percent']:
        label = 'Other'; flags.append('OTHER_TRANSVERSE_EXTENSION_DOMINANT')
    if not fit.supported or old.get('transverse_share', 1.) < .05 or 'INCOMPLETE_TRACK_COVERAGE' in flags:
        label = 'Unresolved'
    row.update(family=label, raw_family=label, percentages=shares, sensitivity=sensitivity,
        rigid_diagnostics=diagnostics, flags=flags, mechanical_eligible=False,
        percentage_kind='GEOMETRIC_PROXY_DIRECT_NODAL_NORM_CURVED_WALL_V2',
        fitted_percentages=None, raw_vs_fitted_max_pp=None,
        energy=None, energy_status='NOT_RECOMPUTED_GEOMETRIC_ARCHIVE_ONLY',
        condition=None, component_norm_sum_over_input=diagnostics['component_norm_sum_over_input'],
        displacement_cross_percent=100.*(1.-diagnostics['component_norm_sum_over_input']),
        relative_residual=diagnostics['reconstruction_relative_error'])
    return row


def wall_plot(output, fit, old_geometry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    xy = fit.xy
    for name, layout in fit.metadata['source_wall_layouts'].items():
        ids = np.where(fit.pieces == name)[0]
        for ax in axes:
            if ax is axes[1] and name != 'P1':
                continue
            ax.plot(xy[ids, 0], xy[ids, 1], color='#adb5bd', lw=2)
            for j, wall in enumerate([w for w in fit.metadata['physical_walls'] if w['piece'] == name]):
                a, b = wall['start'], wall['end']
                path = np.arange(min(a, b), max(a, b)+1)
                ax.plot(xy[path, 0], xy[path, 1], lw=3, color=plt.get_cmap('tab10')(j))
                middle = xy[path[len(path)//2]]
                if ax is axes[1]:
                    ax.annotate(name+'/W%d' % (j+1), middle, xytext=(8, 8), textcoords='offset points', fontsize=9)
                ax.scatter(xy[[a, b], 0], xy[[a, b], 1], s=23, c='black', zorder=5)
            for wall in layout['walls']:
                for point in (wall['start_xy'], wall['end_xy']):
                    ax.scatter(*point, s=45, facecolors='none', edgecolors='#d1495b', zorder=6)
    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.grid(alpha=.15)
    axes[0].set_title('Complete section: detected physical walls')
    axes[1].set_title('P1: curved lips + broad wavy walls\nBlack: mapped nodes | red rings: source endpoints')
    fig.suptitle('Geometry-only wall review | old %d walls / %.1f%% coverage; new %d / %.1f%%\n'
        'Gray gaps: compact bends; maximum endpoint mapping distance %.3f mm' % (
        old_geometry['physical_wall_count'], 100*old_geometry['physical_wall_coverage_fraction'],
        fit.metadata['physical_wall_count'], 100*fit.metadata['physical_wall_coverage_fraction'],
        fit.metadata['maximum_wall_mapping_error_mm']))
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(output/'wall_layout.png', dpi=180); fig.savefig(output/'wall_layout.pdf'); plt.close(fig)


def selected_plot(output, selected, geometry):
    if not selected:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 4.6*len(selected)), squeeze=False)
    xy = np.asarray(geometry['xy']); edges = geometry['edges']; width = max(np.ptp(xy, axis=0))
    for axs, item in zip(axes, selected):
        factor = .13*width/max(np.max(np.linalg.norm(item['U'], axis=1)), 1e-30)
        for ax, key in zip(axs, ('U', 'L', 'D')):
            moved = xy+factor*item[key]
            for a, b in edges:
                ax.plot(xy[[a,b],0], xy[[a,b],1], color='#bbc2cb', lw=.8)
                ax.plot(moved[[a,b],0], moved[[a,b],1], color={'U':'#3157a4','L':'#21936e','D':'#cb7c21'}[key], lw=1.3)
            ax.set_aspect('equal'); ax.set_title('Mode %d | %s | z=%.1f mm' % (item['mode'], key, item['z']))
        axs[0].set_ylabel('Old L/D %.2f / %.2f%%\nNew L/D %.2f / %.2f%%' %
            (*item['old'][:2], *item['new'][:2]))
    fig.suptitle('Actual section and reconstructed components | same amplification within each row\n'
                 'Percentages integrate all saved stations; component plots show one peak station only')
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(output/'selected_mode_components.png', dpi=170)
    fig.savefig(output/'selected_mode_components.pdf'); plt.close(fig)


def write_comparison(output, summary, old_rows):
    rows = []
    for row in summary['modes']:
        old = old_rows[row['mode']]
        item = dict(mode=row['mode'], stress_MPa=row['stress_MPa'], half_wavelength_mm=row['half_wavelength_mm'],
            old_family=old['family'], raw_family=row['raw_family'], family=row['family'],
            eigenspace_stable_family=row['eigenspace_stable_family'])
        for i, family in enumerate('LDG'):
            item['old_'+family+'_percent'] = old['percentages'][i]
            item[family+'_percent'] = row['percentages'][i]
        for key in ('assembly_percent', 'other_percent', 'wall_curvature_index', 'reconstruction_relative_error'):
            item[key] = row['rigid_diagnostics'][key]
        item['flags'] = ';'.join(row['flags']); rows.append(item)
        item['component_norm_sum_over_input'] = row['component_norm_sum_over_input']
        item['displacement_cross_percent'] = row['displacement_cross_percent']
    with open(output/'modal_percentages.csv', 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table = ''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]) if not isinstance(row[k], float) else '%.3f'%row[k])+'</td>'
        for k in ('mode','old_family','family','old_L_percent','L_percent','old_D_percent','D_percent','half_wavelength_mm'))+'</tr>' for row in rows)
    text = '''<!doctype html><meta charset="utf-8"><title>Curved-wall reclassification</title>
<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:35px auto;padding:24px;background:#f7f9fc;color:#172a41}
img{width:100%;background:white}td,th{padding:7px 12px;border-bottom:1px solid #ddd;text-align:right}table{border-collapse:collapse;width:100%}a{color:#145da0}</style>
<h1>Curved-wall reclassification from saved shapes</h1><p>No Abaqus model or solver was run. All eigenvalues and wavelengths are unchanged.
These are geometric displacement screening shares, not a validated mechanical L/D/G basis.</p>
<p>Source and mapped wall boundaries are shown below. The wall detector uses geometry only. Compact folds are separate;
gentle waviness and curved lips remain complete walls. Red rings and black dots expose endpoint mapping error.</p>
<img src="wall_layout.png"><p><a href="modal_explorer.html">Interactive shapes</a> · <a href="eigenspace_validation.html">Eigenspace bounds</a> ·
<a href="modal_percentages.csv">Before/after CSV</a> · <a href="synthetic_controls.json">Analytical controls</a> · <a href="provenance.json">Provenance</a></p>
<img src="selected_mode_components.png"><p>Local and Distortional percentages use their L/D/G denominator; Assembly/Other use all five components.
Final labels also apply geometry sensitivity, Assembly/Other and eigenspace gates. They may differ from individual-mode percentages.</p>
<p>The components are not orthogonal. Their squared norms do not necessarily sum to the input squared norm;
the signed cross contribution is retained in CSV/JSON. Analytical kinematic tests pass, but no independent
mechanical reference or mesh convergence study has certified the L/D/G labels. Geometry thresholds are assumptions,
not fitted to the desired percentages.</p>
<table><thead><tr><th>Mode</th><th>Old</th><th>New final</th><th>Old L%</th><th>New L%</th><th>Old D%</th><th>New D%</th><th>Half-wave mm</th></tr></thead><tbody>__TABLE__</tbody></table>'''
    (output/'comparison.html').write_text(text.replace('__TABLE__', table), encoding='utf-8')


def process(args):
    run = Path(args.run_dir).resolve(); output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError('Output must be a new directory; original reports are preserved')
    audit_path = run/'modal_dsm_audit/modal_audit.json'
    archive_path = run/'modal_dsm_audit/modal_shapes.npz'
    build_paths = list(run.glob('*_build.json'))
    if len(build_paths) != 1:
        raise ValueError('Expected exactly one build report')
    old = json.loads(audit_path.read_text()); build = json.loads(build_paths[0].read_text())
    old_rows = {r['mode']:r for r in old['modes']}
    selected_modes = set(map(int, args.modes.split(','))) if args.modes else set(old_rows)
    if not selected_modes or not selected_modes <= set(old_rows):
        raise ValueError('Requested modes must exist in the saved audit')
    controls = synthetic_controls()
    with ShapeArchive(archive_path) as archive:
        validate_provenance(archive.metadata, old)
        segments = validate_source_geometry(archive, build)
        if list(archive.modes) != [r['mode'] for r in old['modes']] or not np.allclose(
                archive.eigenvalues, [r['eigenvalue'] for r in old['modes']], rtol=1e-10, atol=1e-10):
            raise ValueError('Archive/audit mode table mismatch')
        projectors = [SparseProjector(archive.xy, archive.edges, archive.pieces, archive.weights,
            physical_segments=segments, bend_radius_fraction=fraction) for fraction in (.03,.04,.05)]
        fit = projectors[1]
        output.mkdir(parents=True)
        inputs = [archive_path, audit_path, build_paths[0]]
        source_hashes = {str(path):sha256(path) for path in inputs}
        save_json(output/'synthetic_controls.json', controls)
        save_json(output/'wall_layout.json', fit.metadata)
        wall_plot(output, fit, old['proxy_geometry'])
        settings = dict(old['settings'], run_dir=str(run), output_dir=str(output), archive_only=True)
        summary = dict(schema_version=2, modes=[], clusters=[], settings=settings, modes_processed=0,
            source_odb=old['source_odb'], source_odb_sha256=old['source_odb_sha256'],
            model_signature=old['model_signature'], sigma_ref_MPa=old.get('sigma_ref_MPa'),
            basis_metadata={}, proxy_geometry=fit.metadata,
            percentage_kind='GEOMETRIC_PROXY_DIRECT_NODAL_NORM_CURVED_WALL_V2',
            sensitivity_radius_fractions=[.03,.04,.05], curve_reference=old.get('curve_reference',[]),
            dsm_inputs_MPa={f:None for f in ('Fcrl', 'Fcrd', 'Fcre')},
            limitations=['Geometric screening only; no mapped mechanical basis or new DSM acceptance.',
                        'Source endpoints map to existing mesh nodes; mapping distances are reported.',
                        'Eigenvalues and wavelengths copied unchanged from the original audit.'])
        geometry = dict(xy=archive.xy.tolist(), edges=archive.edges, pieces=archive.pieces)
        previews = []; selected = []
        preview_modes = set(map(int, args.preview_modes.split(',')))
        for old_cluster in old['clusters']:
            values = []; members = []
            for mode in old_cluster['modes']:
                if archive.index >= len(archive.modes) or int(archive.modes[archive.index]) != mode:
                    raise ValueError('Audit clusters must cover archive modes once, in order')
                u = archive.read_next()
                if mode in selected_modes:
                    values.append(u); members.append(mode)
            if not members:
                continue
            vectors = np.column_stack([visuals.weighted_raw(u, archive.zweights, fit.sqrtw) for u in values])
            new_rows = [classified_row(old_rows[mode], vectors[:,i], projectors, settings) for i,mode in enumerate(members)]
            cluster = copy.deepcopy(old_cluster)
            complete_cluster = len(members) == len(old_cluster['modes'])
            bounds = None; error = None; sensitive = False; label = 'Unresolved'
            if complete_cluster:
                try:
                    all_bounds = [validation.proxy_cluster(vectors, p, settings['dominance'],
                        settings['max_assembly_percent'], settings['max_other_percent']) for p in projectors]
                    bounds = all_bounds[1]
                    sensitive = (len({b['stable_family'] for b in all_bounds}) > 1 or any(
                        np.max(np.ptp([b[key] for b in all_bounds], axis=0)) > settings['max_sensitivity_pp']
                        for key in ('min_percent','max_percent')))
                    label = bounds['stable_family']
                    if sensitive or not cluster['spectral_isolation']['isolated']:
                        label = 'Unresolved'
                except ValueError as exc:
                    error = str(exc)
            else:
                error = 'Partial selected cluster: eigenspace bounds not reassessed'
            cluster.update(modes=members, bounds=bounds, family=label, angle_sensitive=sensitive,
                           bound_error=error, energy=None, percentages=None)
            summary['clusters'].append(cluster)
            for row, u, vector in zip(new_rows, values, vectors.T):
                row.update(eigenspace_stable_family=label, eigenspace_bounds=bounds)
                if label != row['raw_family'] or label == 'Unresolved':
                    row['family'] = 'Unresolved'
                if not complete_cluster:
                    row['flags'].append('PARTIAL_EIGENSPACE_NOT_REASSESSED')
                if len(old_cluster['modes']) > 1:
                    row['flags'].append('NEAR_REPEATED_EIGENSPACE_SEE_CLUSTER_PERCENTAGES')
                if not cluster['spectral_isolation']['isolated']:
                    row['flags'].append('SPECTRAL_ISOLATION_NOT_CONFIRMED')
                preview = mode_preview(u, archive); previews.append(preview)
                if row['mode'] in preview_modes:
                    peak = int(np.argmin(abs(archive.z-preview['peak_z_mm'])))
                    x = u.reshape(len(archive.z), -1)
                    selected.append(dict(mode=row['mode'], z=float(archive.z[peak]), U=u[peak],
                        L=(fit.operators['L']@x[peak]).reshape(-1,2),
                        D=(fit.operators['D']@x[peak]).reshape(-1,2),
                        old=old_rows[row['mode']]['percentages'], new=row['percentages']))
                summary['modes'].append(row)
            print('Reclassified %d/%d saved modes' % (len(summary['modes']),len(selected_modes)), flush=True)
        summary['modes_processed'] = len(summary['modes'])
        summary['family_counts'] = dict(Counter(r['family'] for r in summary['modes']))
        summary['raw_family_counts'] = dict(Counter(r['raw_family'] for r in summary['modes']))
        summary['comparison'] = dict(old_wall_count=old['proxy_geometry']['physical_wall_count'],
            old_coverage=old['proxy_geometry']['physical_wall_coverage_fraction'],
            old_max_L=max(r['percentages'][0] for r in old['modes']),
            new_max_L=max(r['percentages'][0] for r in summary['modes']),
            maximum_reconstruction_error=max(r['relative_residual'] for r in summary['modes']))
        if len(summary['modes']) != len(selected_modes):
            raise ValueError('Some selected modes were not processed')
        # Read-only source checks; no solver/database is invoked anywhere.
        if any(sha256(path) != source_hashes[str(path)] for path in inputs):
            raise RuntimeError('A source file changed during reclassification')
        save_json(output/'modal_audit.json', summary)
        save_json(output/'provenance.json', dict(inputs_sha256=source_hashes, source_files_unchanged=True,
            solver_run=False, odb_opened=False, controls_passed=controls['passed'],
            code_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                ('abaqus_reclassify_archive.py','abaqus_physical_walls.py','abaqus_modal_report.py',
                 'abaqus_modal_validation.py','abaqus_modal_visuals.py','abaqus_dsm_modal_audit.py')}))
        write_comparison(output, summary, old_rows)
        selected_plot(output, selected, geometry)
        base_report = next(run.glob('*_modal_wavelengths_report.json'))
        unused_meta, base_rows, spectra, unused_prefix = enhanced.load_results(str(base_report))
        indices = {r['mode']:i for i,r in enumerate(base_rows)}
        spectra = spectra[[indices[r['mode']] for r in summary['modes']]]
        visuals.write_visuals(str(output), summary, previews, geometry, spectra)
        validation.write_cluster_report(str(output), summary)
        (output/'SUMMARY.txt').write_text(json.dumps(dict(
            modes=summary['modes_processed'], family_counts=summary['family_counts'],
            raw_family_counts=summary['raw_family_counts'], comparison=summary['comparison'],
            solver_run=False, source_files_unchanged=True, synthetic_controls=controls), indent=2), encoding='utf-8')
        print('COMPLETE: '+str(output), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--modes', help='Comma-separated saved mode IDs; default: all')
    parser.add_argument('--preview-modes', default='2,124,197')
    return process(parser.parse_args(argv))


if __name__ == '__main__':
    main()
