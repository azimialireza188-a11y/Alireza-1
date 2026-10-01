# -*- coding: utf-8 -*-
"""Read-only physical shell-energy diagnostics from detailed Buckle ODB fields.

Requires the current workflow: planar S4R elements, homogeneous isotropic linear
elastic material, centered constant-thickness section, five Simpson points.
Reconstructs membrane/bending energy from S:E and transverse shear from SF:SE.
Does NOT recover assembled K, numerical stabilization energy, or L/D/G energy.
"""
import argparse
import base64
import csv
import hashlib
import html as html_module
import json
import os
import re
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(globals().get('__file__', sys._getframe().f_code.co_filename)))
if HERE not in sys.path: sys.path.insert(0, HERE)
W = np.array([1., 4., 2., 4., 1.])/12.
REQUIRED = {'S', 'E', 'SF', 'SE', 'SM', 'SK', 'U'}


def components(data, labels, wanted):
    if not set(wanted).issubset(labels):
        raise ValueError('Missing components: '+str(set(wanted)-set(labels)))
    return np.asarray(data)[..., [list(labels).index(k) for k in wanted]]


def plane_stress_matrix(young, nu):
    return young/(1-nu*nu)*np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])


def inplane_partition(stress, strain, thickness):
    # Abaqus E12 is engineering shear gamma12: DO NOT multiply S12*E12 by 2.
    s, e = np.asarray(stress), np.asarray(strain)
    if s.shape != e.shape or s.ndim != 3 or s.shape[0] != 5 or s.shape[2] != 3:
        raise ValueError('Expected matching (5, elements, 3) S11/S22/S12 and E11/E22/E12')
    sm = np.einsum('p,pnc->nc', W, s); em = np.einsum('p,pnc->nc', W, e)
    membrane = .5*thickness*np.sum(sm*em, axis=1)
    bending = .5*thickness*np.einsum('p,pnc,pnc->n', W, s-sm, e-em)
    total = .5*thickness*np.einsum('p,pnc,pnc->n', W, s, e)
    scale = max(float(np.max(np.abs(total))), 1e-250)
    if not np.all(np.isfinite(total)) or min(np.min(membrane), np.min(bending)) < -scale*1e-8:
        raise ValueError('Negative/nonfinite elastic work: incompatible fields, material, or coordinate basis')
    if not np.allclose(total, membrane+bending, rtol=1e-9, atol=scale*1e-12):
        raise ValueError('Through-thickness partition fails closure')
    return dict(membrane=membrane, bending=bending, total=total, mean_stress=sm, mean_strain=em)


def concentration(area, element_energy):
    area, v = np.asarray(area), np.asarray(element_energy)
    total = float(v.sum()); density = v/area
    if total <= 0 or np.any(area <= 0): raise ValueError('Nonpositive energy or area')
    effective = total**2/(area.sum()*np.sum(area*density**2))
    order = np.argsort(density)[::-1]
    taken = np.clip(.1*area.sum()-np.r_[0., np.cumsum(area[order])[:-1]], 0., area[order])
    return dict(effective_area_percent=float(100*effective),
                top_10pct_area_energy_percent=float(100*np.sum(taken*density[order])/total))


def read_json(path):
    with open(path, encoding='utf-8-sig') as f: return json.load(f)


def serialize_result(result):
    def native(value):
        if isinstance(value, np.generic): return value.item()
        raise TypeError('Unsupported report value: '+type(value).__name__)
    return json.dumps(result, indent=2, allow_nan=False, default=native)


def link_audit_report(audit_dir, output_dir):
    """Add navigation only; never rewrite the previous numerical family audit."""
    path = os.path.join(audit_dir, 'modal_explorer.html')
    if not os.path.isfile(path): return
    with open(path, encoding='utf-8') as f: content = f.read()
    relative = os.path.relpath(os.path.join(output_dir, 'shell_energy_report.html'), audit_dir).replace('\\', '/')
    note = ('<p id="shell-energy-link"><a href="'+html_module.escape(relative, quote=True)+'">'
        'NEW: detailed membrane / bending / transverse-shear energy report</a>'
        ' — these mechanisms are separate from L/D/G families.</p>')
    if 'id="shell-energy-link"' in content:
        content = re.sub(r'<p id="shell-energy-link">.*?</p>', note, content)
    else:
        content = content.replace('</main>', note+'</main>', 1)
    content = content.replace('Energy: ${r.energy_status}', 'L/D/G energy: ${r.energy_status}')
    with open(path, 'w', encoding='utf-8') as f: f.write(content)


def one_file(folder, suffix):
    found = [os.path.join(folder, n) for n in os.listdir(folder) if n.endswith(suffix)]
    if len(found) != 1: raise ValueError('Expected one '+suffix+' in '+folder)
    return found[0]


def validate_input(path, source):
    with open(path) as f: text = f.read().upper()
    sections = re.findall(r'^\*SHELL SECTION([^\n]*)\n([^\n]+)', text, re.M)
    if not sections: raise ValueError('No homogeneous shell section found')
    t = float(source['thickness_mm']); young = float(source['E_MPa']); nu = float(source['nu'])
    for options, line in sections:
        values = [float(x.strip()) for x in line.split(',') if x.strip()]
        if ('COMPOSITE' in options or 'GAUSS' in options or 'NODAL' in options or
                'OFFSET' in options or len(values) != 2 or values[1] != 5 or not np.isclose(values[0], t)):
            raise ValueError('Only centered homogeneous constant-thickness 5-point Simpson sections supported')
    elastic = re.findall(r'^\*ELASTIC[^\n]*\n([^\n]+)', text, re.M)
    if len(elastic) != 1 or '*PLASTIC' in text or '*HYPERELASTIC' in text:
        raise ValueError('Expected one isotropic linear elastic material')
    values = [float(x.strip()) for x in elastic[0].split(',') if x.strip()]
    if len(values) != 2 or not np.allclose(values, [young, nu]):
        raise ValueError('Input material differs from build metadata')
    if t <= 0 or young <= 0 or not -1 < nu < .5: raise ValueError('Invalid elastic material')
    return t, young, nu


def geometry(odb, axis):
    keys, areas, centers, tables = [], [], [], {}
    for name in sorted(odb.rootAssembly.instances.keys()):
        inst = odb.rootAssembly.instances[name]
        if not len(inst.elements): continue
        coords = {n.label: np.asarray(n.coordinates, dtype=float) for n in inst.nodes}
        tables[name] = {}
        for element in inst.elements:
            if element.type != 'S4R': raise ValueError('Unsupported element: '+element.type)
            points = np.asarray([coords[n] for n in element.connectivity])
            normal = np.cross(points[1]-points[0], points[3]-points[0])
            size = max(float(np.max(np.linalg.norm(points-points[0], axis=1))), 1e-30)
            if abs(np.dot(points[2]-points[0], normal))/max(np.linalg.norm(normal), 1e-30) > 1e-6*size:
                raise ValueError('Warped shell: exact integration weight not available from this ODB')
            area = .5*np.linalg.norm(np.cross(points[2]-points[0], points[3]-points[1]))
            if area <= 0: raise ValueError('Degenerate shell')
            tables[name][element.label] = len(keys)
            keys.append((name, element.label)); areas.append(area); centers.append(points.mean(axis=0))
    if not keys: raise ValueError('No shell elements')
    return dict(keys=keys, area=np.asarray(areas), centers=np.asarray(centers), tables=tables, axis=axis)


def read_field(field, geo):
    """Map by instance + label + section point; reject duplicate/missing IPs."""
    count = len(geo['keys']); result = {}; bases = {}
    for block in field.bulkDataBlocks:
        name = block.instance.name if block.instance else None
        if name not in geo['tables']: continue
        if str(block.position) != 'INTEGRATION_POINT': raise ValueError('Expected integration-point fields')
        labels = np.asarray(block.elementLabels, dtype=int)
        ips = np.asarray(block.integrationPoints, dtype=int)
        if len(ips) != len(labels) or np.any(ips != 1) or len(np.unique(labels)) != len(labels):
            raise ValueError('Expected one in-plane integration point per S4R element')
        index = np.array([geo['tables'][name][int(n)] for n in labels])
        sp = block.sectionPoint.number if block.sectionPoint is not None else 0
        array = result.setdefault(sp, np.full((count, len(field.componentLabels)), np.nan))
        if np.any(np.isfinite(array[index])): raise ValueError('Duplicate section/IP samples')
        values = None
        for attr in ('dataDouble', 'data'):
            try: values = np.asarray(getattr(block, attr), dtype=float); break
            except Exception: continue
        if values is None or values.shape != (len(index), len(field.componentLabels)):
            raise ValueError('Unavailable bulk data')
        array[index] = values
        quat = np.asarray(block.localCoordSystem, dtype=float)
        if quat.shape != (len(index), 4): raise ValueError('Missing local shell coordinate basis')
        bases.setdefault(sp, np.full((count, 4), np.nan))[index] = quat
    if not result or any(not np.all(np.isfinite(v)) for v in list(result.values())+list(bases.values())):
        raise ValueError('Incomplete field coverage')
    return result, bases


def same_basis(a, b):
    delta = np.minimum(np.linalg.norm(a-b, axis=1), np.linalg.norm(a+b, axis=1))
    if np.max(delta) > 1e-7: raise ValueError('Incompatible local coordinate systems between paired fields')


def process(run_dir, output_dir=None, audit_dir=None):
    from odbAccess import openOdb
    import abaqus_modal_wavelengths as base
    run_dir = os.path.abspath(run_dir)
    output_dir = os.path.abspath(output_dir or os.path.join(run_dir, 'modal_shell_energy'))
    if os.path.exists(output_dir) and os.listdir(output_dir): raise ValueError('Choose a new or empty output directory')
    build = read_json(one_file(run_dir, '_build.json')); source = build['source_inputs']
    odb_path = one_file(run_dir, '.odb')
    if os.path.exists(os.path.splitext(odb_path)[0]+'.lck'): raise ValueError('ODB is locked')
    t, young, nu = validate_input(os.path.splitext(odb_path)[0]+'.inp', source)
    base_report = read_json(one_file(run_dir, '_modal_wavelengths_report.json'))
    if base_report.get('job_completion') != 'completed': raise ValueError('Completed base report required')
    audit_path = os.path.join(audit_dir or os.path.join(run_dir, 'modal_dsm_audit'), 'modal_audit.json')
    audit = read_json(audit_path)
    digest = hashlib.sha256()
    with open(odb_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): digest.update(chunk)
    if digest.hexdigest() != audit['source_odb_sha256']: raise ValueError('Audit and ODB are from different runs')
    by_mode = {r['mode']: r for r in audit['modes']}
    odb = openOdb(path=odb_path, readOnly=True)
    records, profiles, inventory = [], [], []
    maps = {}; c = plane_stress_matrix(young, nu)
    try:
        geo = geometry(odb, 'xyz'.index(base_report['axis'])); area = geo['area']; centers = geo['centers']
        instances = sorted(geo['tables']); part_masks = {n: np.array([k[0] == n for k in geo['keys']]) for n in instances}
        z = centers[:, geo['axis']]; z0 = min(m['start'] for m in base_report['instances'])
        length = base_report['length_mm']; bins = np.linspace(z0, z0+length, 73)
        bin_id = np.clip(np.searchsorted(bins, z, side='right')-1, 0, len(bins)-2)
        band = float(build['target_mesh_mm'])
        near_bolts = np.min(np.abs(z[:, None]-(z0+np.asarray(source['bolt_positions_mm']))), axis=1) <= band
        near_ends = (z-z0 <= 2*band) | (z0+length-z <= 2*band)
        frames = [f for f in odb.steps[base_report['step']].frames if base.frame_eigen(f)]
        if len(frames) != len(by_mode): raise ValueError('Audit mode coverage differs from ODB')
        for frame in frames:
            mode, eigenvalue = base.frame_eigen(frame); old = by_mode[mode]
            if not np.isclose(eigenvalue, old['eigenvalue'], rtol=1e-7): raise ValueError('Eigenvalue mismatch')
            if not REQUIRED.issubset(frame.fieldOutputs.keys()): raise ValueError('Detailed output fields missing')
            fields, axes = {}, {}
            for name in ('S', 'E', 'SF', 'SE', 'SM', 'SK'):
                fields[name], axes[name] = read_field(frame.fieldOutputs[name], geo)
            if set(fields['S']) != set(range(1, 6)) or set(fields['E']) != set(range(1, 6)):
                raise ValueError('All five Simpson section points required')
            for name in ('SF', 'SE', 'SM', 'SK'):
                if set(fields[name]) != {0}: raise ValueError('Unexpected section-resultant sampling')
            for name in axes:
                for sp in axes[name]: same_basis(axes['S'][1], axes[name][sp])
            if not inventory:
                inventory = [dict(field=n, components=list(frame.fieldOutputs[n].componentLabels),
                    precision=str(frame.fieldOutputs[n].values[0].precision), section_points=sorted(fields[n])) for n in fields]
            def take(name, suffixes, sp=0):
                return components(fields[name][sp], frame.fieldOutputs[name].componentLabels, [name+x for x in suffixes])
            stress = np.stack([take('S', ['11', '22', '12'], sp) for sp in range(1, 6)])
            strain = np.stack([take('E', ['11', '22', '12'], sp) for sp in range(1, 6)])
            constitutive_error = float(np.linalg.norm(stress-strain @ c.T)/max(np.linalg.norm(stress), 1e-250))
            if constitutive_error > 1e-4: raise ValueError('Stress/engineering strain elastic consistency failed: '+str(constitutive_error))
            energy = inplane_partition(stress, strain, t)
            sf = take('SF', ['1', '2', '3']); se = take('SE', ['1', '2', '3'])
            force_error = float(np.linalg.norm(sf-t*energy['mean_stress'])/max(np.linalg.norm(sf), 1e-250))
            strain_error = float(np.linalg.norm(se-energy['mean_strain'])/max(np.linalg.norm(se), 1e-250))
            membrane_error = float(np.linalg.norm(.5*np.sum(sf*se, axis=1)-energy['membrane'])/max(np.linalg.norm(energy['membrane']), 1e-250))
            if max(force_error, strain_error, membrane_error) > 1e-4:
                raise ValueError('Independent section force/strain consistency failed')
            shear = .5*np.sum(take('SF', ['4', '5'])*take('SE', ['4', '5']), axis=1)
            scale = max(float(np.max(energy['total'])), 1e-250)
            if np.min(shear) < -scale*1e-8: raise ValueError('Negative transverse shear energy')
            densities = np.column_stack((energy['membrane'], energy['bending'], np.maximum(shear, 0.)))
            element = densities.sum(axis=1)*area; total = float(element.sum())
            if total <= 0: raise ValueError('Zero reconstructed shell energy')
            percentages = 100*np.sum(densities*area[:, None], axis=0)/total
            # Common eigenvector amplitude for stresses and energy; no physical load is implied.
            umax = 0.
            for block in frame.fieldOutputs['U'].bulkDataBlocks:
                if block.instance and block.instance.name in geo['tables']:
                    umax = max(umax, float(np.max(np.linalg.norm(np.asarray(block.data), axis=1))))
            if umax <= 0: raise ValueError('No translational mode amplitude')
            mises = np.sqrt(stress[:, :, 0]**2-stress[:, :, 0]*stress[:, :, 1]+stress[:, :, 1]**2+3*stress[:, :, 2]**2)
            sp_index, hot = np.unravel_index(int(np.argmax(mises)), mises.shape)
            profile = np.bincount(bin_id, weights=element, minlength=72)*100/total
            record = dict(mode=mode, eigenvalue=eigenvalue, stress_MPa=old['stress_MPa'],
                half_wavelength_mm=old['half_wavelength_mm'], geometric_family=old['family'],
                geometric_LDG_percent=old['percentages'], cluster_id=old['cluster_id'],
                membrane_percent=float(percentages[0]), bending_percent=float(percentages[1]),
                transverse_shear_percent=float(percentages[2]),
                shell_energy_per_umax_squared_N_per_mm=total/umax**2,
                peak_mises_per_umax_MPa_per_mm=float(mises.max()/umax), peak_instance=geo['keys'][hot][0],
                peak_element=geo['keys'][hot][1], peak_section_point=int(sp_index+1),
                peak_element_centroid_mm=centers[hot].tolist(),
                part_energy_percent={n: float(100*element[mask].sum()/total) for n, mask in part_masks.items()},
                bolt_row_band_energy_percent=float(100*element[near_bolts].sum()/total),
                end_band_energy_percent=float(100*element[near_ends].sum()/total),
                constitutive_relative_error=constitutive_error, force_resultant_relative_error=force_error,
                membrane_crosscheck_relative_error=membrane_error,
                flags=list(old['flags']), **concentration(area, element))
            for name, unit in (('SM', 'N_per_mm'), ('SK', 'per_mm2')):
                values = take(name, ['1', '2', '3'])
                for j in range(3):
                    record['peak_'+name+str(j+1)+'_per_umax_'+unit] = float(np.max(np.abs(values[:, j]))/umax)
            records.append(record); profiles.append(profile.tolist())
            # Per-element share maps are compressed, reusable plotting data, never used as a new ODB.
            maps[mode] = element/total*100
            if len(records) % 25 == 0: print('Detailed shell diagnostics: %d/%d' % (len(records), len(frames))); sys.stdout.flush()
    finally:
        odb.close()
    result = dict(schema_version=1, source_odb=odb_path, source_odb_sha256=digest.hexdigest(),
        audit_source=audit_path, modes_processed=len(records), field_inventory=inventory,
        thickness_mm=t, E_MPa=young, nu=nu, elements=len(area), total_shell_area_mm2=float(area.sum()),
        longitudinal_bin_edges_mm=bins.tolist(), longitudinal_energy_percent=profiles,
        bolt_row_band_halfwidth_mm=band, bolt_row_band_area_percent=float(100*area[near_bolts].sum()/area.sum()),
        end_band_width_mm=2*band, end_band_area_percent=float(100*area[near_ends].sum()/area.sum()),
        modes=records, limitations=[
            'Physical shell energy reconstructed from normalized modal stresses/strains, not total assembled elastic K energy.',
            'Hourglass/drilling stabilization, contact/constraint contributions and geometric stiffness work are not included.',
            'Membrane/bending/shear shares are NOT local/distortional/global family shares. Existing family labels are unchanged.',
            'Stress amplitudes are per 1 mm maximum nodal displacement; they are not stresses at the critical load or yield checks.',
            'Individual energy patterns in near-repeated eigenvalue clusters depend on the chosen eigenvector basis.',
            'Bolt-row bands include the whole section at those axial stations; this is NOT bolt or fastener energy.',
            'No new family-specific DSM value or nonlinear ultimate capacity is inferred.'])
    serialized = serialize_result(result)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'shell_energy_report.json'), 'w') as f: f.write(serialized)
    flat_keys = [k for k,v in records[0].items() if not isinstance(v, (list, dict))]
    extra = ['energy_'+n+'_percent' for n in instances]+['flags']
    with open(os.path.join(output_dir, 'shell_energy_modes.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys+extra, extrasaction='ignore'); writer.writeheader()
        for r in records:
            row = dict(r, flags=';'.join(r['flags']))
            row.update({'energy_'+n+'_percent': r['part_energy_percent'][n] for n in instances}); writer.writerow(row)
    np.savez_compressed(os.path.join(output_dir, 'shell_energy_maps.npz'),
        modes=[r['mode'] for r in records], instances=np.array([k[0] for k in geo['keys']]),
        elements=[k[1] for k in geo['keys']], centers_mm=centers, areas_mm2=area,
        element_energy_percent=np.asarray([maps[r['mode']] for r in records], dtype=np.float32))
    write_plots(output_dir, result, geo, maps)
    link_audit_report(os.path.dirname(audit_path), output_dir)
    print('SAVED detailed shell report: '+output_dir)
    return result


def write_plots(folder, result, geo, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = result['modes']; modes = [r['mode'] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10)); bottom = np.zeros(len(rows))
    for key, color, label in [('membrane_percent','#438acd','Membrane'),('bending_percent','#df9a38','Bending'),('transverse_shear_percent','#aa73b3','Transverse shear')]:
        y = np.array([r[key] for r in rows]); ax[0,0].bar(modes,y,bottom=bottom,width=1,color=color,label=label); bottom += y
    ax[0,0].set(ylim=(0,100), xlabel='Mode', ylabel='Reconstructed shell energy (%)', title='Physical mechanisms, NOT L/D/G families'); ax[0,0].legend(ncol=3,fontsize=8)
    ax[0,1].plot(modes,[r['effective_area_percent'] for r in rows], label='Effective participating area (%)')
    ax[0,1].plot(modes,[r['top_10pct_area_energy_percent'] for r in rows],label='Energy in highest-density 10% area (%)')
    ax[0,1].set(xlabel='Mode',ylabel='Percent',title='Spatial localization'); ax[0,1].legend(fontsize=8)
    im=ax[1,0].imshow(np.array(result['longitudinal_energy_percent']).T,origin='lower',aspect='auto',
        extent=[.5,len(rows)+.5,result['longitudinal_bin_edges_mm'][0],result['longitudinal_bin_edges_mm'][-1]],cmap='magma')
    width = result['longitudinal_bin_edges_mm'][1]-result['longitudinal_bin_edges_mm'][0]
    ax[1,0].set(xlabel='Mode index',ylabel='Axial coordinate (mm)',title='Energy per %.3g mm axial bin' % width)
    fig.colorbar(im,ax=ax[1,0],label='Percent of reconstructed energy per bin')
    ax[1,1].plot(modes,[r['bolt_row_band_energy_percent'] for r in rows],color='#267d72',label='Energy in bolt-row bands')
    ax[1,1].axhline(result['bolt_row_band_area_percent'],ls='--',color='#267d72',label='Area occupied by these bands')
    ax[1,1].plot(modes,[r['end_band_energy_percent'] for r in rows],color='#ab6674',label='Energy in end bands')
    ax[1,1].set(xlabel='Mode',ylabel='Percent',title='Axial localization; NOT bolt forces/energy'); ax[1,1].legend(fontsize=8)
    fig.suptitle('Detailed Buckle fields | reconstructed physical shell-energy diagnostics',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.97])
    for ext in ('png','pdf'): fig.savefig(os.path.join(folder,'shell_energy_dashboard.'+ext),dpi=170)
    plt.close(fig)
    chosen = list(dict.fromkeys([rows[0]['mode'], rows[min(1,len(rows)-1)]['mode'],
        min(rows,key=lambda r:r['effective_area_percent'])['mode'],max(rows,key=lambda r:r['membrane_percent'])['mode']]))
    centers=geo['centers']; axial=geo['axis']; transverse=[i for i in range(3) if i!=axial]
    fig=plt.figure(figsize=(14,14))
    for i,mode in enumerate(chosen):
        a=fig.add_subplot(2,2,i+1,projection='3d'); density=maps[mode]/geo['area']
        a.scatter(centers[:,transverse[0]],centers[:,transverse[1]],centers[:,axial],color='#adb8c7',s=.4,alpha=.09,depthshade=False)
        active=density >= .02*density.max()
        im=a.scatter(centers[active,transverse[0]],centers[active,transverse[1]],centers[active,axial],c=density[active]/density.max(),s=3,cmap='plasma',vmin=0,vmax=1,depthshade=False)
        a.set_title('Mode %d | relative energy density' % mode,pad=20,fontsize=11)
        a.set_xlabel('Section 1 (mm)',fontsize=8,labelpad=2);a.set_ylabel('Section 2 (mm)',fontsize=8,labelpad=2);a.set_zlabel('Axial (mm)',fontsize=8,labelpad=4)
        a.tick_params(labelsize=7,pad=0)
        a.set_box_aspect((1,1,2)); fig.colorbar(im,ax=a,shrink=.55,pad=.10)
    fig.suptitle('Undeformed centroids | gray: all elements | colored: at least 2% of peak density\nEach view is normalized to its own maximum; display threshold does not affect calculations',fontsize=12)
    fig.subplots_adjust(left=.03,right=.96,bottom=.08,top=.90,hspace=.35,wspace=.15)
    fig.savefig(os.path.join(folder,'shell_energy_localization.png'),dpi=170);plt.close(fig)
    summary_lines=['# گزارش تکمیلی خروجی‌های جدید کمانش', '', 'تعداد مودهای بررسی‌شده: '+str(len(rows)),
        '', 'این گزارش انرژی فیزیکی پوسته را به غشایی، خمشی و برش عرضی تفکیک می‌کند؛ این‌ها سهم انرژی خانواده‌های L/D/G نیستند.', '',
        '| مود | تنش مرجع بحرانی MPa | غشایی % | خمشی % | برش عرضی % | مساحت مؤثر % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows[:10]: summary_lines.append('| %d | %.3f | %.3f | %.3f | %.3f | %.2f |' % (r['mode'],r['stress_MPa'] or r['eigenvalue'],r['membrane_percent'],r['bending_percent'],r['transverse_shear_percent'],r['effective_area_percent']))
    localized = min(rows, key=lambda r: r['effective_area_percent'])
    summary_lines += ['', '## یافته‌های کل مجموعه', '',
        '- سهم خمشی در مودهای بررسی‌شده بین %.2f و %.2f درصد است.' % (min(r['bending_percent'] for r in rows),max(r['bending_percent'] for r in rows)),
        '- سهم غشایی بین %.2f و %.2f درصد و سهم برش عرضی بین %.2f و %.2f درصد است.' % (min(r['membrane_percent'] for r in rows),max(r['membrane_percent'] for r in rows),min(r['transverse_shear_percent'] for r in rows),max(r['transverse_shear_percent'] for r in rows)),
        '- کمترین مساحت مؤثر مربوط به مود %d است: %.2f درصد. در این مود %.2f درصد انرژی در ۱۰ درصد مساحت با بیشترین چگالی قرار دارد.' % (localized['mode'],localized['effective_area_percent'],localized['top_10pct_area_energy_percent']),
        '- سهم قطعات در همین مود: '+', '.join('%s: %.2f%%' % (k,v) for k,v in localized['part_energy_percent'].items()),
        '- مساحت مؤثر یک شاخص تمرکز است؛ به معنی بی‌اثر بودن تمام نقاط خارج از آن نیست.',
        '- این یافته‌ها رتبه‌بندی بررسی شکل مود را کامل‌تر می‌کنند؛ برچسب هندسی یا مقدار Fcr خانواده‌ها با این انرژی‌ها تأیید نمی‌شود.']
    summary_lines += ['', 'حداکثر خطای نسبی سازگاری تنش–کرنش: %.3g' % max(r['constitutive_relative_error'] for r in rows),
        'حداکثر خطای مستقل انرژی غشایی از SF/SE: %.3g' % max(r['membrane_crosscheck_relative_error'] for r in rows), '',
        'این کنترل‌ها سازگاری داخلی داده‌ها و فرمول استخراج را می‌سنجند؛ جایگزین همگرایی مش و اعتبارسنجی فیزیکی نیستند.', '',
        'انرژی پایدارسازی عددی، تماس و کار سختی هندسی در این بازسازی لحاظ نشده‌اند. درصدها نسبت به مجموع انرژی بازسازی‌شده‌اند. تنش مودال گزارش‌شده به دامنهٔ حداکثر جابه‌جایی یک میلی‌متر مقیاس شده؛ تنش واقعی در بار بحرانی نیست.',
        '', 'شکل‌های نزدیک به مقدار ویژهٔ تکراری ممکن است به صورت ترکیب دیگری ظاهر شوند؛ توزیع انرژی یک بردار ویژه در چنین گروهی یکتا نیست.',
        '', 'منابع: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-r-shellgeneral.htm',
        'https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-shearflexshells.htm']
    with open(os.path.join(folder,'SUMMARY_FA.md'),'w',encoding='utf-8') as f:f.write('\n'.join(summary_lines)+'\n')
    embedded=[]
    for name in ('shell_energy_dashboard.png','shell_energy_localization.png'):
        with open(os.path.join(folder,name),'rb') as f:embedded.append(base64.b64encode(f.read()).decode('ascii'))
    table=''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in (r['mode'],r['geometric_family'],round(r['eigenvalue'],5),round(r['membrane_percent'],3),round(r['bending_percent'],3),round(r['transverse_shear_percent'],3),round(r['effective_area_percent'],2)))+'</tr>' for r in rows)
    html='''<!doctype html><meta charset="utf-8"><title>Detailed shell energy</title><style>body{max-width:1400px;margin:30px auto;padding:20px;font:16px system-ui;background:#101a2a;color:#e5edf7}img{width:100%;margin:15px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #45546a}a{color:#9dcaff}.note{background:#24344c;padding:20px;line-height:1.7}</style><h1>Detailed Buckle shell-energy report</h1><p class="note">Membrane / bending / transverse shear are physical shell mechanisms, NOT local / distortional / global families. Percentages exclude numerical stabilization, contact and constraint contributions. Modal stress amplitudes are not loaded-state stresses.</p>'''
    html+=''.join('<img src="data:image/png;base64,'+s+'">' for s in embedded)
    html+='<p><a href="shell_energy_modes.csv">CSV</a> · <a href="shell_energy_report.json">Full JSON</a> · <a href="SUMMARY_FA.md">Persian summary</a></p><table><tr><th>Mode</th><th>Geometric proxy</th><th>Eigenvalue</th><th>Membrane %</th><th>Bending %</th><th>Shear %</th><th>Effective area %</th></tr>'+table+'</table>'
    with open(os.path.join(folder,'shell_energy_report.html'),'w',encoding='utf-8') as f:f.write(html)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--output-dir');parser.add_argument('--audit-dir')
    args=parser.parse_args();process(args.run_dir,args.output_dir,args.audit_dir)
