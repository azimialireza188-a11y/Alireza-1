# -*- coding: utf-8 -*-
"""Compact Parquet sidecar for post-run scientific review.

The Abaqus ODB remains the authoritative result. This module creates a small,
upload-friendly analysis bundle with mode summaries, harmonic content and
canonical cross-section shapes so the run can be reviewed without transferring
the full ODB.

PyArrow is imported lazily: Abaqus/CAE can import this module even when its
bundled Python does not provide PyArrow. A normal system Python with PyArrow can
then execute the exporter after the Abaqus audit completes.
"""
from __future__ import print_function
import argparse
import hashlib
import json
import math
import os
import sys
import zipfile
import numpy as np


FORMAT_VERSION = 'parquet_analysis_bundle_v1'


def runtime_probe():
    try:
        import pyarrow
        import pyarrow.parquet
        return dict(available=True,pyarrow_version=str(pyarrow.__version__),
                    python=sys.executable)
    except Exception as exc:
        return dict(available=False,error='%s: %s' % (type(exc).__name__,exc),
                    python=sys.executable)


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa,pq
    except Exception as exc:
        raise RuntimeError(
            'Parquet export requires PyArrow in the selected Python runtime. '
            'Install it with: python -m pip install pyarrow') from exc


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _finite(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (TypeError,ValueError):
        return None


def _scalar_or_json(value):
    if value is None or isinstance(value,(str,bool,int,float)):
        if isinstance(value,float) and not math.isfinite(value):
            return None
        return value
    return _json(value)


def _flatten(prefix,value,rows):
    if isinstance(value,dict):
        for key in sorted(value):
            child=('%s.%s' % (prefix,key)) if prefix else str(key)
            _flatten(child,value[key],rows)
    else:
        rows.append(dict(key=prefix,value_json=_json(value)))


def _table(rows):
    pa,unused=_arrow()
    return pa.Table.from_pylist(rows)


def _mode_summary_rows(summary):
    keys=[
        'mode','eigenvalue','stress_MPa','dominant_m','half_wavelength_mm',
        'L_energy_percent','D_energy_percent','G_energy_percent','O_energy_percent',
        'L_vector_percent','D_vector_percent','G_vector_percent','O_vector_percent',
        'assembly_percent','seam_normal_opening_index','seam_transverse_slip_index',
        'seam_longitudinal_slip_index','interpiece_interaction_percent',
        'final_family','family','quality_state','harmonic_residual',
        'mechanical_residual_percent','metric_sensitivity_pp','cluster_id',
        'geometric_screening_family','percentage_kind','mechanical_eligible',
        'basis_condition','numeric_backend','energy_closure_relative',
        'eigenspace_stable_family'
    ]
    rows=[]
    for mode in summary.get('modes',[]):
        row={key:_scalar_or_json(mode.get(key)) for key in keys}
        row['flags']=';'.join(str(x) for x in (mode.get('flags') or []))
        row['geometric_screening_percentages_json']=_json(
            mode.get('geometric_screening_percentages') or [])
        row['cross_terms_percent_json']=_json(mode.get('cross_terms_percent') or {})
        row['basis_hashes_json']=_json(mode.get('basis_hashes') or {})
        rows.append(row)
    return rows


def _harmonic_rows(mapped_modes,harmonics):
    rows=[]
    for mapped,h in zip(mapped_modes,harmonics):
        translation=np.asarray(h.get('translation_power',[]),dtype=float)
        rotation=np.asarray(h.get('rotation_power',[]),dtype=float)
        shares=np.asarray(h.get('shares',[]),dtype=float)
        tsum=float(np.sum(translation)); rsum=float(np.sum(rotation))
        for index,share in enumerate(shares):
            m=index+1
            rows.append(dict(
                mode=int(mapped['mode']),eigenvalue=float(mapped['eigenvalue']),
                harmonic_m=m,harmonic_share=float(share),
                translation_power=float(translation[index]) if index<len(translation) else None,
                rotation_power=float(rotation[index]) if index<len(rotation) else None,
                translation_power_fraction=(float(translation[index]/tsum) if tsum>0 else None),
                rotation_power_fraction=(float(rotation[index]/rsum) if rsum>0 else None),
                dominant=bool(m==int(h['dominant_m'])),
                half_wavelength_mm=(None if m<=0 else float(h.get('half_wavelength_mm',0.))*int(h['dominant_m'])/m),
                translation_relative_residual=float(h.get('translation_relative_residual',0.)),
                rotation_relative_residual=float(h.get('rotation_relative_residual',0.)),
                total_relative_residual=float(h.get('relative_residual',0.))))
    return rows


def _node_lookup(reference):
    return {int(node['id']):node for node in reference.get('nodes',[])}


def _peak_rows(reference,mapped_modes):
    nodes=reference.get('nodes',[])
    if not nodes:
        return []
    thickness=float(reference.get('material',{}).get('thickness_mm',1.0))
    rotation_scale=thickness/math.sqrt(12.0)
    rows=[]
    for mapped in mapped_modes:
        U=np.asarray(mapped['U'],dtype=float)
        UR=np.asarray(mapped['UR'],dtype=float)
        amplitude=np.sum(U*U,axis=(1,2))+rotation_scale**2*np.sum(UR*UR,axis=(1,2))
        peak=int(np.argmax(amplitude))
        up=U[peak]; rp=UR[peak]
        unorm=max(float(np.max(np.linalg.norm(up,axis=1))),1e-250)
        rnorm=max(float(np.max(np.linalg.norm(rp,axis=1))),1e-250)
        for j,node in enumerate(nodes):
            rows.append(dict(
                mode=int(mapped['mode']),eigenvalue=float(mapped['eigenvalue']),
                peak_z_mm=float(mapped['z'][peak]),node_id=int(node['id']),
                piece=str(node.get('piece','')),x_mm=float(node['x']),y_mm=float(node['y']),
                U1=float(up[j,0]),U2=float(up[j,1]),U3=float(up[j,2]),
                UR1=float(rp[j,0]),UR2=float(rp[j,1]),UR3=float(rp[j,2]),
                U_norm_factor=unorm,UR_norm_factor=rnorm,
                U1_normalized=float(up[j,0]/unorm),
                U2_normalized=float(up[j,1]/unorm),
                U3_normalized=float(up[j,2]/unorm),
                UR1_normalized=float(rp[j,0]/rnorm),
                UR2_normalized=float(rp[j,1]/rnorm),
                UR3_normalized=float(rp[j,2]/rnorm)))
    return rows


def _harmonic_section_rows(reference,mapped_modes,harmonics,min_share):
    nodes=reference.get('nodes',[])
    rows=[]
    threshold=max(0.0,float(min_share))
    for mapped,h in zip(mapped_modes,harmonics):
        shares=np.asarray(h.get('shares',[]),dtype=float)
        dominant=int(h.get('dominant_m',1))
        components=h.get('components',{})
        for m,share in enumerate(shares,1):
            if float(share)<threshold and m!=dominant:
                continue
            component=components.get(m)
            if component is None:
                continue
            U=np.asarray(component['U'],dtype=float)
            UR=np.asarray(component['UR'],dtype=float)
            for j,node in enumerate(nodes):
                rows.append(dict(
                    mode=int(mapped['mode']),eigenvalue=float(mapped['eigenvalue']),
                    harmonic_m=int(m),harmonic_share=float(share),
                    dominant=bool(m==dominant),node_id=int(node['id']),
                    piece=str(node.get('piece','')),x_mm=float(node['x']),y_mm=float(node['y']),
                    U1=float(U[j,0]),U2=float(U[j,1]),U3=float(U[j,2]),
                    UR1=float(UR[j,0]),UR2=float(UR[j,1]),UR3=float(UR[j,2])))
    return rows


def _cluster_rows(summary):
    rows=[]
    for cluster in summary.get('clusters',summary.get('eigenspace_clusters',[])):
        bounds=cluster.get('bounds') or {}
        isolation=cluster.get('spectral_isolation') or {}
        eigen=cluster.get('eigenvalue_range') or [None,None]
        rows.append(dict(
            cluster_id=cluster.get('cluster_id'),
            modes_json=_json(cluster.get('modes') or []),
            eigenvalue_min=_finite(eigen[0] if len(eigen)>0 else None),
            eigenvalue_max=_finite(eigen[1] if len(eigen)>1 else None),
            family=cluster.get('family'),
            stable_family=bounds.get('stable_family'),
            min_percent_json=_json(bounds.get('min_percent') or []),
            max_percent_json=_json(bounds.get('max_percent') or []),
            mean_percent_json=_json(bounds.get('mean_percent') or []),
            O_min_percent=_finite(bounds.get('O_min_percent')),
            O_max_percent=_finite(bounds.get('O_max_percent')),
            spectral_isolated=isolation.get('isolated'),
            flags=';'.join(str(x) for x in (cluster.get('flags') or []))))
    return rows


def pipeline_report_tables(report):
    report=report or {}
    invocation=report.get('invocation') or {}
    run_row=dict(
        format=report.get('format'),
        status=report.get('status'),
        started_at=report.get('started_at'),
        updated_at=report.get('updated_at'),
        total_elapsed_seconds=_finite(report.get('total_elapsed_seconds')),
        normalized_command=invocation.get('normalized_command'),
        effective_args_json=_json(invocation.get('effective_args') or []),
        process_argv_json=_json(invocation.get('process_argv') or []),
        launch_cwd=invocation.get('launch_cwd'),
        script_path=invocation.get('script_path'),
        script_sha256=invocation.get('script_sha256'),
        git_commit=invocation.get('git_commit'),
        host=invocation.get('host'),
        platform=invocation.get('platform'),
        machine=invocation.get('machine'),
        processor=invocation.get('processor'),
        python_version=invocation.get('python_version'),
        python_executable=invocation.get('python_executable'),
        process_id=invocation.get('process_id'),
        submitted=report.get('submitted'),
        solver_status=report.get('solver_status'),
        solver_api_status=report.get('solver_api_status'),
        completion_evidence=report.get('completion_evidence'),
        resource_plan_json=_json(report.get('resource_plan') or {}),
        effective_settings_json=_json(report.get('effective_settings') or {}))
    stage_rows=[]
    for row in report.get('stages',[]) or []:
        stage_rows.append(dict(
            scope=row.get('scope'),parent_stage=row.get('parent_stage'),
            stage=row.get('stage'),status=row.get('status'),
            weight_percent=_finite(row.get('weight_percent')),
            duration_seconds=_finite(row.get('duration_seconds')),
            started_at=row.get('started_at'),finished_at=row.get('finished_at')))
    output_rows=[]
    for path,item in sorted((report.get('outputs') or {}).items()):
        output_rows.append(dict(
            path=str(path),bytes=int((item or {}).get('bytes',0)),
            modified_at=(item or {}).get('modified_at')))
    return dict(
        pipeline_run=_table([run_row]),
        pipeline_stages=_table(stage_rows),
        pipeline_outputs=_table(output_rows))


def analysis_tables(summary,reference,mapped_modes,harmonics,
                    harmonic_section_min_share=.001,pipeline_report=None):
    if len(mapped_modes)!=len(harmonics):
        raise ValueError('mapped_modes and harmonics must have equal length')
    mode_numbers=[int(x.get('mode')) for x in mapped_modes]
    summary_modes=[int(x.get('mode')) for x in summary.get('modes',[])]
    if summary_modes and mode_numbers!=summary_modes:
        raise ValueError('Mapped modes are not in the same order as modal audit summary')
    provenance=[]
    selected=dict(
        format=FORMAT_VERSION,
        source_odb=summary.get('source_odb'),
        source_odb_sha256=summary.get('source_odb_sha256'),
        model_signature=summary.get('model_signature'),
        percentage_kind=summary.get('percentage_kind'),
        basis_definition_id=summary.get('basis_definition_id'),
        fields_available_in_all_modes=summary.get('fields_available_in_all_modes'),
        mechanical_classification=summary.get('mechanical_classification'),
        basis_metadata=summary.get('basis_metadata'),
        settings=summary.get('settings'),
        reference_hash=reference.get('definition_hash'),
        reference_version=reference.get('version'),
        plate_definition=reference.get('plate_definition'),
        plate_definition_version=reference.get('plate_definition_version'))
    _flatten('',selected,provenance)
    tables=dict(
        modal_summary=_table(_mode_summary_rows(summary)),
        harmonic_summary=_table(_harmonic_rows(mapped_modes,harmonics)),
        peak_sections=_table(_peak_rows(reference,mapped_modes)),
        harmonic_sections=_table(_harmonic_section_rows(
            reference,mapped_modes,harmonics,harmonic_section_min_share)),
        clusters=_table(_cluster_rows(summary)),
        provenance=_table(provenance))
    if pipeline_report is not None:
        tables.update(pipeline_report_tables(pipeline_report))
    return tables


def _sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet(table,path):
    unused,pq=_arrow()
    try:
        pq.write_table(table,path,compression='zstd',use_dictionary=True,
                       write_statistics=True,row_group_size=65536)
        return 'zstd'
    except Exception:
        pq.write_table(table,path,compression='snappy',use_dictionary=True,
                       write_statistics=True,row_group_size=65536)
        return 'snappy'


def write_bundle(tables,output_root,bundle_name='modal_analysis_parquet_bundle'):
    output_root=os.path.abspath(output_root)
    directory=os.path.join(output_root,bundle_name)
    os.makedirs(directory,exist_ok=True)
    files={}
    for name,table in sorted(tables.items()):
        filename=name+'.parquet'
        path=os.path.join(directory,filename)
        compression=_write_parquet(table,path)
        files[filename]=dict(rows=int(table.num_rows),columns=int(table.num_columns),
                             bytes=os.path.getsize(path),sha256=_sha256(path),
                             compression=compression)
    manifest=dict(format=FORMAT_VERSION,files=files,
                  purpose='Compact post-run modal review; ODB remains authoritative',
                  upload_recommendation='Upload the ZIP bundle for review; full ODB is not normally required.')
    manifest_path=os.path.join(directory,'manifest.json')
    with open(manifest_path,'w',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2,sort_keys=True)
    zip_path=_rebuild_bundle_zip(directory,manifest)
    return dict(directory=directory,zip_path=zip_path,manifest_path=manifest_path,
                files=files,bytes=os.path.getsize(zip_path))


def _bundle_zip_path(directory):
    directory=os.path.abspath(directory)
    return os.path.join(os.path.dirname(directory),os.path.basename(directory)+'.zip')


def _rebuild_bundle_zip(directory,manifest):
    zip_path=_bundle_zip_path(directory)
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_STORED) as archive:
        for filename in sorted((manifest.get('files') or {}).keys()):
            path=os.path.join(directory,filename)
            if os.path.isfile(path):
                archive.write(path,arcname=filename)
        manifest_path=os.path.join(directory,'manifest.json')
        archive.write(manifest_path,arcname='manifest.json')
    return zip_path


def refresh_pipeline_tables(bundle_directory,report_path):
    """Refresh only the tiny pipeline Parquet tables after export timing is final."""
    bundle_directory=os.path.abspath(bundle_directory)
    report_path=os.path.abspath(report_path)
    with open(report_path,encoding='utf-8') as stream:
        report=json.load(stream)
    tables=pipeline_report_tables(report)
    manifest_path=os.path.join(bundle_directory,'manifest.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path,encoding='utf-8') as stream:
            manifest=json.load(stream)
    else:
        manifest=dict(format=FORMAT_VERSION,files={})
    files=manifest.setdefault('files',{})
    for name,table in tables.items():
        filename=name+'.parquet'
        path=os.path.join(bundle_directory,filename)
        compression=_write_parquet(table,path)
        files[filename]=dict(
            rows=int(table.num_rows),columns=int(table.num_columns),
            bytes=os.path.getsize(path),sha256=_sha256(path),
            compression=compression)
    with open(manifest_path,'w',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2,sort_keys=True)
    zip_path=_rebuild_bundle_zip(bundle_directory,manifest)
    return dict(directory=bundle_directory,zip_path=zip_path,
                manifest_path=manifest_path,files=files,bytes=os.path.getsize(zip_path))


def _discover_one(directory,suffix):
    paths=[os.path.join(directory,name) for name in os.listdir(directory)
           if name.endswith(suffix)]
    if len(paths)!=1:
        raise ValueError('Expected exactly one %s in %s; found %d' %
                         (suffix,directory,len(paths)))
    return paths[0]


def discover_audit_dir(run_dir):
    candidates=[]
    for name in os.listdir(run_dir):
        path=os.path.join(run_dir,name)
        if (os.path.isdir(path) and name.startswith('modal_dsm_audit') and
                os.path.isfile(os.path.join(path,'modal_audit.json'))):
            candidates.append(path)
    if not candidates:
        raise ValueError('No completed modal_dsm_audit* directory found in '+run_dir)
    return max(candidates,key=os.path.getmtime)


def export_run(run_dir,audit_dir=None,output_root=None,
               harmonic_section_min_share=.001):
    from abaqus_dsm_modal_audit import canonical_reference_from_build
    from abaqus_modal_archive import open_modal_archive,map_mode_to_reference
    from abaqus_modal_harmonics import decompose_mode
    run_dir=os.path.abspath(run_dir)
    audit_dir=os.path.abspath(audit_dir or discover_audit_dir(run_dir))
    output_root=os.path.abspath(output_root or audit_dir)
    with open(os.path.join(audit_dir,'modal_audit.json'),encoding='utf-8') as stream:
        summary=json.load(stream)
    with open(_discover_one(run_dir,'_build.json'),encoding='utf-8') as stream:
        build=json.load(stream)
    reference=canonical_reference_from_build(build)
    archive_path=os.path.join(audit_dir,'modal_shapes_U_UR.npz')
    if not os.path.isfile(archive_path):
        raise ValueError('Mechanical U/UR archive is missing: '+archive_path)
    retained=[int(x) for x in (summary.get('basis_metadata',{}).get('retained_harmonics') or [])]
    if not retained:
        raise ValueError('Audit has no retained mechanical harmonics')
    max_harmonic=max(retained)
    mapped=[]; harmonics=[]
    with open_modal_archive(archive_path) as archive:
        for index in range(len(archive)):
            item=map_mode_to_reference(archive,index,reference)
            h=decompose_mode(item['z'],item['U'],item['UR'],
                             reference['length_mm'],max_harmonic,bc='S-S')
            mapped.append(item); harmonics.append(h)
    pipeline_report=None
    pipeline_path=os.path.join(run_dir,'pipeline_run_report.json')
    if os.path.isfile(pipeline_path):
        with open(pipeline_path,encoding='utf-8') as stream:
            pipeline_report=json.load(stream)
    tables=analysis_tables(
        summary,reference,mapped,harmonics,
        harmonic_section_min_share=harmonic_section_min_share,
        pipeline_report=pipeline_report)
    return write_bundle(tables,output_root)


def parse_arguments(argv=None):
    parser=argparse.ArgumentParser(description='Export compact Parquet modal-analysis bundle')
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--audit-dir')
    parser.add_argument('--output-root')
    parser.add_argument('--harmonic-section-min-share',type=float,default=.001,
                        help='Minimum harmonic share stored with per-node coefficients; dominant harmonic is always stored')
    parser.add_argument('--probe',action='store_true',help='Only report whether PyArrow is available')
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_arguments(argv)
    if args.probe:
        result=runtime_probe()
        print(json.dumps(result,sort_keys=True))
        return 0 if result['available'] else 2
    result=export_run(args.run_dir,args.audit_dir,args.output_root,
                      args.harmonic_section_min_share)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
