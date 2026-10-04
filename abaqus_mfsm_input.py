"""Read-only INP inventory and sparse INITIAL small-rotation BEAM constraints.

This is not a contact tangent, S4R strain operator or verified active-base-state
reduction. Supports expanded, global-axis S4R parts, transformed instances,
singleton BEAM node sets and homogeneous numeric boundary DOFs. Unsupported
input forms fail explicitly instead of producing a partial constraint matrix.
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np

VERSION = 'mfsm-inp-initial-constraints-v2'
SUPPORTED_KEYWORDS = {
    'HEADING','PREPRINT','PART','NODE','ELEMENT','ELSET','NSET','SHELL SECTION','END PART',
    'ASSEMBLY','INSTANCE','END INSTANCE','END ASSEMBLY','MPC','MATERIAL','ELASTIC',
    'PLASTIC','DENSITY','SURFACE','SURFACE INTERACTION','FRICTION','SURFACE BEHAVIOR',
    'BOUNDARY','CONTACT','CONTACT INCLUSIONS','CONTACT EXCLUSIONS','CONTACT PROPERTY ASSIGNMENT',
    'STEP','BUCKLE','END STEP','DSLOAD','CLOAD','DLOAD','OUTPUT','NODE OUTPUT',
    'ELEMENT OUTPUT','CONTACT OUTPUT','RESTART'
}


def _fields(line):
    return [s.strip().strip('"\'') for s in line.split(',')]


def _blocks(text):
    current = None
    for number,line in enumerate(text.splitlines(),1):
        line = line.strip()
        if not line or line.startswith('**'): continue
        if line.startswith('*'):
            if current is not None: yield current
            fields = _fields(line[1:]); attrs = {}; flags = set()
            for token in fields[1:]:
                if '=' in token:
                    k,v = token.split('=',1); attrs[k.strip().upper()] = v.strip().strip('"\'')
                elif token: flags.add(token.upper())
            current = (fields[0].upper(),attrs,flags,[],number)
        else:
            if current is None: raise ValueError('Data before INP keyword')
            current[3].append(line)
    if current is not None: yield current


def _instance_points(nodes, lines):
    if len(lines)>2: raise ValueError('Unsupported instance transformation')
    xyz = np.array(list(nodes.values()),dtype=float)
    if lines:
        translation = np.array([float(x) for x in _fields(lines[0])])
        if translation.shape!=(3,): raise ValueError('Instance translation needs 3 values')
        xyz = xyz+translation
    if len(lines)==2:
        values = np.array([float(x) for x in _fields(lines[1])])
        if values.shape!=(7,): raise ValueError('Instance rotation needs 7 values')
        a = values[:3]; axis = values[3:6]-a
        if np.linalg.norm(axis)==0: raise ValueError('Zero rotation axis')
        x,y,z = axis/np.linalg.norm(axis); theta = np.deg2rad(values[6])
        skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        rotation = np.eye(3)+np.sin(theta)*skew+(1-np.cos(theta))*(skew@skew)
        xyz = (xyz-a)@rotation.T+a
    if not np.all(np.isfinite(xyz)): raise ValueError('Nonfinite instance coordinates')
    return dict(zip(nodes,xyz))


def read_input_text(text):
    parts = {}; instances = {}; sets = {}; boundary = []; mpcs = []
    keywords = []; steps = []; sections = []; elastic = []
    part = None; instance = None; step = None
    for key,attrs,flags,lines,line in _blocks(text):
        if key not in SUPPORTED_KEYWORDS or 'INPUT' in attrs:
            raise ValueError('Unsupported/external INP construct: '+key)
        keywords.append(key)
        if key in ('INCLUDE','PARAMETER','TRANSFORM','SYSTEM'):
            raise ValueError('Expanded global-coordinate input required: '+key)
        if key=='PART':
            part=attrs['NAME'].upper()
            if part in parts: raise ValueError('Duplicate part')
            parts[part]=dict(nodes={},elements={},sets={})
        elif key=='END PART': part=None
        elif key=='NODE':
            if part is None or instance is not None: raise ValueError('Only part-level nodes supported')
            if flags or set(attrs)-{'NSET','SYSTEM'} or attrs.get('SYSTEM','R').upper()!='R':
                raise ValueError('Only expanded rectangular NODE coordinates supported')
            for row in lines:
                f=_fields(row); label=int(f[0]); values=[float(v) for v in f[1:]]
                if len(values)!=3 or label in parts[part]['nodes'] or label<1 or not np.all(np.isfinite(values)):
                    raise ValueError('Invalid/duplicate 3D node')
                parts[part]['nodes'][label]=values
                if 'NSET' in attrs:parts[part]['sets'].setdefault(attrs['NSET'].upper(),[]).append(label)
        elif key=='ELEMENT':
            if part is None or attrs.get('TYPE','').upper()!='S4R': raise ValueError('Only part-level S4R supported')
            if flags or set(attrs)-{'TYPE','ELSET'}:raise ValueError('Unsupported ELEMENT options')
            for row in lines:
                f=[int(v) for v in _fields(row)]
                if len(f)!=5 or f[0] in parts[part]['elements'] or len(set(f[1:]))!=4:
                    raise ValueError('Invalid/duplicate S4R connectivity')
                parts[part]['elements'][f[0]]=tuple(f[1:])
        elif key=='INSTANCE':
            instance=attrs['NAME'].upper()
            if instance in instances: raise ValueError('Duplicate instance')
            if set(attrs)!= {'NAME','PART'} or flags: raise ValueError('Only independent part instances supported')
            instances[instance]=dict(part=attrs['PART'].upper(),transform=lines)
        elif key=='END INSTANCE': instance=None
        elif key=='NSET':
            if instance is not None or 'ELSET' in attrs: raise ValueError('Unsupported NSET form')
            if set(attrs)-{'NSET','INSTANCE'} or flags-{'GENERATE','UNSORTED','INTERNAL'}:raise ValueError('Unsupported NSET options')
            name=attrs['NSET'].upper(); target=parts[part]['sets'] if part else sets
            inst=attrs.get('INSTANCE','').upper()
            values=[]
            for row in lines:
                f=[int(v) for v in _fields(row) if v]
                if 'GENERATE' in flags:
                    if len(f) not in (2,3): raise ValueError('Invalid generated NSET')
                    start,end=f[:2]; stride=f[2] if len(f)==3 else 1
                    if start<1 or stride<1 or end<start or (end-start)%stride: raise ValueError('Invalid NSET range')
                    f=list(range(start,end+1,stride))
                values+=f
            if any(v<1 for v in values): raise ValueError('Invalid NSET labels')
            if part: target.setdefault(name,[]).extend(values)
            else:
                if not inst: raise ValueError('Assembly NSET requires explicit instance')
                target.setdefault(name,[]).extend((inst,v) for v in values)
        elif key=='MPC':
            if attrs or flags or part or instance: raise ValueError('Unsupported MPC options/scope')
            mpcs += [_fields(row) for row in lines]
        elif key=='BOUNDARY':
            boundary.append(dict(step=step,attrs=attrs,flags=flags,rows=[_fields(row) for row in lines]))
        elif key=='STEP':
            step=attrs.get('NAME','STEP_'+str(len(steps)+1)).upper()
            steps.append(dict(name=step,attrs=attrs,flags=sorted(flags)))
        elif key=='END STEP': step=None
        elif key=='BUCKLE':
            if not steps or not lines: raise ValueError('BUCKLE requires step and data')
            steps[-1]['buckle_data']=_fields(lines[0])
        elif key=='SHELL SECTION':
            sections.append(dict(part=part,definition=attrs,data=lines))
        elif key=='ELASTIC': elastic.append(dict(definition=attrs,data=lines))
    nodes={}; elements={}
    for name,spec in instances.items():
        if spec['part'] not in parts: raise ValueError('Unknown instance part')
        p=parts[spec['part']]
        if not p['nodes'] or not p['elements']: raise ValueError('Empty S4R part')
        points=_instance_points(p['nodes'],spec['transform'])
        nodes.update({(name,label):xyz for label,xyz in points.items()})
        elements.update({(name,label):tuple((name,n) for n in conn) for label,conn in p['elements'].items()})
        for set_name,labels in p['sets'].items():sets[name+'.'+set_name]=[(name,n) for n in labels]
    if not nodes: raise ValueError('No assembled shell nodes')
    if any(n not in nodes for conn in elements.values() for n in conn):raise ValueError('Element references missing node')
    return dict(nodes=nodes,elements=elements,sets=sets,mpcs=mpcs,boundary=boundary,
                parts=parts,instances=instances,steps=steps,sections=sections,elastic=elastic,
                keywords=keywords,input_sha256=hashlib.sha256(text.encode()).hexdigest())


def read_input(path):
    raw=Path(path).read_bytes();model=read_input_text(raw.decode('utf-8-sig'))
    model['input_sha256']=hashlib.sha256(raw).hexdigest()
    return model


def _resolve(model,token):
    token=token.upper(); sets=model['sets']; nodes=model['nodes']
    if token in sets: values=list(dict.fromkeys(sets[token]))
    elif '.' in token:
        name,label=token.rsplit('.',1); values=[(name,int(label))]
    else: raise ValueError('Unresolved node/set '+token)
    if not values or any(v not in nodes for v in values): raise ValueError('Empty set or missing node '+token)
    return values


def compile_initial_constraints(model):
    from scipy.sparse import coo_matrix
    from builtup_connection_operator import beam_mpc_matrix
    forbidden={'EQUATION','TIE','COUPLING','KINEMATIC','DISTRIBUTING','RIGID BODY','EMBEDDED ELEMENT','RELEASE'}
    if any(k in forbidden or k.startswith('CONNECTOR') for k in model['keywords']):
        raise ValueError('Unsupported additional kinematic constraints')
    steps=model['steps']
    if len(steps)>1 or any('buckle_data' not in s or 'PERTURBATION' not in s['flags'] or
            s['attrs'].get('NLGEOM','NO').upper()!='NO' for s in steps):
        raise ValueError('Only initial geometry and one linear BUCKLE perturbation supported')
    node_keys=list(model['nodes']);indices={k:i for i,k in enumerate(node_keys)}
    keys=[(name,label,dof) for name,label in node_keys for dof in range(1,7)]
    rows=[];cols=[];values=[];row=0;dependent=set();pairs=[]
    for f in model['mpcs']:
        if len(f)!=3 or f[0].upper()!='BEAM': raise ValueError('Only two-node BEAM MPC supported')
        slave=_resolve(model,f[1]);master=_resolve(model,f[2])
        if len(slave)!=1 or len(master)!=1: raise ValueError('BEAM sets must be singleton; set pairing not inferred')
        a,b=slave[0],master[0]
        if a==b or a in dependent: raise ValueError('Repeated/self dependent BEAM node')
        dependent.add(a);pairs.append(dict(dependent=a,independent=b,offset=(model['nodes'][a]-model['nodes'][b]).tolist()))
        block=beam_mpc_matrix(model['nodes'][b],model['nodes'][a])
        indices12=list(range(6*indices[b],6*indices[b]+6))+list(range(6*indices[a],6*indices[a]+6))
        r,c=np.nonzero(block);rows.extend((r+row).tolist());cols.extend(indices12[i] for i in c);values.extend(block[r,c].tolist());row+=6
    beam_rows=row
    initial=[b for b in model['boundary'] if b['step'] is None]
    mode=[b for b in model['boundary'] if b['step'] is not None and b['attrs'].get('LOAD CASE','1')=='2']
    if mode:
        if any(b['attrs'].get('OP','MOD').upper()!='NEW' for b in mode):
            raise ValueError('Mode LOAD CASE=2 requires explicit OP=NEW in this extractor')
        selected=mode
    elif steps and any(b['step'] is not None for b in model['boundary']):
        raise ValueError('Step boundary definitions require explicit mode LOAD CASE=2')
    else:selected=initial
    fixed=set()
    for b in selected:
        if b['flags'] or set(b['attrs'])-{'OP','LOAD CASE'}:raise ValueError('Unsupported boundary options')
        for f in b['rows']:
            if len(f) not in (2,3,4):raise ValueError('Numeric homogeneous boundary required')
            start=int(f[1]);end=int(f[2]) if len(f)>2 and f[2] else start
            value=float(f[3]) if len(f)>3 and f[3] else 0.
            if not 1<=start<=end<=6 or value!=0:raise ValueError('Homogeneous boundary DOFs 1..6 required')
            for node in _resolve(model,f[0]):
                fixed.update(6*indices[node]+dof-1 for dof in range(start,end+1))
    for i in sorted(fixed): rows.append(row);cols.append(i);values.append(1.);row+=1
    c=coo_matrix((values,(rows,cols)),shape=(row,len(keys))).tocsr()
    metadata=dict(algorithm=VERSION,input_sha256=model['input_sha256'],node_keys=node_keys,
        beam_rows=beam_rows,boundary_rows=row-beam_rows,beam_pairs=pairs,
        coordinate_definition='GLOBAL_INITIAL_U1_U2_U3_UR1_UR2_UR3_NODE_MAJOR',
        boundary_selection='BUCKLE_LOAD_CASE_2_OP_NEW' if mode else 'INITIAL',
        active_base_state_verified=False,scientifically_eligible=False,
        contact_state='UNKNOWN_REQUIRES_BASE_STATE_EVIDENCE',
        limitation='Initial kinematics only; no contact/S4R tangent or nullspace/dense reduction constructed')
    return c,keys,metadata


def input_summary(model):
    return dict(algorithm=VERSION,input_sha256=model['input_sha256'],
        part_count=len(model['parts']),instance_count=len(model['instances']),
        node_count=len(model['nodes']),element_count=len(model['elements']),element_type='S4R',
        mpc_count=len(model['mpcs']),mpc_types=sorted(set(f[0].upper() for f in model['mpcs'])),
        sections=model['sections'],elastic=model['elastic'],steps=model['steps'],
        requested_modes=int(model['steps'][0]['buckle_data'][0]) if model['steps'] and 'buckle_data' in model['steps'][0] else None,
        general_contact_defined='CONTACT' in model['keywords'],
        contact_state='UNKNOWN_REQUIRES_BASE_STATE_EVIDENCE',
        rotation_output_status='REQUIRES_ODB_FIELD_INSPECTION',scientifically_eligible=False,
        default_classifier_activation=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inp',required=True);p.add_argument('--output-dir',required=True)
    args=p.parse_args();model=read_input(args.inp);c,keys,meta=compile_initial_constraints(model)
    from scipy.sparse import save_npz
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    if any((root/name).exists() for name in ('input_audit.json','initial_constraints.npz','initial_dof_map.npz')):
        raise ValueError('Use a new output directory; historical outputs are not overwritten')
    report=input_summary(model);report['constraints']={k:v for k,v in meta.items() if k!='node_keys'}
    report['constraints'].update(shape=list(c.shape),nnz=c.nnz,storage_bytes=c.data.nbytes+c.indices.nbytes+c.indptr.nbytes)
    with tempfile.TemporaryDirectory(prefix='inp-stage-',dir=root) as stage:
        save_npz(os.path.join(stage,'initial_constraints.npz'),c)
        node_keys=meta['node_keys']
        np.savez_compressed(os.path.join(stage,'initial_dof_map.npz'),
            instances=np.array([k[0] for k in keys]),labels=[k[1] for k in keys],dofs=[k[2] for k in keys],
            node_instances=[k[0] for k in node_keys],node_labels=[k[1] for k in node_keys],
            node_coordinates=np.array([model['nodes'][k] for k in node_keys]),
            metadata=json.dumps({k:v for k,v in meta.items() if k!='node_keys'},sort_keys=True))
        report['artifact_sha256']={name:hashlib.sha256(Path(stage,name).read_bytes()).hexdigest()
                                  for name in ('initial_constraints.npz','initial_dof_map.npz')}
        Path(stage,'input_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        for name in ('initial_constraints.npz','initial_dof_map.npz','input_audit.json'):
            os.replace(Path(stage,name),root/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('constraints','sections','elastic','steps','artifact_sha256')}))


if __name__=='__main__':main()
