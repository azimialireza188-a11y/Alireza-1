"""Finite-connection assembly and exact rigid-constraint reduction kernels.

BEAM mapping is a small-rotation rigid link, not an INP/contact-state extractor.
Actual Abaqus equivalence must be established before scientific activation.
"""
import numpy as np
from scipy.linalg import null_space, eigvalsh
from mfsm_model import OperatorPack, symmetric, matrix, content_hash


def beam_mpc_matrix(master_xyz, slave_xyz):
    x,y,z=np.asarray(slave_xyz,dtype=float)-np.asarray(master_xyz,dtype=float)
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    c=np.zeros((6,12)); c[:3,:3]=-np.eye(3); c[:3,3:6]=skew
    c[:3,6:9]=np.eye(3); c[3:,3:6]=-np.eye(3); c[3:,9:]=np.eye(3)
    if not np.all(np.isfinite(c)): raise ValueError('Finite rigid-link coordinates required')
    return c


def assemble_connections(operators, connection_stiffness, connection_definition):
    k=symmetric(connection_stiffness)
    if k.shape!=operators.system.shape or not connection_definition:
        raise ValueError('Connection shape/source required')
    if eigvalsh(k,subset_by_index=[0,0])[0]<-1e-10*max(np.linalg.norm(k),1e-250):
        raise ValueError('Connection elastic stiffness must be positive semidefinite')
    meta=dict(operators.metadata,connection_definition=connection_definition,
              connection_hash=content_hash({'definition':connection_definition},[k]))
    return OperatorPack(operators.system+k,operators.components,operators.source,meta)


def reduce_constraints(operators, constraint_definition, active_contact):
    c=matrix(constraint_definition)
    if c.shape[1]!=len(operators.system): raise ValueError('Constraint DOF mismatch')
    status=active_contact.get('status')
    if status not in ('inactive','tangent') or not active_contact.get('evidence'):
        raise ValueError('CONTACT_TANGENT_UNAVAILABLE_OR_UNVERIFIED')
    k=operators.system
    if status=='tangent':
        tangent=symmetric(active_contact['matrix'])
        if tangent.shape!=k.shape: raise ValueError('Contact tangent DOF mismatch')
        k=k+tangent
    q=null_space(c) if len(c) else np.eye(len(k))
    if not q.shape[1]: raise ValueError('Constraints eliminate every DOF')
    meta=dict(operators.metadata,constraint_hash=content_hash({},[c]),
              contact_status=status,contact_evidence=active_contact['evidence'],
              contact_hash=content_hash({'status':status},[k-operators.system]),
              constraint_residual=float(np.linalg.norm(c@q)),
              constraint_kernel_verified=True,abaqus_constraint_mapping_verified=False)
    return OperatorPack(q.T@k@q,{name:q.T@v@q for name,v in operators.components.items()},
                        operators.source,meta),q
