"""Assemble strain-component operators from explicitly supplied strain maps.

This is an integration kernel, not an automatic Abaqus S4R stiffness extractor.
Strain maps must come from a documented, independently verified interpolation.
"""
import numpy as np
from mfsm_model import OperatorPack, SourceDefinition, matrix


def assemble_operators(section, strain_maps, material, quadrature_weights):
    if material.get('nu_class')!=0:
        raise ValueError('Pure-mode auxiliary operators require nu_class=0')
    young=float(material['E']); t=float(material['thickness'])
    if young<=0 or t<=0 or not np.isfinite(young+t):
        raise ValueError('Positive finite auxiliary material required')
    maps={k:matrix(v) for k,v in strain_maps.items()}
    weights=np.asarray(quadrature_weights,dtype=float)
    if weights.ndim!=1 or not np.all(np.isfinite(weights)) or np.any(weights<=0):
        raise ValueError('Positive finite quadrature weights required')
    factors={'eps_x':young*t,'eps_y':young*t,'gamma_xy':young*t/2,
             'kappa_x':young*t**3/12,'kappa_y':young*t**3/12,'kappa_xy':young*t**3/24}
    parts={}
    for name,b in maps.items():
        if name not in factors or b.shape[0]!=len(weights):
            raise ValueError('Unsupported strain component/quadrature shape')
        parts[name]=b.T@((weights*factors[name])[:,None]*b)
    return OperatorPack(sum(parts.values()),parts,
        SourceDefinition('Khezri-Rasmussen 2019/2023','2019 (94)-(97); 2023 (33)-(35)',
                         'auxiliary_nu_zero'),
        {'section_id':section.definition_id,'material':dict(material),
         'operator_origin':'SUPPLIED_STRAIN_MAPS','s4r_equivalence_verified':False})
