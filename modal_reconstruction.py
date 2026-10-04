"""Explicit inverse-map QC in a supplied dimensional raw U/UR norm."""
import numpy as np
from mfsm_model import matrix

class ReconstructionCheck:
    def __init__(self,mapping,reconstruction,metric_diagonal,tolerance=1e-8):
        self.mapping=matrix(mapping);self.inverse=matrix(reconstruction)
        r,n=self.inverse.shape
        if self.mapping.shape!=(n,r):raise ValueError('Reconstruction DOF mismatch')
        if not np.allclose(self.mapping@self.inverse,np.eye(n),rtol=1e-8,atol=1e-10):
            raise ValueError('Reconstruction is not a compatible right inverse')
        diagonal=np.asarray(metric_diagonal,dtype=float)
        if diagonal.shape!=(r,) or not np.all(np.isfinite(diagonal)) or np.any(diagonal<=0):
            raise ValueError('Positive dimensional raw metric required for U/UR')
        if not np.isfinite(tolerance) or not 0<tolerance<1:raise ValueError('Invalid reconstruction tolerance')
        self.weights=np.sqrt(diagonal/diagonal.max());self.tolerance=float(tolerance)
    def evaluate(self,raw,mapped):
        raw=matrix(raw,self.inverse.shape[0]);mapped=matrix(mapped,self.inverse.shape[1])
        if raw.shape[1]!=mapped.shape[1]:raise ValueError('Reconstruction mode mismatch')
        scale=np.maximum(np.max(np.abs(raw),axis=0),1e-250)
        normalized=raw/scale
        error=(normalized-self.inverse@(mapped/scale))*self.weights[:,None]
        denominator=np.linalg.norm(normalized*self.weights[:,None],axis=0)
        errors=np.linalg.norm(error,axis=0)/np.maximum(denominator,1e-250)
        return dict(relative_errors=errors.tolist(),accepted=(errors<=self.tolerance).tolist(),
                    tolerance=self.tolerance,metric='SUPPLIED_DIMENSIONAL_RAW_U_UR_DIAGONAL_NORM')
