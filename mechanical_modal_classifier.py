"""Joint system-energy projection with signed cross terms and eigenspace bounds."""
import itertools
import numpy as np
from scipy.linalg import cholesky, svd,solve_triangular
from mfsm_model import matrix, ModeBatch,normalized_columns


class EnergyProjector:
    def __init__(self,basis,tolerance=1e-10):
        self.basis=basis; self.metric=basis.metric
        self.lt=cholesky(self.metric,lower=True).T
        blocks=[];self.slices={};start=0
        for name,space in basis.spaces.items():
            if not space.shape[1]: continue
            u,s,unused=svd(self.lt@normalized_columns(space),full_matrices=False)
            keep=s>tolerance*s[0]
            q=u[:,keep]; self.slices[name]=slice(start,start+q.shape[1])
            blocks.append(q);start+=q.shape[1]
        if not blocks: raise ValueError('No energetic family basis')
        self.b=np.column_stack(blocks)
        u,s,vt=svd(self.b,full_matrices=False)
        if len(s)<self.b.shape[1] or s[-1]<=tolerance*s[0]:
            raise ValueError('Overlapping/unidentifiable family spaces')
        self.condition=float(s[0]/s[-1]);self.inverse=(vt.T/s)@u.T
        self.device_arrays={}

    def project(self,x,xp=np):
        x=matrix(x[:,None] if np.asarray(x).ndim==1 else x,len(self.metric))
        if xp is np:
            lt,b,inverse=self.lt,self.b,self.inverse
            convert=lambda a:np.asarray(a)
        else:
            device=int(xp.cuda.runtime.getDevice())
            if device not in self.device_arrays:
                self.device_arrays[device]=tuple(xp.asarray(a) for a in (self.lt,self.b,self.inverse))
            lt,b,inverse=self.device_arrays[device];convert=xp.asnumpy
        y=lt@xp.asarray(x);total=xp.sum(y*y,axis=0)
        if np.any(convert(total)<=1e-250): raise ValueError('Zero modal energy')
        coef=inverse@y
        components={k:b[:,sl]@coef[sl] for k,sl in self.slices.items()}
        residual=y-sum(components.values());components['R']=residual
        shares={k:convert(100*xp.sum(v*v,axis=0)/total) for k,v in components.items()}
        cross={a+':'+b_:convert(200*xp.sum(components[a]*components[b_],axis=0)/total)
               for a,b_ in itertools.combinations(components,2)}
        closure=np.max(np.abs(sum(shares.values())+sum(cross.values())-100))/100
        return dict(shares_percent=shares,cross_percent=cross,
                    relative_residual=np.sqrt(shares['R']/100),closure_error=float(closure),
                    weighted_components={k:convert(v) for k,v in components.items()},condition=self.condition)


def classify_modes(batch,basis,thresholds=None,policy=None,projector=None,backend=np,global_projector=None):
    thresholds=thresholds or {}; dominance=thresholds.get('dominance',.9)
    max_residual=thresholds.get('max_residual',.05)
    projector=projector or EnergyProjector(basis)
    result=projector.project(batch.vectors,xp=backend)
    rows=[]
    for j,mode in enumerate(batch.ids):
        shares={k:float(v[j]) for k,v in result['shares_percent'].items()}
        cross={k:float(v[j]) for k,v in result['cross_percent'].items()}
        candidates={k:shares.get(k,0.) for k in ('L','D','G')}
        dominant=max(candidates,key=candidates.get);flags=[]
        quality='RESOLVED' if candidates[dominant]>=100*dominance else 'MIXED'
        if quality!='RESOLVED': dominant=None
        if result['relative_residual'][j]>max_residual:
            quality='UNRESOLVED';dominant=None;flags.append('HIGH_MECHANICAL_RESIDUAL')
        if sum(abs(v) for v in cross.values())>5.:
            quality='UNRESOLVED';dominant=None;flags.append('HIGH_SIGNED_CROSS_TERMS')
        if result['closure_error']>1e-8:
            quality='UNRESOLVED';dominant=None;flags.append('ENERGY_CLOSURE_FAILED')
        rows.append(dict(mode=mode,dominant_family={'L':'LOCAL','D':'DISTORTIONAL','G':'GLOBAL'}.get(dominant),
            quality_state=quality,global_subtype=None,shares_percent=shares,cross_percent=cross,
            relative_residual=float(result['relative_residual'][j]),closure_error=result['closure_error'],
            condition=result['condition'],flags=flags,scientifically_eligible=False,
            source_method=basis.metadata.get('source_method','MFSM_SUPPLIED_OPERATOR_SEARCH_SPACES'),metric_definition='AUXILIARY_SYSTEM_ENERGY_NU_ZERO'))
    if global_projector is not None and global_projector.basis.metadata.get('mechanical_definition_review') is True:
        if global_projector.metric.shape!=projector.metric.shape or not np.allclose(global_projector.metric,projector.metric,rtol=1e-12,atol=1e-12):
            raise ValueError('Global subtype metric mismatch')
        if 'G' not in projector.slices:raise ValueError('Global family required for subtype')
        g=projector.b[:,projector.slices['G']]
        if np.linalg.norm(global_projector.b-g@(g.T@global_projector.b))>1e-8*max(np.linalg.norm(global_projector.b),1e-250):
            raise ValueError('Subtype spaces must lie inside the mechanical global family')
        selected=[j for j,r in enumerate(rows) if r['dominant_family']=='GLOBAL']
        if selected:
            global_vectors=solve_triangular(projector.lt,result['weighted_components']['G'][:,selected])
            sub=global_projector.project(global_vectors)
            for k,j in enumerate(selected):
                shares={name:float(values[k]) for name,values in sub['shares_percent'].items()}
                rows[j]['global_shares_percent']=shares
                rows[j]['global_definition_id']=global_projector.basis.definition_id
                if (sub['relative_residual'][k]>max_residual or sub['closure_error']>1e-8 or
                        sum(abs(float(v[k])) for v in sub['cross_percent'].values())>5.):continue
                flex=shares.get('FLEXURAL',0.);torsion=shares.get('TORSIONAL',0.)
                if flex>=100*dominance:rows[j]['global_subtype']='FLEXURAL'
                elif torsion>=100*dominance:rows[j]['global_subtype']='TORSIONAL'
                elif min(flex,torsion)>=100*(1-dominance) and flex+torsion>=100*(1-max_residual**2):
                    rows[j]['global_subtype']='FLEXURAL_TORSIONAL'
    return rows


def classify_cluster(batch,basis,thresholds=None,projector=None,global_projector=None):
    projector=projector or EnergyProjector(basis)
    u,s,unused=svd(normalized_columns(projector.lt@batch.vectors),full_matrices=False)
    if len(s)!=len(batch.ids) or s[-1]<=1e-10*s[0]:
        raise ValueError('Rank-deficient observed eigenspace')
    x=np.linalg.solve(projector.lt,u)
    result=projector.project(x)
    trace={k:float(np.mean(v)) for k,v in result['shares_percent'].items()}
    bounds={k:[float(100*np.linalg.eigvalsh(v.T@v)[0]),float(100*np.linalg.eigvalsh(v.T@v)[-1])]
            for k,v in result['weighted_components'].items()}
    dominance=(thresholds or {}).get('dominance',.9)
    stable=[k for k in ('L','D','G') if bounds.get(k,[0,0])[0]>=100*dominance]
    max_residual=(thresholds or {}).get('max_residual',.05)
    cross_bounds={}
    components=result['weighted_components']
    for a,b in itertools.combinations(components,2):
        operator=100*(components[a].T@components[b]+components[b].T@components[a])
        eigenvalues=np.linalg.eigvalsh(operator)
        cross_bounds[a+':'+b]=[float(eigenvalues[0]),float(eigenvalues[-1])]
    cross_upper=sum(max(abs(lo),abs(hi)) for lo,hi in cross_bounds.values())
    if bounds['R'][1]>100*max_residual**2 or cross_upper>5. or result['closure_error']>1e-8: stable=[]
    subtype=None;global_bounds={}
    if (stable==['G'] and global_projector is not None and
            global_projector.basis.metadata.get('mechanical_definition_review') is True):
        # Same containment/metric checks as the per-mode subtype path.
        classify_modes(ModeBatch(x,batch.ids),basis,thresholds,projector=projector,global_projector=global_projector)
        gx=solve_triangular(projector.lt,result['weighted_components']['G'])
        sub=classify_cluster(ModeBatch(gx,batch.ids),global_projector.basis,thresholds,projector=global_projector)
        global_bounds=sub['bounds_percent']
        if sub['cross_absolute_upper_percent']<=5. and global_bounds['R'][1]<=100*max_residual**2:
            flex=global_bounds.get('FLEXURAL',[0.,0.])[0];torsion=global_bounds.get('TORSIONAL',[0.,0.])[0]
            if flex>=100*dominance:subtype='FLEXURAL'
            elif torsion>=100*dominance:subtype='TORSIONAL'
            elif min(flex,torsion)>=100*(1-dominance):subtype='FLEXURAL_TORSIONAL'
    return dict(modes=list(batch.ids),trace_shares_percent=trace,bounds_percent=bounds,
                global_subtype=subtype,global_bounds_percent=global_bounds,
                stable_family=stable[0] if len(stable)==1 else None,cross_bounds_percent=cross_bounds,
                cross_absolute_upper_percent=cross_upper,
                scientifically_eligible=False,closure_error=result['closure_error'])
