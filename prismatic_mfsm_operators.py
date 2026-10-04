"""Source CPT finite-strip interpolation and exact sparse constraint actions.

Part1 (2019) equations1-19: linear membrane, cubic Hermite bending,
S=sin(m*pi*y/L), (L/mu)S'=cos(m*pi*y/L), engineering twist=-2*w_xy.
Global canonical per-node U,V,W,theta = UX,UZ,UY,URZ for extrusion along Z.
These are auxiliary nu=0 faceted CPT operators, NOT Abaqus S4R stiffness.
No contact assumption, family classifier or scientific activation is implied.
"""
import threading
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,diags
from scipy.sparse.linalg import splu

COMPONENTS=('eps_x','eps_y','gamma_xy','kappa_x','kappa_y','kappa_xy')
PARITIES=('sin','cos','sin','sin')


def _terms(values):
    t=np.asarray(values)
    if t.ndim!=1 or not len(t) or t.dtype.kind not in 'iu' or np.any(t<1) or len(set(t.tolist()))!=len(t):
        raise ValueError('Distinct positive integer harmonics required')
    return t.astype(int)


def strip_operators(xy,edges,terms,length,young,thickness):
    xy=np.asarray(xy,dtype=float);terms=_terms(terms);edges=list(edges)
    if (xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(xy)) or not len(xy)
        or not np.all(np.isfinite([length,young,thickness])) or min(length,young,thickness)<=0):
        raise ValueError('Finite cross section and positive material/length required')
    if not edges or len({tuple(sorted(e)) for e in edges})!=len(edges):raise ValueError('Distinct edges required')
    for a,b in edges:
        if not isinstance(a,(int,np.integer)) or not isinstance(b,(int,np.integer)) or min(a,b)<0 or max(a,b)>=len(xy) or a==b:
            raise ValueError('Invalid strip edge')
    if set(i for e in edges for i in e)!=set(range(len(xy))):raise ValueError('Unconnected section node')
    n=len(xy)*4*len(terms);rows=[];cols=[];data={k:[] for k in COMPONENTS}
    q,w=np.polynomial.legendre.leggauss(4);q=(q+1)/2;w=w/2
    factors=[young*thickness,young*thickness,young*thickness/2,
             young*thickness**3/12,young*thickness**3/12,young*thickness**3/24]
    for h,term in enumerate(terms):
        wave=term*np.pi/length
        for a,b in edges:
            delta=xy[b]-xy[a];width=float(np.linalg.norm(delta))
            if width<=0:raise ValueError('Zero width strip')
            tx,ty=delta/width
            # [u_i,v_i,u_j,v_j] and [w_i,slope_i,w_j,slope_j].
            membrane=np.zeros((4,8));bending=np.zeros((4,8))
            for j in range(2):
                membrane[2*j,[4*j,4*j+2]]=[tx,ty];membrane[2*j+1,4*j+1]=1
                bending[2*j,[4*j,4*j+2]]=[ty,-tx];bending[2*j+1,4*j+3]=-1
            indices=[4*len(xy)*h+4*i+d for i in (a,b) for d in range(4)]
            local=[np.zeros((8,8)) for unused in COMPONENTS]
            for s,weight in zip(q,w):
                linear=np.array([1-s,s]);dlinear=np.array([-1.,1.])/width
                hermite=np.array([1-3*s*s+2*s**3,width*(s-2*s*s+s**3),3*s*s-2*s**3,width*(-s*s+s**3)])
                dhermite=np.array([(-6*s+6*s*s)/width,1-4*s+3*s*s,(6*s-6*s*s)/width,-2*s+3*s*s])
                ddhermite=np.array([(-6+12*s)/width**2,(-4+6*s)/width,(6-12*s)/width**2,(-2+6*s)/width])
                bs=[np.array([dlinear[0],0,dlinear[1],0])@membrane,
                    np.array([0,-wave*linear[0],0,-wave*linear[1]])@membrane,
                    np.array([wave*linear[0],dlinear[0],wave*linear[1],dlinear[1]])@membrane,
                    -ddhermite@bending,wave**2*hermite@bending,-2*wave*dhermite@bending]
                for i,(v,factor) in enumerate(zip(bs,factors)):
                    local[i]+=np.outer(v,v)*(weight*width*length/2*factor)
            rows.extend(np.repeat(indices,8));cols.extend(np.tile(indices,8))
            for name,block in zip(COMPONENTS,local):data[name].extend(block.ravel())
    components={name:coo_matrix((data[name],(rows,cols)),shape=(n,n)).tocsr() for name in COMPONENTS}
    for a in components.values():a.eliminate_zeros()
    return dict(system=sum(components.values()),components=components,
        metadata=dict(source='Khezri-Rasmussen Part1 2019 equations1-19',nu_class=0,
            shear_modulus=young/2,young=young,thickness=thickness,length=length,
            harmonics=terms.tolist(),coordinate_definition='HARMONIC_NODE_UX_UZ_UY_URZ',
            longitudinal_axis='GLOBAL_Z',physical_model='FACETED_CPT_NOT_S4R',
            cross_harmonic_shell_terms='EXACT_SINE_COSINE_ORTHOGONALITY',
            connection_harmonics='NOT_REMOVED_OR_ASSUMED_BLOCK_DIAGONAL',
            s4r_equivalence_verified=False,scientifically_eligible=False))


def extruded_reference(model):
    nodes=model['nodes'];stations=np.unique([v[2] for v in nodes.values()])
    if len(stations)<2 or not np.all(np.isfinite(stations)):raise ValueError('Prismatic stations required')
    section={};grid={};pieces=[];xy=[];station_index={z:i for i,z in enumerate(stations)}
    for key,point in nodes.items():
        cross=(key[0],float(point[0]),float(point[1]))
        if cross not in section:section[cross]=len(section);pieces.append(key[0]);xy.append(point[:2])
        index=(station_index[point[2]],section[cross])
        if index in grid:raise ValueError('Duplicate prismatic grid node')
        grid[index]=key
    if len(grid)!=len(stations)*len(section):raise ValueError('Nonextruded/incomplete section stations')
    reverse={key:index for index,key in grid.items()};intervals={}
    for conn in model['elements'].values():
        indices=[reverse[key] for key in conn];zs=sorted(set(i[0] for i in indices));xs=sorted(set(i[1] for i in indices))
        if len(zs)!=2 or zs[1]!=zs[0]+1 or len(xs)!=2 or set(indices)!={(z,x) for z in zs for x in xs}:
            raise ValueError('Only extruded adjacent strip quads supported')
        a,b=xs
        if pieces[a]!=pieces[b]:raise ValueError('Cross-piece shell edge')
        seen=intervals.setdefault((a,b),set())
        if zs[0] in seen:raise ValueError('Duplicate prismatic shell cell')
        seen.add(zs[0])
    if not intervals or any(z!=set(range(len(stations)-1)) for z in intervals.values()):raise ValueError('Incomplete extrusion connectivity')
    return dict(xy=np.asarray(xy),edges=sorted(intervals),pieces=pieces,stations=stations,
        node_keys=[[grid[z,x] for x in range(len(section))] for z in range(len(stations))],
        length=float(stations[-1]-stations[0]),origin=float(stations[0]),
        main_nodes_reviewed=False,closed_loop_reviewed=False)


class ConstraintProjector:
    """Euclidean admissible action P=I-C.T(CC.T)^-1 C; no dense raw Q.

    Independent nonzero rows required; dependent rows fail explicitly. This
    projects kinematics only and supplies no active-contact or shell metric.
    """
    def __init__(self,constraints):
        c=csr_matrix(constraints,dtype=float)
        if not c.shape[1] or not np.all(np.isfinite(c.data)):raise ValueError('Finite constraints required')
        norms=np.sqrt(np.asarray(c.multiply(c).sum(axis=1)).ravel())
        if np.any(norms==0):raise ValueError('Nonzero independent constraint rows required')
        self.c=diags(1/norms)@c;self.lock=threading.Lock();self.factor=None
        if c.shape[0]:
            try:self.factor=splu((self.c@self.c.T).tocsc())
            except RuntimeError as exc:raise ValueError('Independent constraint rows required') from exc
            if np.min(np.abs(self.factor.U.diagonal()))<=1e-12:
                raise ValueError('Constraint Gram is numerically rank deficient')

    def apply(self,values):
        v=np.asarray(values,dtype=float)
        if v.ndim not in (1,2) or v.shape[0]!=self.c.shape[1] or not np.all(np.isfinite(v)):
            raise ValueError('Finite raw DOF vectors required')
        if self.factor is None:return v.copy()
        with self.lock:correction=self.factor.solve(np.asarray(self.c@v))
        return v-self.c.T@correction


def harmonic_design(stations,terms,length,origin=0.):
    z=np.asarray(stations,dtype=float);terms=_terms(terms)
    if z.ndim!=1 or len(z)<2 or np.any(np.diff(z)<=0) or not np.all(np.isfinite(z)) or not np.isfinite(length+origin) or length<=0:
        raise ValueError('Finite increasing stations and length required')
    if z[0]<origin-1e-10*length or z[-1]>origin+length+1e-10*length:raise ValueError('Stations outside member')
    phase=(z[:,None]-origin)*terms[None,:]*np.pi/length
    sine=np.sin(phase);cosine=np.cos(phase)
    sine[np.isclose(z,origin,rtol=0,atol=1e-12*length)|np.isclose(z,origin+length,rtol=0,atol=1e-12*length)]=0
    weights=np.r_[np.diff(z)[0]/2,(z[2:]-z[:-2])/2,np.diff(z)[-1]/2]
    return dict(sin=sine,cos=cosine,weights=weights)


def reconstruct_canonical(stations,coefficients,terms,length,origin=0.):
    d=harmonic_design(stations,terms,length,origin);a=np.asarray(coefficients,dtype=float)
    if a.ndim!=4 or a.shape[0]!=len(terms) or a.shape[2]!=4 or not np.all(np.isfinite(a)):raise ValueError('Expected harmonic,node,4,mode coefficients')
    out=np.empty((len(stations),a.shape[1],4,a.shape[3]))
    for j,parity in enumerate(PARITIES):out[:,:,j,:]=np.einsum('zh,hnm->znm',d[parity],a[:,:,j,:])
    return out


class CanonicalFitter:
    """One weighted inverse per parity, reused across all nodes and mode batches."""
    def __init__(self,stations,terms,length,origin=0.,rotation_length=1.,include_constant_warping=False):
        self.include_constant_warping=bool(include_constant_warping)
        self.design=harmonic_design(stations,terms,length,origin)
        if not np.isfinite(rotation_length) or rotation_length<=0:raise ValueError('Positive rotation length required')
        self.rotation_length=float(rotation_length);self.terms=_terms(terms)
        if self.include_constant_warping:self.design['cos']=np.column_stack([np.ones(len(stations)),self.design['cos']])
        self.inverse={};self.condition={};self.cache={};self.lock=threading.Lock()
        sqrtw=np.sqrt(self.design['weights'])
        for parity in ('sin','cos'):
            design=self.design[parity]*sqrtw[:,None];u,s,vh=np.linalg.svd(design,full_matrices=False)
            if len(s)!=design.shape[1] or s[-1]<=1e-12*s[0]:raise ValueError('Harmonic basis unresolved/aliased')
            self.inverse[parity]=((vh.T/s)@u.T)*sqrtw[None,:]
            self.condition[parity]=float(s[0]/s[-1])

    def fit(self,fields,xp=np,host_output=True):
        raw=xp.asarray(fields,dtype=xp.float64)
        if raw.ndim!=4 or raw.shape[0]!=len(self.design['weights']) or raw.shape[2]!=4 or not bool(xp.all(xp.isfinite(raw))):
            raise ValueError('Expected station,node,4,mode finite fields')
        constant=xp.zeros((raw.shape[1],raw.shape[3]),dtype=xp.float64)
        y=xp.asarray(raw);a=xp.empty((len(self.terms),raw.shape[1],4,raw.shape[3]),dtype=xp.float64);fitted=xp.empty_like(y)
        key='cpu' if xp is np else ('gpu',int(xp.cuda.runtime.getDevice()))
        with self.lock:
            if key not in self.cache:
                self.cache[key]=({p:xp.asarray(self.inverse[p]) for p in ('sin','cos')},
                    {p:xp.asarray(self.design[p]) for p in ('sin','cos')},xp.asarray(np.sqrt(self.design['weights'])))
            inverse,design,sqrtw=self.cache[key]
        for parity,indices in [('sin',[0,2,3]),('cos',[1])]:
            values=y[:,:,indices,:].reshape(len(self.design['weights']),-1)
            all_coeff=inverse[parity]@values
            coeff=all_coeff
            if parity=='cos' and self.include_constant_warping:
                constant=coeff[0].reshape(raw.shape[1],raw.shape[3]);coeff=coeff[1:]
            a[:,:,indices,:]=coeff.reshape(len(self.terms),raw.shape[1],len(indices),raw.shape[3])
            fitted[:,:,indices,:]=(design[parity]@all_coeff).reshape(raw.shape[0],raw.shape[1],len(indices),raw.shape[3])
        metric=xp.asarray([1.,1.,1.,self.rotation_length])[None,None,:,None]*sqrtw[:,None,None,None]
        scaled=y*metric;error=(y-fitted)*metric
        scale=xp.max(xp.abs(scaled),axis=(0,1,2));scale=xp.where(scale>0,scale,1.)
        den=xp.sum((scaled/scale)**2,axis=(0,1,2));num=xp.sum((error/scale)**2,axis=(0,1,2))
        residual=xp.sqrt(xp.divide(num,den,out=xp.zeros_like(num),where=den>0))
        if host_output and xp is not np:a=xp.asnumpy(a);residual=xp.asnumpy(residual);constant=xp.asnumpy(constant)
        return dict(coefficients=a,constant_warping=constant,relative_residual=residual,condition=dict(self.condition),
            metric='TRAPEZOID_STATION_EQUAL_SECTION_NODE_TRANSLATION_PLUS_ROTATION_LENGTH_SQUARED',
            rotation_length=self.rotation_length,unrepresented_raw_rotations=['URX','URY'],scientifically_eligible=False)


def fit_canonical(stations,fields,terms,length,origin=0.,rotation_length=1.):
    return CanonicalFitter(stations,terms,length,origin,rotation_length).fit(fields)


def transverse_rotation_map(xy,edges,terms,length):
    """Facet CPT rotations, averaged at shared nodes; drill=curl(u,v)/2.

    This additional S4R-to-CPT reconstruction is an explicit approximation at
    facet junctions and for numerical drill DOFs; it requires validation.
    """
    xy=np.asarray(xy,dtype=float);terms=_terms(terms);n=len(xy);degree=np.zeros(n)
    for a,b in edges:degree[a]+=1;degree[b]+=1
    if np.any(degree==0) or length<=0:raise ValueError('Connected strips and positive length required')
    rows=[];cols=[];data=[]
    for h,term in enumerate(terms):
        k=term*np.pi/length
        for a,b in edges:
            delta=xy[b]-xy[a];width=float(np.linalg.norm(delta))
            if width==0:raise ValueError('Zero width strip')
            tx,ty=delta/width
            for i in (a,b):
                for r,c,v in [(0,4*i,k*.5*tx*ty),(0,4*i+2,-k*(tx*tx+.5*ty*ty)),
                              (1,4*i,k*(ty*ty+.5*tx*tx)),(1,4*i+2,-k*.5*tx*ty),
                              (0,4*a+1,-.5*ty/width),(0,4*b+1,.5*ty/width),
                              (1,4*a+1,.5*tx/width),(1,4*b+1,-.5*tx/width)]:
                    rows.append(2*n*h+2*i+r);cols.append(4*n*h+c);data.append(v/degree[i])
    out=coo_matrix((data,(rows,cols)),shape=(2*n*len(terms),4*n*len(terms))).tocsr();out.eliminate_zeros();return out


def reconstruct_six_dofs(xy,edges,stations,coefficients,terms,length,origin=0.):
    canonical=reconstruct_canonical(stations,coefficients,terms,length,origin)
    a=np.asarray(coefficients,dtype=float);n=len(xy)
    if a.shape[1]!=n:raise ValueError('Cross-section coefficient shape differs')
    r=transverse_rotation_map(xy,edges,terms,length)@a.reshape(len(terms)*n*4,-1)
    cosine=harmonic_design(stations,terms,length,origin)['cos']
    transverse=np.einsum('zh,hnrm->znrm',cosine,r.reshape(len(terms),n,2,a.shape[3]))
    raw=np.empty((len(stations),n,6,a.shape[3]));raw[:,:,[0,2,1,5],:]=canonical;raw[:,:,[3,4],:]=transverse
    return raw


def constant_warping_operator(xy,edges,length,young,thickness):
    """Independent v0(x), constant along Y: only gamma_xy=v0,x contributes.

    Explicit search-space extension of the m>=1 series; preserves the uniform
    axial rigid-null field. It is reported separately, never called GA buckling.
    """
    xy=np.asarray(xy,dtype=float);rows=[];cols=[];values=[]
    if min(length,young,thickness)<=0 or not np.all(np.isfinite([length,young,thickness])):raise ValueError('Positive finite material/length required')
    for a,b in edges:
        width=float(np.linalg.norm(xy[b]-xy[a]))
        if width<=0:raise ValueError('Nonzero strip width required')
        block=young*thickness/2*length/width*np.array([[1,-1],[-1,1]])
        rows.extend([a,a,b,b]);cols.extend([a,b,a,b]);values.extend(block.ravel())
    return coo_matrix((values,(rows,cols)),shape=(len(xy),len(xy))).tocsr()
