"""Centered CVaR cuts for numerically difficult long-horizon problems."""
import json,warnings,logging
from itertools import combinations
import numpy as np
from scipy.optimize import OptimizeWarning
from ._solver import linprog
from . import sparse_reference as s

def recover(p,k,cap):
    n=len(p['ids']);ext=[j for j in range(n) if j!=p['si']]
    ratio=p['d']/p['t'][:,None]-1
    up=ratio/.05;down=-ratio/(1-1/1.05)
    mass=.2*len(up);whole=int(np.floor(mass));fraction=mass-whole
    cuts=[];cut_keys=set();trace=[]
    base_a=-p['services'].T;base_b=-p['target_services']
    # Normalize service rows only for conditioning; reconstruct in original units.
    scale=np.maximum(1,np.max(np.abs(base_a),axis=1));base_a=base_a/scale[:,None];base_b=base_b/scale
    def value_cut(w):
        a=up@w;b=down@w;values=np.maximum(a,b)
        idx=np.argsort(values)[::-1];weights=np.ones(whole)/mass
        use=idx[:whole]
        if fraction>1e-12:use=np.r_[use,idx[whole]];weights=np.r_[weights,fraction/mass]
        gradient=np.where((a[use]>=b[use])[:,None],up[use],down[use])
        cut=weights@gradient
        return float(weights@values[use]),cut
    def solve(cost,subset,extra,rhs):
        bounds=[(0,1) if j==p['si'] or j in subset else (0,0) for j in range(n)];bounds[p['si']]=(0,cap)
        for iteration in range(500):
            aa=np.vstack([base_a,*[x[None,:] for x in cuts],*[x[None,:] for x in extra]])
            bb=np.r_[base_b,np.ones(len(cuts)),rhs]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',OptimizeWarning)
                result=linprog(cost,A_ub=aa,b_ub=bb,A_eq=np.ones((1,n)),b_eq=[1],bounds=bounds,method='highs-ds',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'threads':1})
            if not result.success:
                if result.status==2:return result
                raise RuntimeError(result.message)
            value,cut=value_cut(result.x)
            if value<=1+5e-10:
                trace.append({'subset':list(subset),'cuts':len(cuts),'iterations':iteration+1,'objective':float(result.fun),'risk':value})
                return result
            key=tuple(np.round(cut,13))
            if key in cut_keys:raise RuntimeError(('Repeated separation cut',value,result.x))
            cut_keys.add(key);cuts.append(cut)
        raise RuntimeError('Separation did not converge')
    sets=list(combinations(ext,min(k,len(ext)))) if ext else [()]
    primary=p['radiance']/p['target_radiance'];extra=[];rhs=[]
    objectives=[primary,p['distance']]
    for j in sorted(range(n),key=lambda j:p['ids'][j])[:-1]:
        c=np.zeros(n);c[j]=-1;objectives.append(c)
    stages=[];candidates=sets
    for stage,cost in enumerate(objectives):
        solved=[(subset,solve(cost,subset,extra,rhs)) for subset in candidates]
        solved=[(subset,r) for subset,r in solved if r.success]
        if not solved:raise RuntimeError(('Empty lexicographic stage',stage,len(cuts),stages))
        best=min(float(r.fun) for _,r in solved)
        solved=[(subset,r) for subset,r in solved if r.fun<=best+s.EPS]
        candidates=[subset for subset,_ in solved];extra.append(cost);rhs.append(best+s.EPS)
        stages.append({'stage':stage,'best':best,'subsets_remaining':len(candidates),'cuts':len(cuts)})
        logging.getLogger(__name__).debug('Recovery stage: %s', stages[-1])
    subset,winner=min(solved,key=lambda x:(float(x[1].fun),tuple(-x[1].x)))
    w=winner.x
    violation=max(abs(w.sum()-1),max(0.,-w.min()),max(0.,w[p['si']]-cap),max(0.,float(np.max(p['target_services']-p['services'].T@w))),max(0.,s.empirical_cvar(s.reciprocal_loss(p['d']@w/p['t']),.8)-1))
    result={'status':'optimal','w':w,'source':'Exact finite CVaR separation over every donor subset','subsets_tested':len(sets),'primary_optimum':stages[0]['best'],'numeric_form':'centered_cvar_cuts','original_unit_maximum_violation':violation,'stages':stages,'cut_count':len(cuts),'trace':trace}
    assert violation<=s.EPS+64*np.finfo(float).eps,result
    return result
