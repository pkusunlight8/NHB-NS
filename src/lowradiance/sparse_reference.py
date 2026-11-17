"""Exact finite-support enumeration; primary/context/donor-ID tie order."""
import itertools,json,logging,warnings
from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning
from ._solver import linprog
from .reference import reciprocal_loss,empirical_cvar
EPS=1e-8

def lp(c,a,b,e,f,bounds,method='highs',retry_infeasible=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',OptimizeWarning)
        r=linprog(c,A_ub=a,b_ub=b,A_eq=e,b_eq=f,bounds=bounds,method=method,options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'threads':1})
    if not r.success and (r.status!=2 or retry_infeasible):
        attempts=[dict(method=method,presolve=True,tolerance=1e-9,status=int(r.status),message=str(r.message))]
        for alternative,presolve,tolerance in [('highs-ds',False,1e-9),('highs-ipm',False,1e-9),('highs-ipm',True,1e-10)]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',OptimizeWarning)
                retry=linprog(c,A_ub=a,b_ub=b,A_eq=e,b_eq=f,bounds=bounds,method=alternative,options={'primal_feasibility_tolerance':tolerance,'dual_feasibility_tolerance':tolerance,'presolve':presolve,'threads':1})
            attempts.append(dict(method=alternative,presolve=presolve,tolerance=tolerance,status=int(retry.status),message=str(retry.message)))
            r=retry
            if retry.success or (retry.status==2 and not retry_infeasible):break
        logging.getLogger(__name__).debug('LP retries: %s', attempts)
    if not r.success and r.status!=2:raise RuntimeError(f'LP status {r.status}: {r.message}')
    return r


def matrices(p,arm,cap):
    n=len(p['ids']);need=arm!='A0';B=len(p['t']) if need else 0;expanded=p.get('expanded_epigraph',False);m=n+((2*B+1 if expanded else B+1) if need else 0)
    a=[np.pad(-p['services'].T,((0,0),(0,m-n)))];b=[-p['target_services']]
    e=[np.r_[np.ones(n),np.zeros(m-n)]];f=[1.]
    if p['task']=='short_24_month':
        mass=np.zeros(m);mass[:n]=1;mass[p['si']]=0;e.append(mass);f.append(.5)
        rr=np.r_[p['radiance']/p['target_radiance'],np.zeros(m-n)]
        if p['band']==0:e.append(rr);f.append(.8)
        else:a.extend([rr[None,:],-rr[None,:]]);b.extend([np.array([.8+p['band']]),np.array([-.8+p['band']])])
    risk=np.zeros(m)
    if need:
        ratio=p['d']/p['t'][:,None];low=1-1/1.05
        z=sparse.eye(B,format='csr');t=-np.ones((B,1))
        if expanded:
            zero=sparse.csr_matrix((B,B+1))
            a.extend([sparse.hstack([ratio/.05,-z,zero],format='csr'),sparse.hstack([-ratio/low,-z,zero],format='csr')]);b.extend([np.full(B,20.),np.full(B,-1/low)])
            a.append(sparse.hstack([sparse.csr_matrix((B,n)),z,t,-z],format='csr'));b.append(np.zeros(B))
            risk[n+B]=1.;risk[n+B+1:]=1/(.2*B)
        else:
            a.extend([sparse.hstack([ratio/.05,t,-z],format='csr'),sparse.hstack([-ratio/low,t,-z],format='csr')]);b.extend([np.full(B,20.),np.full(B,-1/low)])
            risk[n]=1.;risk[n+1:]=1/(.2*B)
        if p['task']=='long_30_year':a.append(risk[None,:]);b.append(np.array([1.]))
    a=sparse.vstack([sparse.csr_matrix(block) for block in a],format='csr');b=np.concatenate(b);e=sparse.csr_matrix(np.stack(e));f=np.array(f)
    primary=risk.copy();context=np.r_[p['distance'],np.zeros(m-n)]
    if p['task']=='long_30_year':primary[:]=0;primary[:n]=p['radiance']/p['target_radiance']
    elif arm=='A0':primary=context.copy()
    bounds=[(0.,1.)]*n+[(0.,None)]*(m-n);bounds[p['si']]=(0.,cap)
    return a,b,e,f,bounds,primary,context


def exact_sparse(p,arm,k,cap,baseline=None):
    n=len(p['ids']);ext=[i for i in range(n) if i!=p['si']]
    a,b,e,f,bounds,primary,context=matrices(p,arm,cap)
    # A globally optimal unrestricted solution inside the sparse subset remains
    # globally optimal under that restriction, preserving original tie choices.
    if baseline is not None and sum(baseline[i]>EPS for i in ext)<=k:
        active=[i for i in ext if baseline[i]>0]
        if len(active)<=k and np.min(baseline)>=0:
            return dict(status='optimal',w=baseline,source='Feasible unrestricted optimum retained',subsets_tested=0,primary_optimum=None)
    sets=list(combinations(ext,min(k,len(ext)))) if ext else [()]
    candidates=[];infeasible_verified=0
    for subset in sets:
        bb=list(bounds)
        for j in ext:
            if j not in subset:bb[j]=(0.,0.)
        r=lp(primary,a,b,e,f,bb)
        if r.success:candidates.append((bb,r))
        else:
            check=lp(primary,a,b,e,f,bb,'highs-ipm')
            if check.success:raise AssertionError('Infeasibility disagrees between simplex and IPM')
            infeasible_verified+=1
    if not candidates:return dict(status='infeasible',source='All donor subsets independently infeasible',subsets_tested=len(sets),infeasible_verified=infeasible_verified)
    best=min(float(r.fun) for _,r in candidates);primary_best=best
    candidates=[(bb,r) for bb,r in candidates if r.fun<=best+EPS]
    extra=[primary.copy()];rhs=[best+EPS]
    objectives=[context]
    for j in sorted(range(n),key=lambda x:p['ids'][x])[:-1]:
        c=np.zeros(len(primary));c[j]=-1;objectives.append(c)
    for c in objectives:
        aa=sparse.vstack([a,sparse.csr_matrix(np.stack(extra))],format='csr');bbb=np.r_[b,rhs]
        solved=[]
        for bb,_ in candidates:
            r=lp(c,aa,bbb,e,f,bb)
            if r.success:solved.append((bb,r))
        if not solved:
            for bb,_ in candidates:
                r=lp(c,aa,bbb,e,f,bb,'highs-ipm',retry_infeasible=True)
                if r.success:solved.append((bb,r))
        if not solved:
            if not p.get('expanded_epigraph',False):
                logging.getLogger(__name__).debug('Retry with expanded scenario losses')
                return exact_sparse({**p,'expanded_epigraph':True},arm,k,cap,baseline)
            if p['task']=='long_30_year':
                from .sparse_recovery import recover
                return recover(p,k,cap)
            raise AssertionError('Global lexicographic feasible union disappeared in both equivalent epigraphs')
        best=min(float(r.fun) for _,r in solved);candidates=[(bb,r) for bb,r in solved if r.fun<=best+EPS]
        extra.append(c);rhs.append(best+EPS)
    winner=min(candidates,key=lambda br:(float(br[1].fun),tuple(-br[1].x[:n])))
    return dict(status='optimal',w=winner[1].x[:n],source='Exhaustive support union optimization',subsets_tested=len(sets),infeasible_verified=infeasible_verified,primary_optimum=primary_best,numeric_form='expanded_scenario_losses' if p.get('expanded_epigraph',False) else 'compact_excess_losses')


def describe(p,arm,k,cap,result,baseline):
    row={key:p[key] for key in ['task','pool','target_site','year','outer_fold']};row.update(arm=arm,k=k,self_cap=cap,status=result['status'],candidate_count=len(p['ids'])-1,selection_source=result.get('source'),numeric_form=result.get('numeric_form','unspecified'),subsets_tested=result.get('subsets_tested',0),infeasible_subsets_rechecked=result.get('infeasible_verified',0),gap_band=p.get('band',np.nan))
    weights=[]
    if result['status']!='optimal':return row,weights
    w=np.asarray(result['w']);ext=w.copy();ext[p['si']]=0;mass=ext.sum();active=int((ext>EPS).sum());gap=1-p['radiance']@w/p['target_radiance']
    violation=max(abs(w.sum()-1),max(0.,-w.min()),max(0.,w[p['si']]-cap),max(0.,float(np.max(p['target_services']-p['services'].T@w))))
    if p['task']=='short_24_month':violation=max(violation,abs(mass-.5),max(0.,abs(gap-.2)-p['band']))
    row.update(self_weight=float(w[p['si']]),external_mass=float(mass),active_external_donors=active,external_prototype=active>0,exact_one_external_prototype=active==1,achieved_gap=gap,minimum_service_ratio=float(np.min((p['services'].T@w)/p['target_services'])),maximum_normalized_external_weight=float(ext.max()/mass) if mass>EPS else np.nan,portfolio_weights_json=json.dumps(dict(zip(p['ids'],w.tolist())),sort_keys=True),primary_optimum=result.get('primary_optimum'))
    if arm!='A0':
        ratio=p['d']@w/p['t'];tail=empirical_cvar(reciprocal_loss(ratio),.8)
        row.update(model_internal_tail_loss=tail,model_internal_match_probability=float(np.mean((ratio>=1/1.05-EPS)&(ratio<=1.05+EPS))),mean_demand_ratio=float(ratio.mean()),point_demand_ratio=float(p['point']@w/p['target_point']))
        if p['task']=='long_30_year':violation=max(violation,max(0.,tail-1))
    if baseline is not None:
        be=baseline.copy();be[p['si']]=0;bm=be.sum();oldset=set(np.flatnonzero(be>EPS));newset=set(np.flatnonzero(ext>EPS))
        row.update(full_portfolio_tv=float(np.abs(w-baseline).sum()/2),external_tv=float(np.abs(ext/mass-be/bm).sum()/2) if min(mass,bm)>EPS else np.nan,donor_retention=len(oldset&newset)/len(oldset) if oldset else np.nan,donor_jaccard=len(oldset&newset)/len(oldset|newset) if oldset|newset else 1.,gap_change=float(-p['radiance']@(w-baseline)/p['target_radiance']))
    row['maximum_constraint_violation']=violation
    assert violation<=EPS+64*np.finfo(float).eps,(row,violation)
    assert active<=k,(p['target_site'],active,k)
    if k==1 and active==0:row['status']='self_only_optimum'
    for site,ww in zip(p['ids'],w):weights.append({key:row[key] for key in ['task','pool','target_site','year','outer_fold','arm','k','self_cap']}|dict(peer_site=site,weight=float(ww),is_self=site==p['target_site']))
    return row,weights
