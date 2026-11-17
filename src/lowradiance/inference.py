"""City bootstrap conditional on frozen prediction rules."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .allocation import shares, project_share

def batch_project(need_mass,baseline_mass):
    q=np.asarray(need_mass,float).copy();c=np.asarray(baseline_mass,float).copy()
    if q.shape!=c.shape or q.ndim!=2 or not np.isfinite(q).all() or not np.isfinite(c).all() or (q<0).any() or (c<0).any() or (c.sum(axis=1)<=0).any():
        raise ValueError('Invalid city-resampling masses.')
    c/=c.sum(axis=1,keepdims=True)
    total=q.sum(axis=1,keepdims=True)
    q=np.divide(q,total,out=c.copy(),where=total>0)
    if ((c==0)&(q>1e-14)).any():raise ValueError('Predicted mass outside the sampled cohort support.')
    ratio=np.divide(q,c,out=np.zeros_like(q),where=c>0)
    low=np.full(len(c),-2.);high=ratio.max(axis=1)+1
    for _ in range(100):
        middle=(low+high)/2
        a=np.clip(q-middle[:,None]*c,0,2*c)
        above=a.sum(axis=1)>1
        low=np.where(above,middle,low);high=np.where(above,high,middle)
    a=np.clip(q-((low+high)/2)[:,None]*c,0,2*c)
    if np.max(np.abs(a.sum(axis=1)-1))>1e-12:raise ArithmeticError('Resampling projection budget residual.')
    return a


def evaluate_city_table(city_table,score_columns,event_column='event_mass',mass_column='baseline_mass',draws=2000,seed=20261103):
    d=city_table.copy()
    if d.index.duplicated().any():raise ValueError('Each inference unit must be one independent city.')
    mass=d[mass_column].to_numpy(float);events=d[event_column].to_numpy(float)
    c=shares(mass);n=shares(events)
    oracle=float(np.maximum(n-2*c,0).sum())
    point=[];allocations={}
    for rule in score_columns:
        a=project_share(d[rule].to_numpy(float),c);value=float(np.maximum(n-a,0).sum())
        point.append({'rule':rule,'loss':value,'oracle_loss':oracle,'regret':value-oracle,'absolute_mismatch':2*value})
        allocations[rule]=a
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(len(d),np.full(len(d),1/len(d)),size=draws)
    weighted_mass=counts*mass
    weighted_events=counts*events
    defined=weighted_events.sum(axis=1)>0
    use=counts[defined];bm=weighted_mass[defined];be=weighted_events[defined]
    bc=bm/bm.sum(axis=1,keepdims=True);bn=be/be.sum(axis=1,keepdims=True)
    bo=np.maximum(bn-2*bc,0).sum(axis=1)
    bootstrap=pd.DataFrame({'draw_id':np.arange(draws)[defined],'oracle_loss':bo})
    for rule in score_columns:
        a=batch_project(use*d[rule].to_numpy(float),bm)
        bootstrap[rule]=np.maximum(bn-a,0).sum(axis=1)-bo
    return pd.DataFrame(point),bootstrap,{'draws_planned':draws,'draws_defined':int(defined.sum()),'zero_event_draws':int((~defined).sum()),'city_units':len(d),'seed':seed,'conditioned_on_frozen_models':True,'resampled_city_counts':counts},allocations


def paired_summary(point,bootstrap,research='M7_cumulative_cal',comparator='selected_traditional_risk'):
    p=point.set_index('rule');difference=float(p.loc[research,'regret']-p.loc[comparator,'regret'])
    draws=(bootstrap[research]-bootstrap[comparator]).to_numpy(float)
    if not len(draws):
        return {'research':research,'comparator':comparator,'difference':difference,'status':'undefined_resampling_distribution','p':1.}
    interval=np.quantile(draws,[.025,.975]);centered=draws-difference
    pvalue=float((1+np.sum(np.abs(centered)>=abs(difference)-1e-15))/(len(draws)+1))
    standard_error=float(draws.std(ddof=1)) if len(draws)>1 else np.nan
    reference=float(p.loc[comparator,'regret'])
    return {'research':research,'comparator':comparator,'difference':difference,'ci_lower':float(interval[0]),'ci_upper':float(interval[1]),
            'p':pvalue,'standard_error':standard_error,'normal_approximation_mde_80pct':float((norm.ppf(.975)+norm.ppf(.8))*standard_error) if standard_error>0 else None,
            'relative_regret_reduction':-difference/reference if reference>1e-12 else None,
            'research_regret':float(p.loc[research,'regret']),'traditional_regret':reference,'defined_draws':len(draws),
            'interval_method':'city bootstrap percentile interval with fixed fitted models and reprojected common constraints',
            'p_method':'two-sided centered city bootstrap with plus-one correction','status':'estimated'}
