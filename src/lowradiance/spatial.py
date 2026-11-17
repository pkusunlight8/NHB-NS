"""Array adapter for the fixed within-city matching algorithm.

Feature extraction supplies the aligned cell inputs.
Definition years: 2012--2021; independent optical verification: 2022--2025.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .spatial_statistics import median, slope

COVARIATES = ['log_population','relative_baseline_slope','baseline_valid_area',
              'log_hospital_minutes','log_elder_access2025','log_bus_distance',
              'tcc_baseline2012_2021','t2m_baseline2012_2021',
              'tp_baseline2012_2021','sde_baseline2012_2021']


def match_cells(features, annual_radiance, annual_coverage, radius_m=25000, caliper_sd=.5):
    """Match one city's fixed cells; return input row indices and denominators.

    Annual arrays must be years x cells, ordered 2012--2025. Matching uses only
    2012--2021 radiance; service covariates retain their stated source years.
    The control set is persistent above the within-city annual median.
    """
    d=features.reset_index(drop=True).copy()
    y=np.asarray(annual_radiance,float);coverage=np.asarray(annual_coverage,float)
    if y.shape!=(14,len(d)) or coverage.shape!=y.shape:
        raise ValueError('Expected aligned (14, n_cells) arrays for 2012--2025.')
    if caliper_sd not in (.25,.5) or radius_m not in (10000,25000):
        raise ValueError('Use a prespecified caliper (.25/.5) and radius (10000/25000).')
    d['baseline_valid_years']=np.isfinite(y[:10]).sum(axis=0)
    d['baseline_valid_area']=coverage[:10].mean(axis=0)
    d['baseline_radiance_median']=median(y[:10])
    d['relative_baseline_slope']=slope(y[:10])/np.maximum(d.baseline_radiance_median.to_numpy(),.1)
    d['log_population']=np.log1p(d.population_total2020.clip(lower=0))
    d['log_hospital_minutes']=np.log1p(d.hospital_drive_minutes2020.clip(lower=0))
    d['log_elder_access2025']=np.log1p(d.elder_total_beds_per10002025.clip(lower=0))
    d['log_bus_distance']=np.log1p(d.nearest_bus_stop_m2024.clip(lower=0))
    eligible=(d.radius_m.le(radius_m)&d.population65plus2020.ge(1)
              &d.population65plus2020_coverage.ge(.9)&d.population_total2020_coverage.ge(.9)
              &d.baseline_valid_years.ge(8))
    domain=np.flatnonzero(eligible)
    if len(domain):
        q25=np.nanquantile(y[:10,domain],.25,axis=1)
        q50=np.nanquantile(y[:10,domain],.50,axis=1)
        low_fraction=((y[:10]<=q25[:,None])&np.isfinite(y[:10])).sum(axis=0)/np.maximum(d.baseline_valid_years.to_numpy(),1)
        high_fraction=((y[:10]>=q50[:,None])&np.isfinite(y[:10])).sum(axis=0)/np.maximum(d.baseline_valid_years.to_numpy(),1)
    else:
        low_fraction=high_fraction=np.full(len(d),np.nan)
    low_group=eligible&(low_fraction>=.8);control_group=eligible&(high_fraction>=.8)
    ambiguous=low_group&control_group
    low_group&=~ambiguous;control_group&=~ambiguous
    complete=(np.isfinite(d[COVARIATES].to_numpy()).all(axis=1)
              &d.hospital_drive_minutes2020_coverage.ge(.9)
              &d.elder_total_beds_per10002025_coverage.ge(.9))
    low=np.flatnonzero(low_group&complete);control=np.flatnonzero(control_group&complete)
    x=d[COVARIATES].to_numpy(float);reference=np.r_[low,control]
    sd=np.nanstd(x[reference],axis=0,ddof=1) if len(reference)>1 else np.ones(len(COVARIATES))
    sd=np.where(sd>1e-10,sd,1.);z=x/sd
    if len(low) and len(control):
        maximum=cdist(z[low],z[control],metric='chebyshev')
        distance=cdist(z[low],z[control],metric='euclidean')
        costs=np.full((len(low),len(control)+len(low)),1000.)
        costs[:,:len(control)]=np.where(maximum<=caliper_sd,distance,1e6)
        li,ci=linear_sum_assignment(costs);good=ci<len(control)
        good[good]&=costs[li[good],ci[good]]<1000
        il,ic=low[li[good]],control[ci[good]];scores=costs[li[good],ci[good]]
    else:
        il=ic=np.array([],dtype=int);scores=np.array([])
    return {'low_indices':il,'control_indices':ic,'covariate_distance':scores,
            'standardization_sd':sd,'covariates':COVARIATES,
            'eligible_cells':int(eligible.sum()),'persistent_low_cells':int(low_group.sum()),
            'persistent_control_cells':int(control_group.sum()),'ambiguous_cells':int(ambiguous.sum()),
            'low_complete':len(low),'control_complete':len(control),'matched_pairs':len(il)}
