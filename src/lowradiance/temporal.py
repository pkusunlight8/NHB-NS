"""Fixed-identity external prototypes using the published annual stability rule.

Annual radiance and donor weights are supplied as tables.
"""
import numpy as np
import pandas as pd


def fixed_reference_trajectory(annual, weights, target_id, selection_year=2018):
    """Evaluate one target and fixed weights; retain invalid annual observations.

    `annual`: city_id/year/radiance/valid. `weights`: donor_id/weight.
    The self component is excluded; every strictly positive external weight is
    retained and normalized. Only later years determine the stable flag.
    """
    d=annual.copy();d['city_id']=d.city_id.astype(str)
    if d.duplicated(['city_id','year']).any():
        raise ValueError('Annual input must have unique city-year keys.')
    if not d.valid.isin([True,False,0,1]).all():
        raise ValueError('valid must contain explicit boolean values.')
    if weights.donor_id.astype(str).duplicated().any():
        raise ValueError('Each donor must occur once.')
    w=weights.copy();w['donor_id']=w.donor_id.astype(str)
    w=w[w.donor_id.ne(str(target_id))&w.weight.gt(0)].copy()
    if not len(w):
        return pd.DataFrame(),{'stable':False,'temporal_status':'no_positive_external_reference','validation_valid_years':0}
    if not np.isfinite(w.weight).all():raise ValueError('Nonfinite external weight.')
    w['normalized_weight']=w.weight/w.weight.sum();peers=w.donor_id.tolist()
    pv=w.normalized_weight.to_numpy(float);a=d.set_index(['city_id','year']);rows=[]
    for year in range(2012,2026):
        keys=[(str(target_id),year)]+[(p,year) for p in peers]
        available=all(k in a.index for k in keys)
        sub=a.loc[keys] if available else None
        valid=(available and bool(sub.valid.all()) and np.isfinite(sub.radiance).all()
               and float(sub.iloc[0].radiance)>0)
        row={'year':year,'period':'validation' if year>selection_year else ('selection' if year==selection_year else 'early_description'),'valid':bool(valid)}
        if valid:
            tr=float(sub.iloc[0].radiance);ext=float(pv@sub.iloc[1:].radiance.to_numpy())
            row.update(target_radiance=tr,external_radiance=ext,external_relative_gap=1-ext/tr,external_lower=ext<tr)
        rows.append(row)
    result=pd.DataFrame(rows);late=result[result.period.eq('validation')&result.valid]
    fraction=float(late.external_lower.mean()) if len(late) else np.nan
    summary={'validation_valid_years':len(late),'validation_lower_fraction':fraction,
             'stable':bool(len(late)>=5 and fraction>=.8),'positive_external_donors':len(w),
             'temporal_status':'sufficient_later_years' if len(late)>=5 else 'insufficient_later_years'}
    return result,summary
