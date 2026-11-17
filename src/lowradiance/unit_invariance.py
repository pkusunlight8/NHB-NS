"""Compose selected policies in dimensionless city-need shares."""
import numpy as np
import pandas as pd
from .allocation import shares,project_share

def correct_predictions(pred,rules,frozen_q=None):
    columns=['baseline_mass',*rules]
    city=pred.groupby('city_key',sort=True)[columns].sum()
    if frozen_q is None:
        q=pd.DataFrame(index=city.index)
        for rule in rules:q[rule]=shares(city[rule]) if city[rule].sum()>0 else shares(city.baseline_mass)
    else:q=frozen_q.loc[city.index,rules].copy()
    decision=pred.groupby('city_key').agg(class_fold=('class_fold','first'),selected_traditional_rule=('selected_traditional_rule','first'))
    decision['candidate_need_share_component']=[float(q.loc[c,decision.loc[c,'selected_traditional_rule']]) for c in decision.index]
    decision['selected_need_share']=shares(decision.candidate_need_share_component)
    decision['allocation_share']=project_share(decision.candidate_need_share_component,city.baseline_mass)
    result=pred.copy();result['selected_traditional_raw_score_initial']=result.selected_traditional_risk
    result['selected_traditional_risk']=0.
    for citykey,row in decision.iterrows():
        mask=result.city_key.eq(citykey);raw=result.loc[mask,row.selected_traditional_rule].to_numpy(float)
        if raw.sum()>0:result.loc[mask,'selected_traditional_risk']=raw/raw.sum()*row.candidate_need_share_component
        else:result.loc[mask,'selected_traditional_risk']=row.candidate_need_share_component*result.loc[mask,'baseline_mass']/result.loc[mask,'baseline_mass'].sum()
    replay=result.groupby('city_key').selected_traditional_risk.sum().reindex(decision.index)
    assert np.max(np.abs(replay-decision.candidate_need_share_component))<1e-13
    for rule in rules:assert np.array_equal(pred[rule].to_numpy(),result[rule].to_numpy())
    return result,decision,city,q
