"""Cohort-standardised care need and observed service-preserving light profiles."""
from config import *
import pandas as pd,numpy as np
from scipy.stats import spearmanr
composition=pd.read_csv(SOURCE/'cohort_standardised_care_need.csv').sort_values('exposure_site_id')
f=np.load(RUN/'analysis/independence_expectancy_draws.npz');bt=f['profile_rewards'];specs=f['specs']
e=pd.read_csv(RUN/'results/independence_expectancies.csv').query('age==60 and initial_state==0 and metric=="unfavourable"').sort_values(['female','rural','low_education'])
care=e.estimate.to_numpy();rng=np.random.default_rng(SEED);rows=[];boot={}
for r in composition.itertuples(index=False):
    site=r.exposure_site_id
    n=int(r.participants)
    weights=np.array([getattr(r,f'profile_{j}_share') for j in range(8)])
    counts=np.rint(weights*n).astype(int)
    assert counts.sum()==n and np.allclose(counts/n,weights,atol=1e-12)
    weights=counts/n
    w=rng.multinomial(n,weights,size=len(bt))/n
    yy=(w*bt[:,:,1]).sum(axis=1);boot[site]=yy
    rows.append(dict(exposure_site_id=site,participants=n,female_share=weights[[4,5,6,7]].sum(),rural_share=weights[[2,3,6,7]].sum(),primary_or_less_share=weights[[1,3,5,7]].sum(),care_need_years=weights@care,care_need_lower=np.quantile(yy,.025),care_need_upper=np.quantile(yy,.975),**{f'profile_{j}_share':weights[j] for j in range(8)}))
g=pd.DataFrame(rows);g.to_csv(RUN/'results/cohort_standardised_care_need.csv',index=False)
u=pd.read_csv(SOURCE/'city_year_light_services.csv')
x=u.merge(g,on='exposure_site_id',validate='many_to_one').dropna(subset=['rad_auc_nw_km2','ambient_density_per_km2','gdp_per_registered_resident',*SERVICES]);x=x[(x[SERVICES+['ambient_density_per_km2','gdp_per_registered_resident']]>0).all(axis=1)].copy()
reference=u[SERVICES].median();x['relative_capacity']=np.exp(np.log(x[SERVICES].to_numpy()/reference.to_numpy()).mean(axis=1))
x['need_capacity_ratio']=x.care_need_years/x.relative_capacity
x['need_priority_rank']=x.need_capacity_ratio.rank(pct=True,method='average')
x['brightness_priority_rank']=(-x.rad_p90).rank(pct=True,method='average')
rows=[];pairs=[];NEED_FACTOR=1.05
for _,t in x.iterrows():
    ctx=x[x.year.eq(t.year)&x.city_key.ne(t.city_key)&(np.abs(np.log(x.ambient_density_per_km2/t.ambient_density_per_km2))<=np.log(2))&(np.abs(np.log(x.gdp_per_registered_resident/t.gdp_per_registered_resident))<=np.log(2))]
    peers=ctx[(ctx[SERVICES].to_numpy()>=t[SERVICES].to_numpy(float)).all(axis=1)&ctx.rad_auc_nw_km2.lt(t.rad_auc_nw_km2*(1-1e-8))]
    comparable=peers[np.abs(np.log(peers.care_need_years/t.care_need_years))<=np.log(NEED_FACTOR)]
    best=comparable.sort_values('rad_auc_nw_km2').iloc[0] if len(comparable) else None
    rows.append(dict(exposure_site_id=t.exposure_site_id,city_key=t.city_key,year=t.year,unrestricted_peer_count=len(peers),need_matched_peer_count=len(comparable),peer_count_10pct=int((np.abs(np.log(peers.care_need_years/t.care_need_years))<=np.log(1.10)).sum()),peer_count_25pct=int((np.abs(np.log(peers.care_need_years/t.care_need_years))<=np.log(1.25)).sum()),peer_city=best.city_key if best is not None else '',gap_percent=100*(1-best.rad_auc_nw_km2/t.rad_auc_nw_km2) if best is not None else 0.,care_need_years=t.care_need_years,need_capacity_ratio=t.need_capacity_ratio,need_priority_rank=t.need_priority_rank,brightness_priority_rank=t.brightness_priority_rank))
    for _,p in comparable.iterrows():
        need_similarity=np.mean(np.abs(np.log(boot[p.exposure_site_id]/boot[t.exposure_site_id]))<=np.log(NEED_FACTOR))
        pairs.append(dict(target_site=t.exposure_site_id,peer_site=p.exposure_site_id,target_city=t.city_key,peer_city=p.city_key,year=t.year,need_ratio=p.care_need_years/t.care_need_years,need_comparability_probability=need_similarity,gap_percent=100*(1-p.rad_auc_nw_km2/t.rad_auc_nw_km2),**{s+'_ratio':p[s]/t[s] for s in SERVICES}))
z=pd.DataFrame(rows);p=pd.DataFrame(pairs);z.to_csv(RUN/'results/need_matched_benchmarks.csv',index=False);p.to_csv(RUN/'results/need_matched_pairs.csv',index=False);x.to_csv(RUN/'results/need_allocation_city_years.csv',index=False)
top=z.need_priority_rank.gt(.75);dim=z.brightness_priority_rank.gt(.75)
clusters=z.city_key.unique();bb=[]
for i in range(2000):
    cities=rng.choice(clusters,len(clusters),replace=True);ix=np.concatenate([np.flatnonzero(z.city_key.eq(c)) for c in cities]);b=z.iloc[ix]
    bb.append(dict(peer_fraction=b.need_matched_peer_count.gt(0).mean(),eligible_gap=b.loc[b.need_matched_peer_count.gt(0),'gap_percent'].median()))
bs=pd.DataFrame(bb);bs.to_csv(RUN/'results/need_benchmark_bootstrap.csv',index=False)
summary=dict(n_site_years=len(z),n_sites=x.exposure_site_id.nunique(),n_cities=x.city_key.nunique(),participants=g.participants.sum(),minimum_site_participants=10,need_year_range=[g.care_need_years.min(),g.care_need_years.max()],priority_correlation=spearmanr(z.need_priority_rank,z.brightness_priority_rank).statistic,need_priority_top_quartile=int(top.sum()),also_dim_quartile=int((top&dim).sum()),priority_top_overlap_fraction=float((top&dim).sum()/top.sum()),with_need_matched_peer=int(z.need_matched_peer_count.gt(0).sum()),need_peer_fraction=float(z.need_matched_peer_count.gt(0).mean()),need_peer_fraction_ci=bs.peer_fraction.quantile([.025,.975]).to_list(),gap_median_among_eligible=float(z.loc[z.need_matched_peer_count.gt(0),'gap_percent'].median()),gap_ci=bs.eligible_gap.quantile([.025,.975]).to_list(),care_comparability_factor=NEED_FACTOR,with_peer_10pct=int(z.peer_count_10pct.gt(0).sum()),with_peer_25pct=int(z.peer_count_25pct.gt(0).sum()),all_pairs=len(p),stable_pairs=int(p.need_comparability_probability.ge(.8).sum()),reference_capacity_medians=reference.to_dict(),estimand='At age 60, initially independent, common city context and site-specific observed cohort social composition; 30-year restricted dependency years')
save_json(RUN/'results/need_allocation_summary.json',summary);print((RUN/'results/need_allocation_summary.json').read_text(encoding='utf-8'),flush=True)

# Re-evaluate every otherwise qualifying pair in every coefficient/composition
# draw, including pairs outside the point-estimate need caliper.
xx=x.reset_index(drop=True);draw_gap=np.zeros((len(bt),len(xx)));all_candidates=[]
for i,t in xx.iterrows():
 ctx=xx[xx.year.eq(t.year)&xx.city_key.ne(t.city_key)&(np.abs(np.log(xx.ambient_density_per_km2/t.ambient_density_per_km2))<=np.log(2))&(np.abs(np.log(xx.gdp_per_registered_resident/t.gdp_per_registered_resident))<=np.log(2))]
 peers=ctx[(ctx[SERVICES].to_numpy()>=t[SERVICES].to_numpy(float)).all(axis=1)&ctx.rad_auc_nw_km2.lt(t.rad_auc_nw_km2*(1-1e-8))]
 for _,p in peers.iterrows():
  ok=np.abs(np.log(boot[p.exposure_site_id]/boot[t.exposure_site_id]))<=np.log(NEED_FACTOR)
  gap=100*(1-p.rad_auc_nw_km2/t.rad_auc_nw_km2);draw_gap[ok,i]=np.maximum(draw_gap[ok,i],gap)
  all_candidates.append(dict(target_site=t.exposure_site_id,peer_site=p.exposure_site_id,year=t.year,gap_percent=gap,need_ratio=p.care_need_years/t.care_need_years,probability=ok.mean()))
ds=pd.DataFrame({'draw':np.arange(len(bt))+1,'eligible_targets':(draw_gap>0).sum(axis=1),'eligible_fraction':(draw_gap>0).mean(axis=1),'median_gap_among_eligible':np.nanmedian(np.where(draw_gap>0,draw_gap,np.nan),axis=1)})
ds.to_csv(RUN/'results/need_uncertainty_propagation.csv',index=False)
pd.DataFrame(all_candidates).to_csv(RUN/'results/need_all_candidate_pairs.csv',index=False)
xx['capacity_priority_rank']=(1/xx.relative_capacity).rank(pct=True)
xx['priority_change_percentile']=(xx.need_priority_rank-xx.capacity_priority_rank)*100
cols=['exposure_site_id','city_key','year','care_need_years','relative_capacity','need_priority_rank','capacity_priority_rank','priority_change_percentile']
xx[cols].to_csv(RUN/'results/need_capacity_comparison.csv',index=False)
summary['capacity_priority_correlation']=spearmanr(xx.need_priority_rank,xx.capacity_priority_rank).statistic
summary['top_quartile_overlap_with_capacity']=int((xx.need_priority_rank.gt(.75)&xx.capacity_priority_rank.gt(.75)).sum())
summary['maximum_absolute_rank_change_percentile']=xx.priority_change_percentile.abs().max()
summary['without_need_matching_targets']=int(z.unrestricted_peer_count.gt(0).sum())
summary['without_need_matching_pair_count']=len(all_candidates)
summary['need_propagation']={c:ds[c].quantile([.025,.5,.975]).to_list() for c in ds.columns if c!='draw'}
save_json(RUN/'results/need_allocation_summary.json',summary);print('PROPAGATION',summary['need_propagation']);print('CAPACITY',summary['capacity_priority_correlation'],summary['top_quartile_overlap_with_capacity'],summary['maximum_absolute_rank_change_percentile'])
