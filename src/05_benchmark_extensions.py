"""Context-caliper and resource-floor analysis on observed city configurations."""
from config import *
import pandas as pd,numpy as np
from scipy.optimize import linprog
u=pd.read_csv(SOURCE/'city_year_light_services.csv')
req=['rad_auc_nw_km2','ambient_density_per_km2','gdp_per_registered_resident',*SERVICES]
x=u.dropna(subset=req).copy();x=x[(x[req[1:]]>0).all(axis=1)&x[req[0]].ge(0)].reset_index(drop=True)
rows=[]
for caliper in [1.5,2.,3.]:
 for idx,t in x.iterrows():
  other=x[x.year.eq(t.year)&x.city_key.ne(t.city_key)&(np.abs(np.log(x.ambient_density_per_km2/t.ambient_density_per_km2))<=np.log(caliper))&(np.abs(np.log(x.gdp_per_registered_resident/t.gdp_per_registered_resident))<=np.log(caliper))]
  eligible=other[(other[SERVICES].to_numpy()>=t[SERVICES].to_numpy(float)).all(axis=1)]
  lower=eligible[eligible.rad_auc_nw_km2<t.rad_auc_nw_km2*(1-1e-8)]
  best=lower.sort_values('rad_auc_nw_km2').iloc[0] if len(lower) else None
  peers=pd.concat([other,x.loc[[idx]]]);A=-peers[SERVICES].to_numpy().T/t[SERVICES].to_numpy(float)[:,None]
  fit=linprog(peers.rad_auc_nw_km2.to_numpy()/max(1.,t.rad_auc_nw_km2),A_ub=A,b_ub=-np.ones(3),A_eq=np.ones((1,len(peers))),b_eq=[1],bounds=(0,None),method='highs')
  assert fit.success
  rr=fit.x@peers.rad_auc_nw_km2.to_numpy();ss=fit.x@peers[SERVICES].to_numpy()
  assert np.min(ss/t[SERVICES].to_numpy(float))>=1-1e-7
  rows.append(dict(caliper=caliper,exposure_site_id=t.exposure_site_id,city_key=t.city_key,year=t.year,context_peers=len(other),observed_peer_count=len(lower),observed_gap=100*(1-best.rad_auc_nw_km2/t.rad_auc_nw_km2) if best is not None else 0.,convex_gap=max(0.,100*(1-rr/t.rad_auc_nw_km2)) if t.rad_auc_nw_km2>0 else 0.,peer_city=best.city_key if best is not None else '',**{k:t[k] for k in req}))
z=pd.DataFrame(rows);z.to_csv(RUN/'results/benchmark_calipers.csv',index=False)
summary=z.groupby('caliper').apply(lambda d:pd.Series(dict(n=len(d),observed_n=d.observed_peer_count.gt(0).sum(),observed_fraction=d.observed_peer_count.gt(0).mean(),observed_median=d.loc[d.observed_peer_count.gt(0),'observed_gap'].median(),convex_median=d.convex_gap.median(),zero_convex=d.convex_gap.lt(1e-5).sum())),include_groups=False).reset_index()
summary.to_csv(RUN/'results/benchmark_caliper_summary.csv',index=False)
z=z[z.caliper.eq(2)].copy();z['configuration']=np.where(z.convex_gap.lt(1e-5),'On observed convex frontier','Below observed frontier')
comp=z.groupby('configuration')[req+['context_peers']].agg(['median','min','max']);comp.to_csv(RUN/'results/zero_frontier_composition.csv')
print(summary.to_string(index=False));print(comp.to_string())
