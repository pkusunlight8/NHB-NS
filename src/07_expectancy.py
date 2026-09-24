"""Transition-resolved standardised health expectancy and exact Shapley allocation."""
from config import *
import numpy as np,pandas as pd,math,itertools
from numba import njit

@njit(cache=False)
def reward(beta,profiles,age=60.,years=30.,initial=0):
    # profiles has one covariate vector per directed transition, enabling
    # exact transition-channel replacement in the decomposition.
    v=np.zeros(2);v[initial]=1.;e=np.zeros(2);traj=np.zeros((int(years*12)+1,3));traj[0,:2]=v
    for m in range(int(years*12)):
        eta=np.zeros(4)
        for t in range(4):
            eta[t]=beta[t,0]+beta[t,1]*((age-60)/10+m/120.)
            for j in range(8):eta[t]+=beta[t,j+2]*profiles[t,j]
        z=np.exp(eta);p00=1/(1+z[0]+z[1]);p11=1/(1+z[2]+z[3])
        w=np.array([v[0]*p00+v[1]*z[2]*p11,v[0]*z[0]*p00+v[1]*p11])
        e+=(v+w)/24.;v=w;traj[m+1,:2]=v;traj[m+1,2]=1-v.sum()
    return e,traj

@njit(cache=False)
def many_rewards(draws,profiles,age,years,initial):
    out=np.empty((len(draws),len(profiles),2))
    for b in range(len(draws)):
        for j in range(len(profiles)):out[b,j]=reward(draws[b],profiles[j],age,years,initial)[0]
    return out

def profile(female,rural,low,service=0.,brightness=0.):
    return np.tile(np.array([female,rural,low,service,brightness,0.,0.,0.]),(4,1))

def run(endpoint):
    f=np.load(RUN/f'analysis/imach_{endpoint}/fit.npz');beta=f['beta'];cov=f['covariance']
    rng=np.random.default_rng(SEED);draws=rng.multivariate_normal(beta.ravel(),cov,size=2000).reshape(-1,4,10)
    specs=list(itertools.product([0,1],repeat=3));profiles=np.array([profile(*s) for s in specs])
    rows=[];curves=[];allboot=[]
    for age in [60,65,70,75,80,85]:
      for initial in [0,1]:
        point=many_rewards(beta[None],profiles,float(age),float(90-age),initial)[0]
        bt=many_rewards(draws,profiles,float(age),float(90-age),initial)
        for j,(sex,rural,low) in enumerate(specs):
          for h,name in enumerate(['favourable','unfavourable','total']):
            value=point[j,h] if h<2 else point[j].sum();bs=bt[:,j,h] if h<2 else bt[:,j].sum(axis=1)
            rows.append(dict(endpoint=endpoint,age=age,initial_state=initial,female=sex,rural=rural,low_education=low,metric=name,estimate=value,lower=np.quantile(bs,.025),upper=np.quantile(bs,.975)))
        if age==60 and initial==0:
            np.savez(RUN/f'analysis/{endpoint}_expectancy_draws.npz',draws=draws,profile_rewards=bt,specs=np.array(specs))
            for j,s in enumerate(specs):
                _,tr=reward(beta,profiles[j]);
                for m in range(0,361,12):curves.append(dict(endpoint=endpoint,female=s[0],rural=s[1],low_education=s[2],age=60+m/12,favourable=tr[m,0],unfavourable=tr[m,1],dead=tr[m,2]))
    pd.DataFrame(rows).to_csv(RUN/f'results/{endpoint}_expectancies.csv',index=False)
    pd.DataFrame(curves).to_csv(RUN/f'results/{endpoint}_occupancy.csv',index=False)
    # All 16 subsets allocate the urban, higher-education versus rural,
    # primary-or-less contrast to four directed transition mechanisms.
    parts=[];contrasts=[]
    for female in [0,1]:
      for contrast,left,right in [('joint',(0,0),(1,1)),('education_urban',(0,0),(0,1)),('education_rural',(1,0),(1,1)),('rural_high',(0,0),(1,0)),('rural_low',(0,1),(1,1))]:
        good=profile(female,*left);bad=profile(female,*right)
        mix=np.array([np.array([good[t] if mask&(1<<t) else bad[t] for t in range(4)]) for mask in range(16)])
        vv=many_rewards(np.concatenate([beta[None],draws]),mix,60.,30.,0)
        contributions=np.zeros((len(vv),4,2))
        for t in range(4):
            for mask in range(16):
                if not(mask&(1<<t)):
                    m=mask.bit_count();w=math.factorial(m)*math.factorial(3-m)/24
                    contributions[:,t]+=w*(vv[:,mask|(1<<t)]-vv[:,mask])
        assert np.max(np.abs(contributions.sum(axis=1)-(vv[:,15]-vv[:,0])))<1e-10
        for h,name in enumerate(['favourable','unfavourable','total']):
            val=(vv[:,15]-vv[:,0])[:,h] if h<2 else (vv[:,15]-vv[:,0]).sum(axis=1)
            contrasts.append(dict(endpoint=endpoint,contrast=contrast,female=female,metric=name,estimate=val[0],lower=np.quantile(val[1:],.025),upper=np.quantile(val[1:],.975)))
            for t,label in enumerate(['Onset','Death from favourable state','Recovery','Death from unfavourable state']):
                val=contributions[:,t,h] if h<2 else contributions[:,t].sum(axis=1)
                parts.append(dict(endpoint=endpoint,contrast=contrast,female=female,metric=name,transition=['12','13','21','23'][t],channel=label,estimate=val[0],lower=np.quantile(val[1:],.025),upper=np.quantile(val[1:],.975)))
    pd.DataFrame(parts).to_csv(RUN/f'results/{endpoint}_decomposition.csv',index=False)
    pd.DataFrame(contrasts).to_csv(RUN/f'results/{endpoint}_inequality.csv',index=False)
    # Service and brightness profiles are kept separate in the same joint model.
    rows=[]
    for variable in ['service','brightness']:
      for female in [0,1]:
       for rural,low in [(0,0),(1,1)]:
        pp=np.array([profile(female,rural,low,**{variable:v}) for v in [-.5,.5]])
        for initial in [0,1]:
          vv=many_rewards(np.concatenate([beta[None],draws]),pp,60.,30.,initial);dd=vv[:,1]-vv[:,0]
          for h,name in enumerate(['favourable','unfavourable','total']):
            val=dd[:,h] if h<2 else dd.sum(axis=1)
            rows.append(dict(endpoint=endpoint,variable=variable,female=female,rural=rural,low_education=low,initial_state=initial,metric=name,estimate=val[0],lower=np.quantile(val[1:],.025),upper=np.quantile(val[1:],.975)))
    pd.DataFrame(rows).to_csv(RUN/f'results/{endpoint}_context_contrasts.csv',index=False)
    print(endpoint,'expectancies and transition decomposition complete',flush=True)

if __name__=='__main__':
 import sys
 run(sys.argv[1] if len(sys.argv)>1 else 'independence')
