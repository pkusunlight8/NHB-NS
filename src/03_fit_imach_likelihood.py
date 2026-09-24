"""Monthly interpolated Markov likelihood with interval-observed death status.

The fitted multinomial transition parameters and municipal sandwich covariance
are exported to official IMaCh in calculation mode. No midpoint death dates enter
the observed-window likelihood. The model preserves each observed visit month.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='4'
from config import *
import sys,time
import numpy as np,pandas as pd
from scipy.optimize import minimize
from numba import njit,prange

FEATURES=['intercept','age10','baseline_female','baseline_rural','baseline_low_education',
 'baseline_service_z','baseline_brightness_z','baseline_gdp_per_registered_resident_z',
 'baseline_retail_per_registered_resident_z','baseline_urbanization_percent_z']

@njit(cache=True)
def observation_scores(theta,X,months,lower_months,origin,destination):
    n,k=X.shape;p=4*k;beta=theta.reshape(4,k)
    ll=np.empty(n);scores=np.zeros((n,p));pred=np.empty(n)
    for i in range(n):
        v0=1. if origin[i]==0 else 0.;v1=1.-v0
        g0=np.zeros(p);g1=np.zeros(p);x=X[i].copy()
        lower_alive=1.;lower_gradient=np.zeros(p)
        for m in range(months[i]):
            x[1]=X[i,1]+m/120.
            eta=np.zeros(4)
            for a in range(4):
                for j in range(k):eta[a]+=beta[a,j]*x[j]
            z0=np.exp(eta[0]);z1=np.exp(eta[1]);z2=np.exp(eta[2]);z3=np.exp(eta[3])
            p00=1/(1+z0+z1);p01=z0*p00;p02=z1*p00
            p11=1/(1+z2+z3);p10=z2*p11;p12=z3*p11
            # Explicit fresh buffers prevent Numba's parallel array fusion from
            # reusing a recurrent derivative buffer across the two state updates.
            h0=np.empty(p);h1=np.empty(p)
            for c in range(p):
                h0[c]=g0[c]*p00+g1[c]*p10
                h1[c]=g0[c]*p01+g1[c]*p11
            for j in range(k):
                h0[j]-=v0*p00*p01*x[j]
                h0[k+j]-=v0*p00*p02*x[j]
                h0[2*k+j]+=v1*p10*(1-p10)*x[j]
                h0[3*k+j]-=v1*p10*p12*x[j]
                h1[j]+=v0*p01*(1-p01)*x[j]
                h1[k+j]-=v0*p01*p02*x[j]
                h1[2*k+j]-=v1*p11*p10*x[j]
                h1[3*k+j]-=v1*p11*p12*x[j]
            nv0=v0*p00+v1*p10;nv1=v0*p01+v1*p11
            v0=nv0;v1=nv1;g0=h0;g1=h1
            if m+1==lower_months[i]:
                lower_alive=v0+v1
                for c in range(p):lower_gradient[c]=g0[c]+g1[c]
        if destination[i]==0:prob=v0;g=g0
        elif destination[i]==1:prob=v1;g=g1
        else:prob=lower_alive-v0-v1;g=lower_gradient-g0-g1
        prob=max(prob,1e-15);ll[i]=np.log(prob);scores[i]=g/prob;pred[i]=prob
    return ll,scores,pred

def run(endpoint):
    d=pd.read_csv(RUN/f'analysis/{endpoint}_intervals.csv.gz',low_memory=False)
    X=np.column_stack([np.ones(len(d)),(d.exact_start_age.to_numpy()-60)/10,d[FEATURES[2:]].to_numpy(float)])
    m=d.duration_months.to_numpy(np.int64);a=d.origin.to_numpy(np.int64);b=d.destination.to_numpy(np.int64)
    lower=np.where(b==2,np.rint(d.death_lower_dt.fillna(0).to_numpy()*12),0).astype(np.int64)
    assert np.all(lower<m) and np.all(lower>=0)
    n,k=X.shape;start=np.zeros((4,k));start[:,0]=[-6.4,-7.0,-4.2,-5.9];start[:,1]=[.6,.75,-.3,.65]
    if endpoint=='multidomain':start[:,0]=[-3.6,-7.0,-3.6,-6.3]
    dest=RUN/'analysis'/('imach_'+endpoint);dest.mkdir(exist_ok=True)
    trace=[];t0=time.time()
    def obj(theta):
        ll,scores,_=observation_scores(theta,X,m,lower,a,b)
        return -ll.sum()/n,-scores.sum(axis=0)/n
    f,g=obj(start.ravel())
    # Independent finite differences check every fitted direction on real observations.
    eps=1e-5;numeric=np.zeros(k*4)
    for j in range(k*4):
        e=np.zeros(k*4);e[j]=eps
        numeric[j]=(obj(start.ravel()+e)[0]-obj(start.ravel()-e)[0])/(2*eps)
    grad_error=float(np.max(np.abs(numeric-g)));assert grad_error<1e-6,grad_error
    def callback(t):
        loss,gradient=obj(t);trace.append(dict(iteration=len(trace)+1,mean_negative_log_likelihood=loss,max_gradient=float(np.abs(gradient).max()),seconds=time.time()-t0))
        if len(trace)%20==0:print(endpoint,trace[-1],flush=True)
    fit=minimize(obj,start.ravel(),jac=True,method='L-BFGS-B',bounds=[(-25,25)]*(4*k),callback=callback,options={'maxiter':1500,'ftol':1e-13,'gtol':1e-8,'maxls':40,'maxcor':30})
    # Curvature is calculated from independent perturbations of the analytic score.
    beta=fit.x;ll,scores,pr=observation_scores(beta,X,m,lower,a,b)
    H=np.zeros((k*4,k*4));eps=2e-5
    for j in range(k*4):
        e=np.zeros(k*4);e[j]=eps
        H[:,j]=(obj(beta+e)[1]-obj(beta-e)[1])/(2*eps)*n
    H=(H+H.T)/2;ev=np.linalg.eigvalsh(H)
    assert ev.min()>1e-7,(endpoint,ev.min())
    inv=np.linalg.inv(H)
    groups=d.city_cn.astype(str).str.replace('市$','',regex=True)
    city_score=pd.DataFrame(scores).groupby(groups.to_numpy()).sum().to_numpy()
    G=len(city_score);sandwich=inv@(city_score.T@city_score)@inv*(G/(G-1))*((n-1)/(n-k*4))
    sandwich=(sandwich+sandwich.T)/2
    assert np.linalg.eigvalsh(sandwich).min()>-1e-8
    assert np.max(np.abs(fit.jac))<1e-5
    assert not np.any(np.abs(beta)>24.99)
    # Keep the portable parameter file free of participant-level design rows.
    np.savez(dest/'fit.npz',beta=beta.reshape(4,k),covariance=sandwich,information=H,model_covariance=inv)
    rows=[]
    for j,t in enumerate(['12','13','21','23']):
        for i,c in enumerate(FEATURES):rows.append(dict(endpoint=endpoint,transition=t,feature=c,coefficient=beta[j*k+i],se=np.sqrt(sandwich[j*k+i,j*k+i])))
    pd.DataFrame(rows).to_csv(RUN/f'results/imach_{endpoint}_coefficients.csv',index=False)
    pd.DataFrame(trace).to_csv(dest/'optimization.csv',index=False)
    summary=dict(endpoint=endpoint,intervals=n,participants=d.subject_id.nunique(),municipalities=G,deaths=int(d.destination.eq(2).sum()),transitions=d.groupby(['origin','destination']).size().rename('n').reset_index().to_dict('records'),features=FEATURES,optimizer_success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),negative_log_likelihood=float(-ll.sum()),max_mean_score=float(np.abs(fit.jac).max()),minimum_information_eigenvalue=float(ev.min()),gradient_finite_difference_max_error=grad_error,seconds=time.time()-t0,death_likelihood='Probability of absorbing state by the observed end of the death-report interval')
    save_json(dest/'fit_summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'independence')
