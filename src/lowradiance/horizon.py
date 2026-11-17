"""Unchanged monthly trapezoidal integrals at 6, 12, 18 and 24 months."""
import numpy as np
from numba import njit

@njit(cache=True)
def direct_windows(coeff,h,q,origin,step=1.):
    values=np.empty((4,len(coeff),len(h)));terminal=np.empty((len(coeff),len(h)))
    ph=h.shape[1];pq=q.shape[1];half=pq//2;period=int(round(6/step))
    for d in range(len(coeff)):
        for r in range(len(h)):
            lh=0.;lq=0.
            for k in range(ph):lh+=coeff[d,k]*h[r,k]
            for k in range(pq):lq+=coeff[d,ph+1+k]*q[r,k]
            lin=coeff[d,ph+1+half-2]+origin[r]*coeff[d,ph+1+pq-2]
            quad=coeff[d,ph+1+half-1]+origin[r]*coeff[d,ph+1+pq-1]
            prev=origin[r];total=0.;current=0.
            for j in range(1,int(round(24/step))+1):
                t=j*step/12;eta=lh+np.exp(coeff[d,ph])*np.log(t)
                survival=np.exp(-np.exp(eta)) if eta<700 else 0.
                change=(lq+lin*(t-2)+quad*(t-2)**2+np.log(t))*(1-2*origin[r])
                state=1/(1+np.exp(-change)) if change>=0 else np.exp(change)/(1+np.exp(change))
                current=survival*state;total+=(prev+current)*step/24;prev=current
                if j%period==0:values[j//period-1,d,r]=total
            terminal[d,r]=current
    return terminal,values


@njit(cache=True)
def fre_windows(beta,x,origin):
    values=np.empty((4,len(beta),len(x)));terminal=np.empty((len(beta),len(x)))
    for d in range(len(beta)):
        growth=np.empty(4)
        for c in range(4):growth[c]=np.exp(beta[d,c,1]/120.)
        for n in range(len(x)):
            eta=np.zeros(4);safe=True
            for c in range(4):
                for k in range(x.shape[1]):eta[c]+=x[n,k]*beta[d,c,k]
                end=eta[c]+23/120.*beta[d,c,1]
                if min(eta[c],end)<-500. or max(eta[c],end)>500.:safe=False
            s0=1. if origin[n]==0 else 0.;s1=1.-s0;total=0.
            if safe:
                ev=np.exp(eta)
                for m in range(24):
                    z0=1.+ev[0]+ev[1];z1=1.+ev[2]+ev[3]
                    n0=s0/z0+s1*(ev[2]/z1);n1=s0*(ev[0]/z0)+s1/z1
                    total+=(s1+n1)/24.;s0=n0;s1=n1
                    for c in range(4):ev[c]*=growth[c]
                    if (m+1)%6==0:values[(m+1)//6-1,d,n]=total
            else:
                for m in range(24):
                    a=eta[0]+m/120*beta[d,0,1];b=eta[1]+m/120*beta[d,1,1]
                    c=eta[2]+m/120*beta[d,2,1];e=eta[3]+m/120*beta[d,3,1]
                    u=max(0.,a,b);v=max(0.,c,e)
                    z0=np.exp(-u)+np.exp(a-u)+np.exp(b-u);z1=np.exp(c-v)+np.exp(-v)+np.exp(e-v)
                    n0=s0*np.exp(-u)/z0+s1*np.exp(c-v)/z1
                    n1=s0*np.exp(a-u)/z0+s1*np.exp(-v)/z1
                    total+=(s1+n1)/24.;s0=n0;s1=n1
                    if (m+1)%6==0:values[(m+1)//6-1,d,n]=total
            terminal[d,n]=s1
    return terminal,values
