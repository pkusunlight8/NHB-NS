"""Exact likelihood derivatives and projection kernels."""
import numpy as np
from numba import njit

@njit(cache=True)
def loglik_scores(beta, design, origin, destination, ends, lowers, weights):
    n,p=design.shape
    loglik=np.empty(n);scores=np.empty((n,4*p))
    for i in range(n):
        s0=1.0 if origin[i]==0 else 0.0;s1=1.0-s0
        d0=np.zeros(4*p);d1=np.zeros(4*p)
        low_surv=1.0;low_grad=np.zeros(4*p)
        x=design[i].copy()
        for month in range(ends[i]):
            x[1]=design[i,1]+month/120.0
            eta=np.zeros(4)
            for c in range(4):
                for k in range(p):eta[c]+=beta[c,k]*x[k]
            top0=max(0.0,eta[0],eta[1]);top1=max(0.0,eta[2],eta[3])
            z0=np.exp(-top0)+np.exp(eta[0]-top0)+np.exp(eta[1]-top0)
            z1=np.exp(-top1)+np.exp(eta[2]-top1)+np.exp(eta[3]-top1)
            p00=np.exp(-top0)/z0;p01=np.exp(eta[0]-top0)/z0;p02=np.exp(eta[1]-top0)/z0
            p10=np.exp(eta[2]-top1)/z1;p11=np.exp(-top1)/z1;p12=np.exp(eta[3]-top1)/z1
            nd0=np.empty(4*p);nd1=np.empty(4*p)
            for c in range(4):
                if c==0:a0=-p00*p01;a1=p01*(1-p01);b0=0.0;b1=0.0
                elif c==1:a0=-p00*p02;a1=-p01*p02;b0=0.0;b1=0.0
                elif c==2:a0=0.0;a1=0.0;b0=p10*(1-p10);b1=-p11*p10
                else:a0=0.0;a1=0.0;b0=-p10*p12;b1=-p11*p12
                for k in range(p):
                    j=c*p+k
                    nd0[j]=d0[j]*p00+d1[j]*p10+(s0*a0+s1*b0)*x[k]
                    nd1[j]=d0[j]*p01+d1[j]*p11+(s0*a1+s1*b1)*x[k]
            ns0=s0*p00+s1*p10;ns1=s0*p01+s1*p11
            s0=ns0;s1=ns1;d0=nd0;d1=nd1
            if month+1==lowers[i]:
                low_surv=s0+s1
                for j in range(4*p):low_grad[j]=d0[j]+d1[j]
        if destination[i]==0:q=s0;dq=d0
        elif destination[i]==1:q=s1;dq=d1
        else:q=low_surv-s0-s1;dq=low_grad-d0-d1
        safe=max(1e-300,min(1.0,q));loglik[i]=weights[i]*np.log(safe)
        for j in range(4*p):scores[i,j]=weights[i]*dq[j]/safe if q>1e-300 and q<1.0 else 0.0
    return loglik,scores


def expectancy_draws_fast(draws, design, start_age=60, end_age=90, initial_state=0):
    from scipy.special import softmax
    beta=np.asarray(draws,float);x=np.atleast_2d(np.asarray(design,float))
    horizon=12*(float(end_age)-float(start_age))
    if horizon<0 or not np.isclose(horizon,round(horizon)) or initial_state not in (0,1,2):raise ValueError("Invalid horizon or initial state")
    state=np.zeros((len(beta),len(x),2))
    if initial_state<2:state[:,:,initial_state]=1
    reward=np.zeros_like(state);zeros=np.zeros(state.shape[:2]+(1,))
    for m in range(int(round(horizon))):
        current=x.copy();current[:,1]=(float(start_age)-60)/10+m/120
        eta=np.einsum('hp,bcp->bhc',current,beta,optimize=True)
        first=softmax(np.concatenate([zeros,eta[:,:,:2]],axis=-1),axis=-1)
        second=softmax(np.concatenate([eta[:,:,2:3],zeros,eta[:,:,3:4]],axis=-1),axis=-1)
        after=np.empty_like(state)
        after[:,:,0]=state[:,:,0]*first[:,:,0]+state[:,:,1]*second[:,:,0]
        after[:,:,1]=state[:,:,0]*first[:,:,1]+state[:,:,1]*second[:,:,1]
        reward+=(state+after)/24;state=after
    return reward


def shapley_draws_fast(draws, a, b, start_age=60, end_age=90):
    import math
    beta=np.asarray(draws,float);mixed=np.broadcast_to(beta[:,None,:,:],(len(beta),16,4,beta.shape[2])).copy()
    delta=np.einsum('bcp,p->bc',beta,np.asarray(a)-np.asarray(b))
    for mask in range(16):
        for j in range(4):
            if mask&(1<<j):mixed[:,mask,j,0]+=delta[:,j]
    values=expectancy_draws_fast(mixed.reshape(-1,4,beta.shape[2]),b,start_age,end_age).reshape(len(beta),16,2)
    out=np.zeros((len(beta),4,2))
    for j in range(4):
        for mask in range(16):
            if mask&(1<<j):continue
            k=mask.bit_count();w=math.factorial(k)*math.factorial(3-k)/math.factorial(4)
            out[:,j]+=w*(values[:,mask|(1<<j)]-values[:,mask])
    if not np.allclose(out.sum(axis=1),values[:,15]-values[:,0],atol=1e-10,rtol=1e-10):raise ArithmeticError("Shapley sum mismatch")
    return out
