"""Fixed-budget allocation, shortage and oracle regret."""
import numpy as np

def shares(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('Expected finite nonnegative one-dimensional masses.')
    if a.sum()<=0:
        raise ValueError('A zero total has no defined normalized need share.')
    return a/a.sum()


def project_share(q,c,cap_multiple=2.0):
    """Unique minimizer of sum((a-q)**2/c), sum(a)=1, 0<=a<=cap*c."""
    c=shares(c);q=np.asarray(q,dtype=float)
    if q.shape!=c.shape or not np.isfinite(q).all() or (q<0).any() or (c<=0).any():
        raise ValueError('Projection requires positive baseline masses and nonnegative scores.')
    if cap_multiple<1:
        raise ValueError('The stated total budget is infeasible below a unit cap multiplier.')
    if q.sum()<=0:q=c.copy()
    else:q=shares(q)
    if cap_multiple==1:return c.copy()
    upper=cap_multiple*c
    low=-cap_multiple
    high=float(np.max(q/c)+1)
    for _ in range(100):
        middle=(low+high)/2
        a=np.clip(q-middle*c,0,upper)
        if a.sum()>1:low=middle
        else:high=middle
    allocation=np.clip(q-((low+high)/2)*c,0,upper)
    if abs(allocation.sum()-1)>1e-12 or (allocation>upper+1e-12).any():
        raise ArithmeticError('Fixed-budget residual exceeds the numerical tolerance.')
    return allocation


def allocate_risk(predicted_risk,baseline_mass):
    c=shares(baseline_mass)
    score=np.asarray(predicted_risk,float)
    if score.shape!=c.shape or not np.isfinite(score).all() or (score<0).any():
        raise ValueError('Predicted risks must be finite and nonnegative.')
    return project_share(score*c,c)


def loss(n,a):
    n=np.asarray(n,float);a=np.asarray(a,float)
    if n.shape!=a.shape:raise ValueError('Need and allocation support must agree.')
    return float(np.maximum(n-a,0).sum())


def oracle(n,c):
    n=shares(n);c=shares(c)
    a=project_share(n,c)
    exact=float(np.maximum(n-2*c,0).sum())
    if abs(loss(n,a)-exact)>1e-11:
        raise ArithmeticError('Oracle projection and exact unmet-share bound disagree.')
    return a,exact


def fair_project(q,c,group_proportions,tolerance=.05):
    """Same quadratic projection with fixed baseline group-coverage constraints."""
    from scipy.optimize import minimize
    c=shares(c);q=np.asarray(q,float);q=c.copy() if q.sum()<=0 else shares(q)
    p=np.asarray(group_proportions,float)
    if p.ndim!=2 or p.shape[1]!=len(c) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Group proportions must be a finite group-by-city matrix in [0,1].')
    lower=p@c-tolerance
    def objective(a):return .5*np.sum((a-q)**2/c),(a-q)/c
    fit=minimize(objective,c.copy(),jac=True,method='SLSQP',bounds=list(zip(np.zeros_like(c),2*c)),
                 constraints=[{'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones_like(a)},
                              {'type':'ineq','fun':lambda a:p@a-lower,'jac':lambda a:p}],
                 options={'maxiter':3000,'ftol':1e-12})
    a=fit.x
    residual=max(abs(a.sum()-1),float(np.maximum(a-2*c,0).max()),float(np.maximum(-a,0).max()),float(np.maximum(lower-p@a,0).max()))
    if not fit.success or residual>1e-8:raise ArithmeticError(f'Fair allocation failed: {fit.message}; residual={residual}')
    return a


def fair_oracle(n,c,group_proportions,tolerance=.05):
    """Independent LP oracle for the group-constrained shortage loss."""
    from scipy.optimize import linprog
    n=shares(n);c=shares(c);p=np.asarray(group_proportions,float);k=len(c)
    objective=np.r_[np.zeros(k),np.ones(k)]
    inequality=np.vstack([np.c_[-np.eye(k),-np.eye(k)],np.c_[-p,np.zeros_like(p)]])
    bound=np.r_[-n,-p@c+tolerance]
    lp=linprog(objective,A_ub=inequality,b_ub=bound,A_eq=np.r_[np.ones(k),np.zeros(k)][None,:],b_eq=[1],bounds=[(0,float(v)) for v in 2*c]+[(0,None)]*k,method='highs')
    if not lp.success:raise ArithmeticError(lp.message)
    return lp.x[:k],float(lp.fun)


def upper_tail_mean(values,alpha=.90):
    """Equal-city finite-sample CVaR, with fractional mass on the final order statistic."""
    x=np.sort(np.asarray(values,float))[::-1]
    if not len(x) or not np.isfinite(x).all() or not 0<=alpha<1:raise ValueError('Invalid tail sample.')
    mass=(1-alpha)*len(x);whole=int(np.floor(mass));fraction=mass-whole
    numerator=x[:whole].sum()
    if fraction>0:numerator+=fraction*x[whole]
    return float(numerator/mass)


def evaluate_pair(need_mass,baseline_mass,research_allocation,traditional_allocation):
    n=shares(need_mass);c=shares(baseline_mass)
    a=np.asarray(research_allocation,float);b=np.asarray(traditional_allocation,float)
    oa,ol=oracle(n,c)
    shortage_a=np.maximum(n-a,0);shortage_b=np.maximum(n-b,0)
    rate_a=np.divide(shortage_a,n,out=np.zeros_like(n),where=n>0)
    rate_b=np.divide(shortage_b,n,out=np.zeros_like(n),where=n>0)
    return {'research_loss':float(shortage_a.sum()),'traditional_loss':float(shortage_b.sum()),
            'oracle_loss':ol,'research_regret':float(shortage_a.sum()-ol),
            'traditional_regret':float(shortage_b.sum()-ol),
            'regret_difference':float((shortage_a-shortage_b).sum()),
            'research_absolute_mismatch':float(np.abs(n-a).sum()),
            'traditional_absolute_mismatch':float(np.abs(n-b).sum()),
            'research_shortage_cvar90':upper_tail_mean(rate_a),
            'traditional_shortage_cvar90':upper_tail_mean(rate_b),
            'cities_benefiting':int((shortage_a<shortage_b-1e-10).sum()),
            'cities_harmed':int((shortage_a>shortage_b+1e-10).sum()),
            'cities_equal':int((np.abs(shortage_a-shortage_b)<=1e-10).sum()),
            'cities_allocation_changed':int((np.abs(a-b)>1e-10).sum())}
