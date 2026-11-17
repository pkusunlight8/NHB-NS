"""Exact city sign-flip inference and finite-observation summaries."""
import warnings
import numpy as np

def mean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(a, axis=axis)


def median(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def slope(y):
    x = np.arange(y.shape[0], dtype=float)[:,None]
    ok = np.isfinite(y)
    n = ok.sum(axis=0)
    mx = np.sum(np.where(ok,x,0),axis=0)/np.maximum(n,1)
    my = mean(y)
    xx = np.where(ok,x-mx,0)
    yy = np.where(ok,y-my,0)
    return np.sum(xx*yy,axis=0)/np.maximum(np.sum(xx*xx,axis=0),1e-12)


def exact_p(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return np.nan
    if np.all(v == 0):
        return 1.
    observed = abs(v.sum())
    exceed = 0
    for start in range(0, 2**n, 32768):
        integers = np.arange(start,min(start+32768,2**n),dtype=np.uint32)
        signs = (((integers[:,None] >> np.arange(n,dtype=np.uint32)) & 1)*2).astype(float)-1
        exceed += int((np.abs(signs@v) >= observed-1e-12).sum())
    return exceed/2**n
