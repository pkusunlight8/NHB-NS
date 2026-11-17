"""Consistent HiGHS scheduler settings across the packaged LP entry points."""
import warnings
from scipy.optimize import OptimizeWarning, linprog as scipy_linprog


def linprog(*args, **kwargs):
    # The original sparse solver specifies one thread. Apply this setting to
    # every packaged LP so HiGHS can share its process-wide scheduler safely.
    options = dict(kwargs.pop('options', {}) or {})
    options['threads'] = 1
    kwargs['options'] = options
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*threads',
                                category=OptimizeWarning)
        return scipy_linprog(*args, **kwargs)
