"""Interval-observed monthly three-state model and functional life expectancy."""
import itertools
import math
from .core import MissingInputError, require_columns
CHANNELS = ("12", "13", "21", "23")
DEFAULT_FEATURES = ("intercept", "age10", "female", "rural", "low_education", "service_z", "brightness_z", "gdp_z", "retail_z", "urbanization_z")

def transition_matrix(coefficients, design):
    """Monthly multinomial transition probabilities, vectorized over profiles."""
    import numpy as np
    from scipy.special import softmax
    beta, x = np.asarray(coefficients, dtype=float), np.asarray(design, dtype=float)
    if beta.ndim != 2 or beta.shape[0] != 4 or x.shape[-1] != beta.shape[1]:
        raise ValueError("Expected coefficients (4,p) and design (...,p)")
    eta = x @ beta.T
    zeros = np.zeros(eta.shape[:-1] + (1,))
    first = softmax(np.concatenate([zeros, eta[..., :2]], axis=-1), axis=-1)
    second = softmax(np.concatenate([eta[..., 2:3], zeros, eta[..., 3:4]], axis=-1), axis=-1)
    out = np.zeros(eta.shape[:-1] + (3, 3))
    out[..., 0, :], out[..., 1, :], out[..., 2, 2] = first, second, 1.0
    return out


def ctmc_matrix(coefficients, design, step_months=1.0):
    """CTMC sensitivity with log transition intensities in month units.

    These coefficients require a separate CTMC fit: monthly-logit estimates are
    not reinterpreted as CTMC estimates. Age is held within each monthly step.
    """
    import numpy as np
    from scipy.linalg import expm
    beta, x = np.asarray(coefficients, dtype=float), np.asarray(design, dtype=float)
    rates = np.exp(np.clip(x @ beta.T, -60, 40))
    q = np.zeros(rates.shape[:-1] + (3, 3))
    q[..., 0, 1], q[..., 0, 2] = rates[..., 0], rates[..., 1]
    q[..., 1, 0], q[..., 1, 2] = rates[..., 2], rates[..., 3]
    q[..., 0, 0] = -q[..., 0, :].sum(axis=-1)
    q[..., 1, 1] = -q[..., 1, :].sum(axis=-1)
    return expm(q * float(step_months))


def _validate_intervals(frame):
    import numpy as np
    require_columns(frame, ["origin", "destination", "months", "death_lower", "death_upper", "age"], "Clinical intervals")
    origins = frame["origin"].to_numpy(float)
    destinations = frame["destination"].to_numpy(float)
    if not np.isin(origins, [0, 1]).all() or not np.isin(destinations, [0, 1, 2]).all():
        raise ValueError("Clinical origins must be living states; destinations must be classified")
    if not np.isfinite(frame["age"].to_numpy(float)).all():
        raise ValueError("Attained age is missing")
    death = destinations == 2
    ends = np.where(death, frame["death_upper"], frame["months"]).astype(float)
    lowers = np.where(death, frame["death_lower"], 0).astype(float)
    if not np.isfinite(ends).all() or not np.isfinite(lowers).all():
        raise MissingInputError("Exact interview months and verified death-reporting windows are required")
    if not np.allclose(ends, np.round(ends)) or not np.allclose(lowers, np.round(lowers)):
        raise ValueError("Monthly likelihood requires integer month endpoints")
    if (ends <= lowers).any() or (lowers < 0).any():
        raise ValueError("Invalid observed interview/death window")
    return death, ends.astype(int), lowers.astype(int)


def interval_probabilities(coefficients, frame, design, age_index=1, model="monthly"):
    """Observed living endpoint P-product or death S(L)-S(U) probability."""
    import numpy as np
    death, ends, lowers = _validate_intervals(frame)
    x = np.asarray(design, dtype=float)
    if x.ndim != 2 or len(x) != len(frame) or not np.isfinite(x).all():
        raise ValueError("Clinical design must be finite and have one row per interval")
    if age_index is None or not 0 <= age_index < x.shape[1]:
        raise ValueError("Specify the column holding (attained_age-60)/10")
    if model not in ("monthly", "ctmc"):
        raise ValueError("model must be monthly or ctmc")
    matrix = transition_matrix if model == "monthly" else ctmc_matrix
    n = len(frame)
    state = np.zeros((n, 3))
    state[np.arange(n), frame["origin"].to_numpy(int)] = 1
    s_lower = np.where(lowers == 0, 1.0, np.nan)
    probability = np.full(n, np.nan)
    for month in range(int(ends.max(initial=0))):
        active = ends > month
        current = x[active].copy()
        current[:, age_index] = (frame.loc[active, "age"].to_numpy(float) - 60) / 10 + month / 120
        state[active] = np.einsum("ni,nij->nj", state[active], matrix(coefficients, current))
        elapsed = month + 1
        at_lower = death & (lowers == elapsed)
        s_lower[at_lower] = state[at_lower, :2].sum(axis=1)
        at_end = ends == elapsed
        alive_end = at_end & ~death
        ix = np.flatnonzero(alive_end)
        probability[ix] = state[ix, frame.loc[alive_end, "destination"].to_numpy(int)]
        died_end = at_end & death
        probability[died_end] = s_lower[died_end] - state[died_end, :2].sum(axis=1)
    if (probability < -1e-10).any() or (probability > 1 + 1e-10).any():
        raise FloatingPointError("Invalid likelihood probability; check numerical integration")
    return np.clip(probability, 1e-300, 1)


def central_jacobian(function, point, relative_step=1e-5):
    import numpy as np
    point = np.asarray(point, dtype=float)
    base = np.atleast_1d(function(point))
    jac = np.empty((base.size, point.size))
    for j in range(point.size):
        h = relative_step * max(1.0, abs(point[j]))
        plus, minus = point.copy(), point.copy()
        plus[j], minus[j] = plus[j] + h, minus[j] - h
        jac[:, j] = (np.atleast_1d(function(plus)) - np.atleast_1d(function(minus))) / (2 * h)
    return jac


def fit_monthly_model(frame, design, clusters, initial=None, model="monthly", weights=None, maxiter=1000, bounds=None, calculate_covariance=True, return_cluster_influence=False):
    """Maximum observed-window likelihood with municipality-cluster covariance.

    Returns optimizer, score and observed-information diagnostics alongside the
    coefficients and municipality-cluster sandwich covariance.
    """
    import numpy as np
    from scipy.optimize import minimize
    if return_cluster_influence and (not calculate_covariance or bounds is not None):
        raise ValueError("Cluster influence requires unbounded fitting with observed information")
    x = np.asarray(design, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or len(x) != len(frame) or len(frame) == 0:
        raise ValueError("Design/frame mismatch")
    if not np.isfinite(x).all():
        raise ValueError("Clinical design must contain finite values")
    if model not in ("monthly", "ctmc"):
        raise ValueError("model must be monthly or ctmc")
    _validate_intervals(frame)
    if not frame["destination"].eq(2).any():
        raise MissingInputError("Four-channel mortality model is not identifiable from a training sample with no observed deaths; no mortality covariance or expectancy draws are generated")
    p = 4 * x.shape[1]
    start = np.zeros((4, x.shape[1]))
    start[:, 0] = -6
    start[:, 1] = 0.5
    if initial is not None:
        start = np.asarray(initial, dtype=float).reshape(4, x.shape[1])
    w = np.ones(len(frame)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(frame),) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Positive finite interval weights are required")
    fast_monthly = model == "monthly"
    if fast_monthly:
        from .clinical_fast import loglik_scores
        _, ends, lowers = _validate_intervals(frame)
        origin=frame["origin"].to_numpy(np.int64)
        destination=frame["destination"].to_numpy(np.int64)
        fast_x=x.copy(); fast_x[:,1]=(frame["age"].to_numpy(float)-60)/10
        def evaluated(flat):
            return loglik_scores(flat.reshape(4,-1),fast_x,origin,destination,ends,lowers,w)
        def objective_score(flat):
            values, score=evaluated(flat)
            return -values.sum(),-score.sum(axis=0)
        def contributions(flat): return evaluated(flat)[0]
        def objective(flat): return -contributions(flat).sum()
        opt=minimize(objective_score,start.ravel(),method="L-BFGS-B",jac=True,bounds=bounds,
                     options={"maxiter":int(maxiter),"ftol":1e-12,"gtol":1e-7,"maxls":40})
        scores=evaluated(opt.x)[1]
    else:
        def contributions(flat):
            return w * np.log(interval_probabilities(flat.reshape(4, -1), frame, x, model=model))
        def objective(flat):
            return -contributions(flat).sum()
        opt = minimize(objective, start.ravel(), method="L-BFGS-B", jac="3-point",
                       bounds=bounds, options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7, "maxls": 40})
        scores = central_jacobian(contributions, opt.x)
    mean_score = float(np.max(np.abs(scores.mean(axis=0))))
    result = {"coefficients": opt.x.reshape(4, -1), "log_likelihood": -float(opt.fun), "optimizer_success": bool(opt.success), "optimizer_message": str(opt.message), "maximum_absolute_mean_score": mean_score, "model": model, "n_intervals": len(frame), "n_parameters": p, "covariance": None, "information_positive_definite": None, "converged": bool(opt.success and mean_score < 1e-5)}
    if calculate_covariance:
        cluster = np.asarray(clusters)
        if len(cluster) != len(frame) or any(v is None or str(v) == "nan" for v in cluster):
            raise MissingInputError("Municipality cluster IDs must be supplied for every interval")
        unique, inverse = np.unique(cluster.astype(str), return_inverse=True)
        g, n = len(unique), len(frame)
        if g < 2 or n <= p:
            raise ValueError("Cluster sandwich requires >=2 clusters and n>p")
        def gradient(flat):
            return -evaluated(flat)[1].sum(axis=0) if fast_monthly else central_jacobian(lambda v: np.array([objective(v)]), flat, relative_step=1e-5).ravel()
        info = central_jacobian(gradient, opt.x, relative_step=1e-4)
        info = (info + info.T) / 2
        eigenvalues = np.linalg.eigvalsh(info)
        result["information_min_eigenvalue"] = float(eigenvalues.min())
        result["information_condition_number"] = float(np.linalg.cond(info))
        result["information_positive_definite"] = bool(eigenvalues.min() > 0)
        if eigenvalues.min() <= 0:
            result["converged"] = False
            result["covariance_issue"] = "Nonpositive observed information: no coefficient draws or expectancy intervals are generated"
        else:
            totals = np.zeros((g, p))
            np.add.at(totals, inverse, scores)
            bread = np.linalg.inv(info)
            correction = g / (g - 1) * (n - 1) / (n - p)
            result["covariance"] = correction * bread @ (totals.T @ totals) @ bread
            result["n_clusters"] = g
            result["finite_sample_correction"] = float(correction)
            if return_cluster_influence:
                result["cluster_ids"] = unique
                centered = totals - totals.mean(axis=0)
                result["cluster_influence"] = np.sqrt(correction) * centered @ bread.T
    return result


def coefficient_draws(coefficients, covariance, n_draws=2000, seed=20261022):
    import numpy as np
    beta, cov = np.asarray(coefficients), np.asarray(covariance, dtype=float)
    if cov.shape != (beta.size, beta.size) or not np.isfinite(cov).all():
        raise ValueError("Full finite joint coefficient covariance is required")
    cov = (cov + cov.T) / 2
    if np.linalg.eigvalsh(cov).min() < -1e-9:
        raise ValueError("Joint covariance is not positive semidefinite")
    return np.random.default_rng(seed).multivariate_normal(beta.ravel(), cov, size=int(n_draws), check_valid="raise").reshape((int(n_draws),) + beta.shape)


def occupancy(coefficients, design, start_age=60, end_age=90, initial_state=0, model="monthly"):
    """Monthly state occupancy and exact shared horizon for given profiles."""
    import numpy as np
    x = np.atleast_2d(np.asarray(design, dtype=float))
    horizon = 12 * (float(end_age) - float(start_age))
    if horizon < 0 or not np.isclose(horizon, round(horizon)):
        raise ValueError("Projection horizon must contain a nonnegative integer number of months")
    if initial_state not in (0, 1, 2):
        raise ValueError("Unknown initial state")
    if model not in ("monthly", "ctmc"):
        raise ValueError("model must be monthly or ctmc")
    matrix = transition_matrix if model == "monthly" else ctmc_matrix
    states = np.zeros((int(round(horizon)) + 1, len(x), 3))
    states[0, :, initial_state] = 1
    for m in range(len(states) - 1):
        current = x.copy()
        current[:, 1] = (float(start_age) - 60) / 10 + m / 120
        states[m + 1] = np.einsum("hi,hij->hj", states[m], matrix(coefficients, current))
    return states


def restricted_expectancy(coefficients, design, start_age=60, end_age=90, initial_state=0, model="monthly"):
    states = occupancy(coefficients, design, start_age, end_age, initial_state, model)
    rewards = ((states[:-1, :, :2] + states[1:, :, :2]) / 24).sum(axis=0)
    return rewards


def profile_design(n_features=10):
    import numpy as np
    if n_features < 5:
        raise ValueError("Profile projection requires intercept, age10, female, rural, and low_education")
    profiles = np.asarray(list(itertools.product([0, 1], repeat=3)), dtype=int)
    x = np.zeros((8, n_features))
    x[:, 0], x[:, 2:5] = 1, profiles
    return profiles, x


def expectancy_draws(draws, design, **kwargs):
    import numpy as np
    if kwargs.get("model", "monthly") == "monthly":
        from .clinical_fast import expectancy_draws_fast
        return expectancy_draws_fast(draws, design, **{k:v for k,v in kwargs.items() if k != "model"})
    return np.stack([restricted_expectancy(beta, design, **kwargs) for beta in np.asarray(draws)])


def shapley_decomposition(coefficients, profile_a, profile_b, start_age=60, end_age=90, initial_state=0):
    """Exact four-channel replacement in years; rows are I/D rewards."""
    import numpy as np
    beta = np.asarray(coefficients, dtype=float)
    a, b = np.asarray(profile_a, dtype=float), np.asarray(profile_b, dtype=float)
    if a.shape != b.shape or a.shape != (beta.shape[1],):
        raise ValueError("Social profile design mismatch")
    values = {}
    for mask in range(16):
        # Profiles differ only in fixed covariates; replace their complete
        # predictors by adding channel-specific intercept differences to B.
        mixed = beta.copy()
        for j in range(4):
            if mask & (1 << j):
                mixed[j, 0] += beta[j] @ (a - b)
        values[mask] = restricted_expectancy(mixed, b, start_age, end_age, initial_state)[0]
    contributions = np.zeros((4, 2))
    for j in range(4):
        for mask in range(16):
            if mask & (1 << j):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(3 - k) / math.factorial(4)
            contributions[j] += weight * (values[mask | (1 << j)] - values[mask])
    difference = values[15] - values[0]
    if not np.allclose(contributions.sum(axis=0), difference, atol=1e-10, rtol=1e-10):
        raise ArithmeticError("Exact Shapley identity failed")
    return {"channels": CHANNELS, "contributions": contributions, "difference": difference, "subset_values": values}


def calibration_table(coefficients, frame, design, n_bins=5, model="monthly"):
    """Observed-interval endpoint calibration; mortality must be evaluated separately.

    Living-endpoint probabilities are conditional on survival through each
    individual's observed interval. This is not a fixed-time calibration curve.
    """
    import numpy as np
    import pandas as pd
    _validate_intervals(frame)
    living = frame["destination"].isin([0, 1])
    sub, x = frame.loc[living].reset_index(drop=True), np.asarray(design)[living]
    p = []
    for dest in (0, 1):
        target = sub.copy()
        target["destination"] = dest
        p.append(interval_probabilities(coefficients, target, x, model=model))
    predicted = p[1] / np.maximum(p[0] + p[1], 1e-300)
    data = pd.DataFrame({"predicted": predicted, "observed": sub["destination"].eq(1).astype(float)})
    data["bin"] = pd.qcut(data["predicted"], q=min(int(n_bins), len(data)), duplicates="drop")
    return data.groupby("bin", observed=True).agg(n=("observed", "size"), predicted=("predicted", "mean"), observed=("observed", "mean")).reset_index().assign(bin=lambda d: d["bin"].astype(str), transition_model=model)


def predict_fixed_horizon(coefficients, frame, design, horizon_months=24, model="monthly"):
    """Unconditional I/D/death risk at an explicitly defined follow-up horizon.

    Observed outcomes at this horizon must be supplied separately for validation;
    a four-year CLASS endpoint cannot be treated as a two-year observed label.
    """
    import numpy as np
    require_columns(frame, ["origin", "age"], "Fixed-horizon predictions")
    if int(horizon_months) != horizon_months or horizon_months <= 0:
        raise ValueError("horizon_months must be a positive integer")
    if model not in ("monthly", "ctmc"):
        raise ValueError("model must be monthly or ctmc")
    x = np.asarray(design, dtype=float)
    if x.shape[0] != len(frame) or not np.isfinite(x).all():
        raise ValueError("Finite aligned prediction design is required")
    state = np.zeros((len(frame), 3))
    origins = frame["origin"].to_numpy(int)
    if not np.isin(origins, [0, 1]).all():
        raise ValueError("Predictions require living origin states")
    state[np.arange(len(frame)), origins] = 1
    matrix = transition_matrix if model == "monthly" else ctmc_matrix
    for m in range(int(horizon_months)):
        current = x.copy()
        current[:, 1] = (frame["age"].to_numpy(float) - 60) / 10 + m / 120
        state = np.einsum("ni,nij->nj", state, matrix(coefficients, current))
    return state
