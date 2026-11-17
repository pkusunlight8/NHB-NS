"""Convex city-reference optimization with explicit external mass."""
from __future__ import annotations
from dataclasses import dataclass,field,replace
from functools import lru_cache
from typing import Any,Mapping,Sequence
import hashlib,json
import numpy as np
import pandas as pd
from ._solver import linprog
from scipy import sparse
from .core import MissingInputError

@dataclass(frozen=True)
class OperatingPoint:
    gap: float | None = None
    external_mass: float | None = None
    gap_band: float = 0.0
    self_cap: float = 1.0


@dataclass
class ReferenceProblem:
    target_id: str
    donor_ids: Sequence[str]
    radiance: np.ndarray
    target_radiance: float
    services: np.ndarray
    target_services: np.ndarray
    donor_need_draws: np.ndarray | None = None  # B x donors
    target_need_draws: np.ndarray | None = None  # B
    context_distance: np.ndarray | None = None
    year: int | None = None
    pool: str = "individual"
    metadata: dict[str, Any] = field(default_factory=dict)
    donor_point_need: np.ndarray | None = None
    target_point_need: float | None = None
    linear_coefficients: np.ndarray | None = None  # m x donors, A @ w <= b
    linear_upper: np.ndarray | None = None
    linear_names: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.target_id = str(self.target_id)
        self.donor_ids = [str(value) for value in self.donor_ids]
        for name in ("radiance", "services", "target_services", "donor_need_draws", "target_need_draws", "context_distance", "donor_point_need", "linear_coefficients", "linear_upper"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=float))

    @property
    def self_index(self) -> int | None:
        ids = [str(x) for x in self.donor_ids]
        return ids.index(str(self.target_id)) if str(self.target_id) in ids else None

    def validate(self, need: bool, epsilon: float) -> None:
        n = len(self.donor_ids)
        if not n or len(set(map(str, self.donor_ids))) != n:
            raise ValueError("Donor IDs must form a nonempty unique list")
        if np.asarray(self.radiance).shape != (n,) or np.asarray(self.services).ndim != 2 or len(self.services) != n:
            raise ValueError("Radiance/service arrays do not align to donor IDs")
        if np.asarray(self.target_services).shape != (self.services.shape[1],):
            raise ValueError("Target service floors do not align")
        if not np.isfinite(self.target_radiance) or self.target_radiance <= epsilon:
            raise ValueError("Radiance gap requires positive target radiance")
        for array in (self.radiance, self.services, self.target_services):
            if not np.isfinite(array).all() or (np.asarray(array) < 0).any():
                raise ValueError("Radiance and service inputs must be finite and nonnegative")
        if (self.linear_coefficients is None) != (self.linear_upper is None):
            raise ValueError("Supply both extra linear coefficients and upper bounds")
        if self.linear_coefficients is not None:
            matrix, upper = self.linear_coefficients, self.linear_upper
            if matrix.ndim != 2 or matrix.shape[1] != n or upper.shape != (matrix.shape[0],):
                raise ValueError("Extra linear constraints must align to the ordered donor pool")
            if not np.isfinite(matrix).all() or not np.isfinite(upper).all():
                raise ValueError("Extra linear constraints require finite coefficients and bounds")
            if len(self.linear_names) != len(upper) or len(set(self.linear_names)) != len(upper):
                raise ValueError("Extra linear constraints require unique descriptive names")
        if need:
            if self.donor_need_draws is None or self.target_need_draws is None:
                raise MissingInputError("This demand method needs real predictions/draws for both target and donors")
            d, t = np.asarray(self.donor_need_draws), np.asarray(self.target_need_draws)
            if d.ndim != 2 or d.shape != (len(t), n) or not len(t):
                raise ValueError("Need draws must be B-by-donors with aligned target draws")
            if not np.isfinite(d).all() or not np.isfinite(t).all() or (d < 0).any() or (t <= epsilon).any():
                raise ValueError("Ratio loss needs positive target demand and finite nonnegative donor demand")
            if self.self_index is not None and not np.array_equal(d[:, self.self_index], t):
                raise ValueError("Self-reference must use exactly the same target draw in numerator and denominator")


@dataclass
class DecisionResult:
    target_id: str
    method: str
    rule: str
    operating_point: OperatingPoint
    donor_ids: list[str]
    weights: np.ndarray | None
    status: str
    solver_status: int
    message: str
    objective: float | None = None
    achieved_gap: float | None = None
    external_mass: float | None = None
    match_probability: float | None = None
    tail_loss: float | None = None
    maximum_constraint_violation: float | None = None
    year: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.status == "optimal" and self.weights is not None

    def external_weights(self, epsilon: float = 1e-8) -> dict[str, float]:
        if not self.feasible or self.external_mass is None or self.external_mass <= epsilon:
            return {}
        return {j: float(w / self.external_mass) for j, w in zip(self.donor_ids, self.weights)
                if str(j) != str(self.target_id) and w > 0}

    def record(self) -> dict[str, Any]:
        identity = {"target": self.target_id, "year": self.year, "method": self.method, "rule": self.rule,
                    "point": self.operating_point.__dict__,
                    **{key: self.metadata.get(key) for key in ("pool", "alpha", "delta", "objective", "risk_limit", "loss_mode", "estimand", "horizon_years", "population_basis", "units", "uncertainty_design_id", "experiment")}}
        if self.metadata.get("hospital_constraint_digest"):
            identity["hospital_constraint_digest"] = self.metadata["hospital_constraint_digest"]
        identity["morphology_constraint_digest"] = self.metadata.get("morphology_constraint_digest")
        identity["donor_ids"] = list(self.donor_ids)
        identity["weights"] = None if self.weights is None else np.asarray(self.weights).tolist()
        for key in ("training_population_hash", "prediction_population_hash", "forecast_cutoff", "outer_fold", "replicate_id", "design_id", "candidate_pool_digest", "decision_suite", "selection_estimand", "selection_units", "selection_horizon_years", "scenario_method"):
            identity[key] = self.metadata.get(key)
        decision_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        external = self.external_weights()
        return {**{key: self.metadata.get(key) for key in ("decision_suite", "selection_estimand", "selection_units", "selection_horizon_years", "selection_start_age", "selection_initial_state", "selection_context", "evaluation_estimand", "evaluation_units", "evaluation_horizon_years", "comparison_scope", "scenario_method")},
                "evaluation_mode": self.metadata.get("evaluation_mode", "point_identified"),
                "training_ascertainment_policy": self.metadata.get("training_ascertainment_policy", "complete_H_risk_set"),
                "weather_information_basis": self.metadata.get("weather_information_basis", "not_adjusted"),
                "decision_id": decision_id, "target_site": self.target_id, "year": self.year, "method": self.method,
                "rule": self.rule, "gap_requested": self.operating_point.gap,
                "external_mass_requested": self.operating_point.external_mass,
                "gap_band": self.operating_point.gap_band, "self_cap": self.operating_point.self_cap,
                "risk_limit": self.metadata.get("risk_limit"), "objective_type": self.metadata.get("objective"),
                "alpha": self.metadata.get("alpha"), "delta": self.metadata.get("delta"), "pool": self.metadata.get("pool"),
                "loss_mode": self.metadata.get("loss_mode"),
                "uncertainty_design_id": self.metadata.get("uncertainty_design_id"),
                "frozen_at": self.metadata.get("verified_frozen_at"),
                "freeze_source_reference": self.metadata.get("freeze_source_reference"),
                "forecast_cutoff": self.metadata.get("forecast_cutoff"),
                "decision_timing_basis": self.metadata.get("decision_timing_basis", "unspecified"),
                "hospital_constraint_digest": self.metadata.get("hospital_constraint_digest"),
                "eldercare_constraint_digest": self.metadata.get("eldercare_constraint_digest"),
                "eldercare_constraint_scope": self.metadata.get("eldercare_constraint_scope"),
                "spatial_constraint_scope": self.metadata.get("eldercare_constraint_scope") or self.metadata.get("hospital_constraint_scope"),
                "morphology_constraint_digest": self.metadata.get("morphology_constraint_digest"),
                "morphology_constraint_scope": self.metadata.get("morphology_constraint_scope", "disabled"),
                "status": self.status, "solver_status": self.solver_status,
                "message": self.message, "objective": self.objective,
                "achieved_gap": self.achieved_gap, "external_mass": self.external_mass,
                "self_weight": 1 - self.external_mass if self.external_mass is not None else None,
                "active_external_references": len(external) if self.feasible else None,
                "effective_external_donors": 1 / sum(v * v for v in external.values()) if external else None,
                "model_internal_match_probability": self.match_probability,
                "model_internal_tail_loss": self.tail_loss,
                "maximum_constraint_violation": self.maximum_constraint_violation}


@lru_cache(maxsize=4096)
def _cached_physical_feasibility(key):
    """Solve only the shared n-weight feasibility problem, independent of need."""
    (radiance, target_radiance, services, target_services, extra, upper_extra,
     self_index, gap, external_mass, gap_band, self_cap, tolerance) = key
    radiance = np.asarray(radiance, float)
    n = len(radiance)
    services = np.asarray(services, float).reshape(n, len(target_services))
    inequalities = [-services.T]
    upper = [-np.asarray(target_services, float)]
    if extra:
        inequalities.append(np.asarray(extra, float).reshape(len(upper_extra), n))
        upper.append(np.asarray(upper_extra, float))
    equalities = [np.ones(n)]
    equal_rhs = [1.]
    if external_mass is not None:
        row = np.ones(n)
        if self_index is not None:
            row[self_index] = 0.
        equalities.append(row); equal_rhs.append(external_mass)
    if gap is not None:
        ratio = radiance / target_radiance
        if gap_band == 0:
            equalities.append(ratio); equal_rhs.append(1 - gap)
        else:
            inequalities.extend([ratio[None, :], -ratio[None, :]])
            upper.extend([np.array([1 - gap + gap_band]), np.array([-(1 - gap - gap_band)])])
    bounds = [(0., 1.)] * n
    if self_index is not None:
        bounds[self_index] = (0., self_cap)
    aub, bub = sparse.csr_matrix(np.vstack(inequalities)), np.concatenate(upper)
    aeq, beq = np.vstack(equalities), np.asarray(equal_rhs)
    result = linprog(np.zeros(n), A_ub=aub,
        b_ub=bub, A_eq=sparse.csr_matrix(aeq), b_eq=beq,
        bounds=bounds, method="highs", options={"primal_feasibility_tolerance": tolerance, "dual_feasibility_tolerance": tolerance})
    unique = bool(result.success and np.linalg.matrix_rank(aeq) == n and np.linalg.cond(aeq) < 1e10)
    residual = None
    if result.success:
        residual = max(float(np.max(np.abs(aeq @ result.x - beq), initial=0.)),
                       float(np.max(aub @ result.x - bub, initial=0.)),
                       float(max(0., -result.x.min(), result.x.max() - 1.)))
        if self_index is not None:
            residual = max(residual, float(max(0., result.x[self_index] - self_cap)))
    return int(result.status), str(result.message), tuple(result.x) if result.success else None, unique, residual


def physical_feasibility(problem, point, tolerance):
    """Cache by actual physical arrays and operating point, with no label input."""
    key = (tuple(problem.radiance), float(problem.target_radiance), tuple(problem.services.ravel()),
           tuple(problem.target_services),
           tuple(problem.linear_coefficients.ravel()) if problem.linear_coefficients is not None else (),
           tuple(problem.linear_upper) if problem.linear_upper is not None else (),
           problem.self_index, point.gap, point.external_mass, point.gap_band, point.self_cap, float(tolerance))
    return _cached_physical_feasibility(key)


def reciprocal_loss(ratio: np.ndarray, delta: float = 0.05) -> np.ndarray:
    if delta <= 0:
        raise ValueError("delta must be positive")
    ratio = np.asarray(ratio, dtype=float)
    return np.maximum((ratio - 1) / delta, (1 - ratio) / (1 - 1 / (1 + delta)))


def empirical_cvar(losses: np.ndarray, alpha: float = 0.8) -> float:
    """Exact equally weighted empirical CVaR, including a fractional tail draw."""
    x = np.sort(np.asarray(losses, dtype=float))
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 <= alpha < 1:
        raise ValueError("Invalid empirical CVaR inputs")
    mass = (1 - alpha) * len(x)
    whole = int(np.floor(mass))
    fraction = mass - whole
    total = x[-whole:].sum() if whole else 0.0
    if fraction > 1e-12:
        total += fraction * x[-whole - 1]
    return float(total / mass)


def solve_reference(problem: ReferenceProblem, *, method: str = "FRE", rule: str = "cvar",
                    point: OperatingPoint | None = None, objective: str = "mismatch",
                    risk_limit: float | None = None, alpha: float = 0.8,
                    delta: float = 0.05, epsilon: float = 1e-8,
                    solver_tolerance: float = 1e-8, tie_tolerance: float = 1e-8,
                    lexicographic: bool = True,
                    loss_mode: str = "reciprocal_caliper") -> DecisionResult:
    """Solve a point/expected/CVaR epigraph LP using SciPy HiGHS.

    ``objective='radiance', risk_limit=1`` reproduces the original FRE problem.
    Fair fixed-gap/mass experiments minimise mismatch; A0 minimises context
    distance because it deliberately has no demand score. A1/A2 callers must
    supply independently trained demand predictions, never FLE-generated labels.
    """
    if method not in {"A0", "A1", "A2", "FRE"} or rule not in {"point", "expected", "cvar"}:
        raise ValueError("Unknown information arm or risk rule")
    if not 0 <= alpha < 1 or delta <= 0 or objective not in {"mismatch", "radiance"}:
        raise ValueError("Invalid risk level, caliper or objective")
    if loss_mode not in {"reciprocal_caliper", "absolute_relative"}:
        raise ValueError("Unknown demand loss")
    point = point or OperatingPoint()
    if point.gap is not None and not 0 <= point.gap <= 1:
        raise ValueError("Requested gap must be between zero and one")
    if point.external_mass is not None and not 0 <= point.external_mass <= 1:
        raise ValueError("External mass must be between zero and one")
    if not 0 <= point.self_cap <= 1 or point.gap_band < 0:
        raise ValueError("Invalid self cap or common gap band")
    use_need = method != "A0"
    problem.validate(use_need, epsilon)
    if method in {"A1", "A2"}:
        if problem.metadata.get("label_origin") != "observed_training_outcomes":
            raise MissingInputError(f"{method} must use observed training outcomes; set verified label_origin metadata after training")
    n = len(problem.donor_ids)
    if use_need:
        d = np.asarray(problem.donor_need_draws, float)
        t = np.asarray(problem.target_need_draws, float)
        if rule == "point":
            if problem.donor_point_need is not None or problem.target_point_need is not None:
                if problem.donor_point_need is None or problem.target_point_need is None:
                    raise ValueError("Supply both target and donor point estimates")
                point_d = np.asarray(problem.donor_point_need, float)
                if point_d.shape != (n,) or not np.isfinite(point_d).all() or (point_d < 0).any() or not np.isfinite(problem.target_point_need) or problem.target_point_need <= epsilon:
                    raise ValueError("Invalid point-estimate needs")
                d, t = point_d[None, :], np.array([problem.target_point_need])
            else:
                d, t = d.mean(axis=0, keepdims=True), np.array([t.mean()])
        b = len(t)
    else:
        b = 0
    context = np.asarray(problem.context_distance if problem.context_distance is not None else np.zeros(n), float)
    if context.shape != (n,) or not np.isfinite(context).all():
        raise ValueError("Context distances must align with donors")
    if not use_need and risk_limit is not None:
        raise ValueError("A0 has no demand risk constraint")
    metadata = {**problem.metadata, "pool": problem.pool, "alpha": alpha, "delta": delta,
                "epsilon": epsilon, "objective": objective, "risk_limit": risk_limit,
                "loss_mode": loss_mode,
                "extra_linear_constraints": list(problem.linear_names),
                "tie_break": "primary then context then donor-ID lexicographic" if lexicographic else "primary then context",
                "validation_type": "model_internal"}
    if problem.linear_coefficients is not None:
        signature = {"donors": list(problem.donor_ids), "names": list(problem.linear_names),
                     "coefficients": problem.linear_coefficients.tolist(), "upper": problem.linear_upper.tolist()}
        metadata["hospital_constraint_digest"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
    physical_status, physical_message, physical_weights, unique_physical, physical_residual = physical_feasibility(problem, point, solver_tolerance)
    if physical_status == 2:
        return DecisionResult(str(problem.target_id), method, rule, point, list(map(str, problem.donor_ids)), None,
                              "infeasible", 2, physical_message, year=problem.year, metadata=metadata)
    if unique_physical:
        w = np.asarray(physical_weights, float)
        risk = 0.
        if use_need:
            ratio = d @ w / t
            losses = np.abs(ratio - 1) / delta if loss_mode == "absolute_relative" else reciprocal_loss(ratio, delta)
            risk = empirical_cvar(losses, alpha) if rule == "cvar" else float(losses.mean())
        if risk_limit is not None:
            limit = float(risk_limit)
            if not np.isfinite(limit):
                raise ValueError("Risk limit must be finite")
            if risk > limit + solver_tolerance:
                return DecisionResult(str(problem.target_id), method, rule, point, list(map(str, problem.donor_ids)), None,
                    "infeasible", 2, "The unique physical portfolio exceeds the declared risk limit", year=problem.year, metadata=metadata)
            physical_residual = max(physical_residual, max(0., risk - limit))
        value = float(np.dot(w, problem.radiance) / problem.target_radiance) if objective == "radiance" else float(context @ w) if method == "A0" else risk
        match = tail_loss = None
        if problem.donor_need_draws is not None and problem.target_need_draws is not None:
            problem.validate(True, epsilon)
            ratio = problem.donor_need_draws @ w / problem.target_need_draws
            match = float(np.mean((ratio >= 1 / (1 + delta) - solver_tolerance) & (ratio <= 1 + delta + solver_tolerance)))
            tail_loss = empirical_cvar(reciprocal_loss(ratio, delta), alpha)
        external = float(1 - w[problem.self_index]) if problem.self_index is not None else float(w.sum())
        metadata["solver_reduction"] = "full_rank_physical_equalities_determine_unique_weights"
        return DecisionResult(str(problem.target_id), method, rule, point, list(map(str, problem.donor_ids)), w,
            "optimal", 0, "Unique portfolio determined by the unchanged physical equalities", value,
            float(1 - np.dot(w, problem.radiance) / problem.target_radiance), external, match, tail_loss,
            physical_residual, problem.year, metadata)
    risk_cvar = use_need and rule == "cvar"
    total = n + b + (1 + b if risk_cvar else 0)
    threshold = n + b
    excess_start = threshold + 1
    rows: list[sparse.csr_matrix] = []
    bounds_rhs: list[float] = []
    equalities: list[np.ndarray] = []
    equal_rhs: list[float] = []
    def upper(coeff: np.ndarray, rhs: float) -> None:
        rows.append(sparse.csr_matrix(coeff.reshape(1, -1))); bounds_rhs.append(float(rhs))
    def equality(coeff: np.ndarray, rhs: float) -> None:
        equalities.append(coeff); equal_rhs.append(float(rhs))
    csum = np.zeros(total); csum[:n] = 1; equality(csum, 1)
    for k in range(problem.services.shape[1]):
        row = np.zeros(total); row[:n] = -problem.services[:, k]
        upper(row, -problem.target_services[k])
    if problem.linear_coefficients is not None:
        for coefficients, bound in zip(problem.linear_coefficients, problem.linear_upper):
            row = np.zeros(total); row[:n] = coefficients
            upper(row, bound)
    self_index = problem.self_index
    if point.external_mass is not None:
        row = np.zeros(total); row[:n] = 1
        if self_index is not None:
            row[self_index] = 0
        equality(row, point.external_mass)
    if point.gap is not None:
        row = np.zeros(total); row[:n] = problem.radiance / problem.target_radiance
        if point.gap_band == 0:
            equality(row, 1 - point.gap)
        else:
            upper(row, 1 - point.gap + point.gap_band)
            upper(-row, -(1 - point.gap - point.gap_band))
    cost_risk = np.zeros(total)
    if use_need:
        low = delta if loss_mode == "absolute_relative" else 1 - 1 / (1 + delta)
        tail = sparse.csr_matrix((b, total - n - b))
        scenario_identity = sparse.eye(b, format="csr")
        upper_need = sparse.hstack([sparse.csr_matrix(d / t[:, None] / delta), -scenario_identity, tail], format="csr")
        lower_need = sparse.hstack([sparse.csr_matrix(-d / t[:, None] / low), -scenario_identity, tail], format="csr")
        # Preserve the original upper/lower row order for numerical tie-breaking.
        interleave = np.column_stack([np.arange(b), np.arange(b) + b]).ravel()
        rows.append(sparse.vstack([upper_need, lower_need], format="csr")[interleave])
        bounds_rhs.extend(np.tile([1 / delta, -1 / low], b))
        if risk_cvar:
            cost_risk[threshold] = 1
            cost_risk[excess_start:] = 1 / ((1 - alpha) * b)
            rows.append(sparse.hstack([sparse.csr_matrix((b, n)), scenario_identity,
                                      sparse.csr_matrix(-np.ones((b, 1))), -scenario_identity], format="csr"))
            bounds_rhs.extend(np.zeros(b))
        else:
            cost_risk[n:n+b] = 1 / b
        if risk_limit is not None:
            upper(cost_risk.copy(), risk_limit)
    elif risk_limit is not None:
        raise ValueError("A0 has no demand risk constraint")
    context = np.asarray(problem.context_distance if problem.context_distance is not None else np.zeros(n), float)
    if context.shape != (n,) or not np.isfinite(context).all():
        raise ValueError("Context distances must align with donors")
    cost = cost_risk.copy()
    if objective == "radiance":
        cost[:] = 0; cost[:n] = problem.radiance / problem.target_radiance
    elif method == "A0":
        cost[:n] = context
    variable_bounds = [(0, 1)] * n + [(0, None)] * (total - n)
    if self_index is not None:
        variable_bounds[self_index] = (0, point.self_cap)
    aub = sparse.vstack(rows, format="csr") if rows else None
    bub = np.asarray(bounds_rhs) if rows else None
    aeq = sparse.csr_matrix(np.vstack(equalities)); beq = np.asarray(equal_rhs)
    options = {"primal_feasibility_tolerance": solver_tolerance, "dual_feasibility_tolerance": solver_tolerance}
    result = linprog(cost, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq,
                     bounds=variable_bounds, method="highs", options=options)
    metadata = {**problem.metadata, "pool": problem.pool, "alpha": alpha, "delta": delta,
                "epsilon": epsilon, "objective": objective, "risk_limit": risk_limit,
                "loss_mode": loss_mode,
                "extra_linear_constraints": list(problem.linear_names),
                "tie_break": "primary then context then donor-ID lexicographic" if lexicographic else "primary then context",
                "validation_type": "model_internal"}
    if problem.linear_coefficients is not None:
        signature = {"donors": list(problem.donor_ids), "names": list(problem.linear_names),
                     "coefficients": problem.linear_coefficients.tolist(), "upper": problem.linear_upper.tolist()}
        metadata["hospital_constraint_digest"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
    if not result.success:
        return DecisionResult(str(problem.target_id), method, rule, point, list(map(str, problem.donor_ids)), None,
                              "infeasible" if result.status == 2 else "solver_failed", int(result.status),
                              str(result.message), year=problem.year, metadata=metadata)
    primary_optimum = float(result.fun)
    # Every tie-break is itself an LP under the same primal constraints.
    extra_rows = [cost.copy()]; extra_rhs = [primary_optimum + tie_tolerance]
    context_cost = np.zeros(total); context_cost[:n] = context
    def tie_solve(tie_cost: np.ndarray):
        aug = sparse.vstack([aub, sparse.csr_matrix(np.vstack(extra_rows))], format="csr") if aub is not None else sparse.csr_matrix(np.vstack(extra_rows))
        return linprog(tie_cost, A_ub=aug, b_ub=np.r_[bub if bub is not None else [], extra_rhs],
                       A_eq=aeq, b_eq=beq, bounds=variable_bounds, method="highs", options=options)
    second = tie_solve(context_cost)
    if second.success:
        result = second
        extra_rows.append(context_cost.copy()); extra_rhs.append(float(second.fun) + tie_tolerance)
        if lexicographic:
            for j in sorted(range(n), key=lambda k: str(problem.donor_ids[k]))[:-1]:
                lex_cost = np.zeros(total); lex_cost[j] = -1
                candidate = tie_solve(lex_cost)
                if not candidate.success:
                    metadata["tie_break_warning"] = str(candidate.message)
                    break
                result = candidate
                row = np.zeros(total); row[j] = -1
                extra_rows.append(row); extra_rhs.append(float(-candidate.x[j] + tie_tolerance))
    else:
        metadata["tie_break_warning"] = str(second.message)
    w = np.asarray(result.x[:n])
    residuals = [float(np.max(np.abs(aeq @ result.x - beq)))]
    residuals.extend([float(max(0, -w.min())), float(max(0, w.max() - 1))])
    if self_index is not None:
        residuals.append(float(max(0, w[self_index] - point.self_cap)))
    if aub is not None:
        residuals.append(float(max(0, np.max(aub @ result.x - bub))))
    external = float(1 - w[self_index]) if self_index is not None else float(w.sum())
    match = tail = None
    if problem.donor_need_draws is not None and problem.target_need_draws is not None:
        problem.validate(True, epsilon)
        ratio = np.asarray(problem.donor_need_draws) @ w / np.asarray(problem.target_need_draws)
        match = float(np.mean((ratio >= 1 / (1 + delta) - solver_tolerance) & (ratio <= 1 + delta + solver_tolerance)))
        tail = empirical_cvar(reciprocal_loss(ratio, delta), alpha)
    return DecisionResult(str(problem.target_id), method, rule, point, list(map(str, problem.donor_ids)), w,
                          "optimal", int(result.status), str(result.message), primary_optimum,
                          float(1 - np.dot(w, problem.radiance) / problem.target_radiance), external,
                          match, tail, max(residuals), problem.year, metadata)


def solve_fair_comparison(problems: Mapping[str, ReferenceProblem], *, gap: float,
                          external_mass: float, rules: Mapping[str, str] | None = None,
                          common_relaxation: float = 0.01, **kwargs: Any) -> list[DecisionResult]:
    """Never relax one method alone; all arms receive the identical gap band."""
    if not problems:
        raise ValueError("At least one method is required")
    first = next(iter(problems.values()))
    for p in problems.values():
        if (list(p.donor_ids) != list(first.donor_ids) or p.target_id != first.target_id
                or p.target_radiance != first.target_radiance or p.pool != first.pool or p.year != first.year
                or not np.array_equal(p.radiance, first.radiance)
                or not np.array_equal(p.services, first.services)
                or not np.array_equal(p.target_services, first.target_services)
                or not np.array_equal(p.context_distance, first.context_distance)
                or not np.array_equal(p.linear_coefficients, first.linear_coefficients)
                or not np.array_equal(p.linear_upper, first.linear_upper)
                or list(p.linear_names) != list(first.linear_names)):
            raise ValueError("Fair comparison requires the same ordered candidate pool and physical/service constraints")
    rules = rules or {}
    def solve_band(band: float) -> list[DecisionResult]:
        return [solve_reference(p, method=m, rule=rules.get(m, "cvar"),
                                point=OperatingPoint(gap, external_mass, band), **kwargs)
                for m, p in problems.items()]
    exact = solve_band(0)
    if all(r.feasible for r in exact) or common_relaxation <= 0:
        return exact
    relaxed = solve_band(common_relaxation)
    for r in relaxed:
        r.metadata["exact_gap_statuses"] = {x.method: x.status for x in exact}
        r.metadata["common_gap_relaxation"] = common_relaxation
    return relaxed


def portfolio_total_variation(a: DecisionResult, b: DecisionResult) -> float:
    va, vb = a.external_weights(), b.external_weights()
    if not va or not vb:
        return float("nan")
    return float(sum(abs(va.get(k, 0) - vb.get(k, 0)) for k in va.keys() | vb.keys()) / 2)


def donor_change(a: DecisionResult, b: DecisionResult, threshold: float = 0.10) -> dict[str, Any]:
    va, vb = a.external_weights(), b.external_weights()
    tv = portfolio_total_variation(a, b)
    dominant_a = min(va, key=lambda k: (-va[k], k)) if va else None
    dominant_b = min(vb, key=lambda k: (-vb[k], k)) if vb else None
    return {"external_availability_changed": bool(va) != bool(vb), "dominant_donor_changed": dominant_a != dominant_b,
            "external_portfolio_tv": tv,
            "material_change": bool(va) != bool(vb) or (bool(va) and (dominant_a != dominant_b or tv >= threshold))}


def common_feasible(results: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    keys = [k for k in ["target_site", "year", "experiment", "rule", "alpha", "delta", "gap_requested", "external_mass_requested", "gap_band", "self_cap", "risk_limit", "hospital_constraint_digest", "eldercare_constraint_digest", "eldercare_constraint_scope", "spatial_constraint_scope", "morphology_constraint_digest", "morphology_constraint_scope"] if k in results]
    if not keys or "method" not in results or "status" not in results:
        raise ValueError("Need operating-point keys, method and solver status")
    selected = results[results.method.isin(methods)].copy()
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Common feasibility requires distinct nonempty method names")
    if selected.duplicated(keys + ["method"]).any():
        raise ValueError("Repeated operating-point/method rows cannot define common feasibility")
    good = selected.groupby(keys, dropna=False).agg(
        n_methods=("method", "nunique"), all_optimal=("status", lambda x: bool((x == "optimal").all())))
    eligible = good[(good.n_methods == len(set(methods))) & good.all_optimal].reset_index()[keys]
    return selected.merge(eligible, on=keys, how="inner", validate="many_to_one")


def trace_pareto(problem: ReferenceProblem, *, method: str = "FRE",
                 self_caps: Sequence[float] = (1., .75, .5, 0.),
                 risk_limits: Sequence[float] = (.5, 1., 2.),
                 **kwargs: Any) -> list[DecisionResult]:
    """Trace risk/radiance/self trade-offs, retaining infeasible operating points."""
    results = []
    for cap in self_caps:
        for limit in risk_limits:
            if limit < 0:
                raise ValueError("Risk limits must be nonnegative")
            result = solve_reference(problem, method=method, objective="radiance", risk_limit=limit,
                                     point=OperatingPoint(self_cap=cap), **kwargs)
            result.metadata.update(experiment="self_cap_risk_pareto", risk_limit=limit)
            results.append(result)
    return results


def benchmark_regret(selected: DecisionResult, problem: ReferenceProblem, outcomes: Mapping[str, float],
                     *, epsilon: float = 1e-8, **kwargs: Any) -> dict[str, Any]:
    """Retrospective external-only oracle under exactly the same feasible set."""
    if not selected.feasible or not selected.external_weights(epsilon):
        return {"regret": np.nan, "reason": "no_external_decision"}
    required = set(problem.donor_ids) | {problem.target_id}
    missing = required - outcomes.keys()
    if missing:
        raise MissingInputError(f"Independent oracle outcomes missing for {len(missing)} target/donor supports")
    target = float(outcomes[problem.target_id])
    if target <= epsilon:
        return {"regret": np.nan, "reason": "zero_independent_target_demand"}
    oracle_problem = replace(problem, donor_need_draws=np.array([[outcomes[j] for j in problem.donor_ids]]),
                             target_need_draws=np.array([target]), donor_point_need=None, target_point_need=None,
                             metadata={"oracle": "retrospective_evaluation_only"})
    # A fixed e makes full-portfolio and external-only absolute loss differ only
    # by a positive constant. Preserve the selected achieved ambition even when
    # evaluating an originally unconstrained-radiance solution.
    oracle_point = replace(selected.operating_point, external_mass=selected.external_mass)
    if oracle_point.gap is None:
        oracle_point = replace(oracle_point, gap=selected.achieved_gap, gap_band=0.)
    oracle = solve_reference(oracle_problem, method="FRE", rule="point", point=oracle_point,
                             loss_mode="absolute_relative", **kwargs)
    if not oracle.feasible:
        return {"regret": np.nan, "reason": "oracle_infeasible"}
    loss = lambda r: abs(sum(v * outcomes[j] for j, v in r.external_weights().items()) - target) / target
    return {"regret": float(loss(selected) - loss(oracle)), "selected_loss": float(loss(selected)),
            "oracle_loss": float(loss(oracle)), "reason": "retrospective_benchmark_selection"}
