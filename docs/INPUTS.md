# Input contracts

Array positions and table rows must agree exactly.
Use floating-point arrays in their documented units and preserve missingness.

## Clinical intervals and functional projections

The interval CSV has one observed interval per row. Required columns are `origin`
(0 independent, 1 dependent), `destination` (0 independent, 1 dependent, 2 death),
`months`, `death_lower`, `death_upper`, `age` and `municipality_id`. Origins must be
living states. Age is attained age in years at interval start. Living endpoints use
integer `months`; deaths use a verified integer-month interval `(death_lower,
death_upper]`. Death windows retain the observed lower and upper bounds.

The design NPZ contains `design`, shape `(n_intervals, p)`. The default p=10 order is
intercept, (age−60)/10, female, rural residence, low education, standardized service
factor, brightness, gross domestic product, retail sales, and urbanization. Apply
the manuscript's harmonization and training-only preprocessing before supplying
this matrix. The intercept is one. The fit uses municipality clusters and returns
convergence, score and information diagnostics alongside coefficients/covariance.

Projection NPZ contains `coefficients` of shape `(4,p)` and `design` of shape
`(n_profiles,p)`. Channel order is independent→dependent, independent→death,
dependent→independent, dependent→death. `clinical.occupancy` advances age each month;
other profile covariates remain fixed. Rewards use monthly trapezoidal integration
and are reported in years. `coefficient_draws` requires the full joint covariance.
The two living-state columns of `restricted_expectancy` are independent and
dependent years. For `horizon.fre_windows`, `beta` has shape `(n_draws,4,p)`, `x`
contains baseline profiles and `origin` is an aligned integer living-state vector.
The returned cumulative windows are 6/12/18/24 months, in years.

## Convex reference problem

The reference JSON contains `problem`, `operating_point` and `options` objects.
`problem` maps directly to `ReferenceProblem`:

| Field | Definition |
|---|---|
| `target_id`, `donor_ids` | One target string and a unique ordered list of candidates; include the target for self-reference |
| `radiance`, `target_radiance` | Aligned nonnegative candidate radiances and positive target radiance in the same unit |
| `services`, `target_services` | Candidate-by-service matrix and matching target service floors, using common units and denominators |
| `donor_need_draws`, `target_need_draws` | Scenario-by-donor matrix and aligned positive target scenario vector |
| `context_distance` | One nonnegative context dissimilarity per candidate |
| `donor_point_need`, `target_point_need` | Point estimates for comparisons requiring point-level predictions |
| `year`, `pool`, `metadata` | Optional task descriptors |

The target's need column must equal `target_need_draws` exactly. Independent
shuffling of donor and target scenarios breaks the joint uncertainty design.
Candidate arrays and service/context constraints must be the same across arms.
`operating_point` accepts `gap`, `external_mass`, `gap_band` and `self_cap`.
`options` accepts the named arguments of `solve_reference`, including `method`,
`rule`, `alpha`, `delta`, `objective`, `risk_limit` and `loss_mode`.

The short task uses a common requested mixed radiance gap of 0.2, fixed external
mass 0.5 and the previously determined common gap band. It minimizes demand loss.
The long task minimizes radiance with the conditional-value-at-risk limit of 1,
alpha 0.8 and reciprocal caliper delta 0.05. These settings define separate short- and long-horizon tasks.

## Exact sparse problem

The sparse JSON contains `problem`, `arm`, `k` and `self_cap`. `problem` uses the
same ordered candidate information with keys `ids`, `si` (self index), `radiance`,
`target_radiance`, `services`, `target_services`, `distance`, `d` (scenario-by-donor),
`t` (target scenarios), `task`, and `band`. `task` is `short_24_month` or
`long_30_year`; `band` is required in the short task. `arm` is `FRE_long` for the
long-horizon formulation or one of the stated short-horizon information arms.
K is the maximum number of external supports. Self-only solutions are possible;
count positive external mass explicitly. A baseline shortcut is available only
through the Python API and is valid only for a verified global unrestricted optimum.
The CLI always enumerates supports without that shortcut.

The sparse solver's diagnostic `describe` helper additionally expects task
metadata (`pool`, `target_site`, `year`, `outer_fold`) and point need values.

## Allocation and independent care evaluation

City-score CSV has exactly one row per independent city, a unique `city_id`,
positive `baseline_mass`, nonnegative observed `event_mass`, and the requested
nonnegative score columns. The source evaluation retains all planned cities,
including zero-event cities. Unknown outcomes remain missing during endpoint construction.

Every candidate score is normalized across the common city roster before composing
fold-selected conventional rules. `unit_invariance.correct_predictions` implements
the person-to-city version of this operation. Its input requires `city_key`,
`baseline_mass`, `class_fold`, `selected_traditional_rule`,
`selected_traditional_risk` and one prediction column for every supplied rule.
It accepts an explicit ordered `rules` list and an optional frozen city-by-rule
normalized-share table. The selected-rule allocation is invariant to positive rescaling of a candidate prediction column.

The common budget is one, city upper bounds are twice baseline shares, and the
loss is summed positive unmet need share. Regret subtracts the oracle loss under
the same bounds. `evaluate_city_table` resamples cities and reprojects allocations
for every draw; `paired_summary` uses paired differences and the centered two-sided
bootstrap p value. Apply the multiplicity families defined in Supporting Method 9 to the resulting endpoint probabilities. Group-coverage scenarios use `fair_project` and `fair_oracle`.

## Within-city matching

The features CSV has one row per fixed 1-km cell, aligned to the annual arrays.
Required fields are `radius_m`, `population_total2020`,
`population65plus2020`, their `_coverage` fields,
`hospital_drive_minutes2020` and its `_coverage`,
`elder_total_beds_per10002025` and its `_coverage`,
`nearest_bus_stop_m2024`, and four baseline climate summaries:
`tcc_baseline2012_2021`, `t2m_baseline2012_2021`,
`tp_baseline2012_2021`, `sde_baseline2012_2021`.
Coverage fields are fractions in [0,1]. Source years remain distinct.
The eldercare accessibility product retains its provider denominator of residents
aged 60 years and older; the separate population-support variable counts people
aged 65 years and older. Each quantity retains its own age denominator.

The NPZ contains `radiance` and `coverage`, each shape `(14,n_cells)` in year order
2012–2025. `coverage` is the valid-area fraction for the annual optical product.
Persistent low cells are at or below the within-city annual 25th percentile in at
least 80% of valid definition years; controls are at or above the annual median.
At least eight definition years and the stated population/source coverage are
required. Tied membership is removed from both groups. Covariates are standardized
over the pooled complete low/control candidate set. The matching cost is Euclidean
distance, with a maximum coordinatewise standardized difference caliper. Matching
is one-to-one without replacement using dummy unmatched alternatives.

Compute each city's mean paired outcome difference first. Across-city summaries
use equal city weights, t intervals and exact city sign flips as described in the
Supplementary Methods. Cities are the independent units for pooled uncertainty; intervals require at least two cities. `exact_p` enumerates 2^n
sign assignments and is intended for the study's small city sets.

## Fixed-reference temporal evaluation

Annual CSV uses unique `city_id`, `year`, `radiance` and boolean `valid` fields.
Weights CSV uses unique `donor_id` and numeric `weight`, and may include self.
Evaluate each policy/reference separately. Freeze identities and positive weights
at the selection year. Exclude self, retain every strictly positive external
component and normalize external weights to one. A year is valid only when target
and every required donor have valid observations and target radiance is positive.

The external gap is 1 − external radiance / target radiance. The population-support
fixed-2018 rule marks a reference stable if at least five valid later years are
available and the external prototype is lower in at least 80% of those years.
This rule is distinct from the long-horizon unique-pair and cross-radius summaries;
those require their own prespecified minimum-year denominators.
