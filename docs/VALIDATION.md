# Numerical validation

Version 0.2.0 was tested on 17 November 2026 with Python 3.12.10, NumPy 2.5.5, SciPy 1.18.1, pandas 3.1.1 and Numba 0.68.1. Exact runtime dependencies are listed in `requirements.txt`.

All 18 tests passed using generated mathematical inputs.

## Numerical identities

- Transition probabilities sum to one and death is absorbing.
- Accelerated likelihoods and scores agree with the interval kernel and finite differences.
- Cumulative dependence agrees with restricted functional-time integration.
- Four-channel Shapley contributions sum to the profile contrast.
- Full and sparse reference solvers recover known optima and identify an infeasible self cap.
- Budget projection preserves total capacity and is invariant to score units.
- Selected-rule composition is invariant to candidate prediction units.
- City bootstrap accounting retains zero-event replicates in the attempted-draw denominator.
- Spatial matching uses each control once and removes tied optical classifications.
- Temporal evaluation requires all positive-weight donors and retains the full annual calendar.
- Exact city sign-flip probabilities agree with known cases.

## Command-line workflows

All seven commands load their help and execute with synthetic inputs. Tests fit a monthly model with cluster covariance, project state occupancy, solve full and sparse references, evaluate two allocation policies, match cells and evaluate annual reference persistence. Missing columns, misaligned rows and nonfinite clinical design values fail before writing results. Existing output files remain unchanged when an output name is reused.

Run the suite after installation:

```bash
python -m unittest discover -s tests -v
```
