# Low-Radiance Cities and Functional Life Expectancy

Core analysis code for *Functional Life Expectancy for Low Radiance Urban Comparisons in Aging Populations*.

**Code authors and copyright holders:** Zhenchao Wu, Taikang Yao, Ning Shen.

## Analyses

| Modules | Computation | Supporting Methods |
|---|---|---|
| `clinical`, `clinical_fast` | Monthly three-state likelihood, municipality-cluster covariance, restricted functional years, joint coefficient draws and four-channel Shapley decomposition | 4–5 |
| `horizon` | Cumulative dependence at 6, 12, 18 and 24 months | 7 |
| `reference` | Joint-draw city references with service floors, reciprocal demand loss, CVaR and self-weight constraints | 8 |
| `sparse_reference`, `sparse_recovery` | Exact external-support enumeration and centered-CVaR numerical recovery for clinical-era references | 10, clinical-era formulation |
| `unit_invariance`, `allocation`, `inference` | Dimensionless policy composition, fixed-budget allocation, oracle regret and paired city bootstrap | 9 |
| `temporal` | Fixed external-reference trajectories and later-year persistence | 11 |
| `spatial`, `spatial_statistics` | Within-city optical classification, covariate matching and exact city sign-flip inference | 12 |

The package accepts prepared arrays and tables. Survey harmonization, geographic extraction and city-grouped prediction supply these inputs. Input fields, units, array ordering and task settings are described in [docs/INPUTS.md](docs/INPUTS.md). The joint-draw reference solvers implement the clinical-era formulation; the independent 2018 point-score formulation is described separately in Supporting Method 10.

## Installation

Use Python 3.10 or later:

```bash
python -m pip install -e .
lowradiance --help
python -m unittest discover -s tests -v
```

Dependencies are NumPy, SciPy, pandas and Numba. Linear programs use SciPy's HiGHS solver with one thread. The tested environment is recorded in [docs/VALIDATION.md](docs/VALIDATION.md) and `requirements.txt`. To install the tested versions on Python 3.12, run `python -m pip install -r requirements.txt` before installing the package.

## Workflow

1. Prepare observed intervals, the design matrix and municipality cluster keys.
2. Fit monthly transitions and retain the full joint coefficient covariance.
3. Project functional years and cumulative dependence with aligned joint draws.
4. Supply common candidate pools and service constraints for city comparisons.
5. Solve full or sparse references at the specified operating point.
6. Evaluate city need scores under a common budget and capacity bounds.
7. Evaluate fixed reference weights over time and match within-city cells.

```bash
lowradiance fit --intervals inputs/intervals.csv --arrays inputs/design.npz --output outputs/model.json
lowradiance fle --arrays inputs/projection.npz --output outputs/functional_years.json
lowradiance reference --input inputs/reference_problem.json --output outputs/reference.json
lowradiance sparse --input inputs/sparse_problem.json --output outputs/sparse.json
lowradiance allocation --table inputs/city_scores.csv --score-columns functional conventional --output outputs/allocation.json
lowradiance spatial --features inputs/cells.csv --arrays inputs/annual_cells.npz --radius 25000 --caliper .5 --output outputs/matches.json
lowradiance temporal --annual inputs/city_annual.csv --weights inputs/weights.csv --target-id target --output outputs/trajectory.json
```

These commands illustrate user-supplied filenames. `fit` returns coefficients and covariance; the projection input combines coefficients with the desired profile design. Python APIs additionally expose joint draws, Shapley decomposition, group-coverage allocation and paired bootstrap summaries. Output files are JSON; existing outputs require a new filename.

## Data sources

Research data are obtained separately from their providers. This repository distributes code and documentation; tests generate synthetic mathematical inputs.

- [CHARLS](https://charls.pku.edu.cn/en/) and [CLASS](https://class.ruc.edu.cn/): longitudinal survey records.
- [NASA Black Marble](https://viirsland.gsfc.nasa.gov/Products/NASA/BlackMarble.html): nighttime radiance.
- [GHSL](https://human-settlement.emergency.copernicus.eu/): settlement layers.
- [Hospital accessibility](https://doi.org/10.1038/s41597-024-03981-y).
- [Eldercare accessibility](https://doi.org/10.1038/s41597-026-07188-1) and its [dataset](https://doi.org/10.6084/m9.figshare.30478925).
- [National Bureau of Statistics](https://www.stats.gov.cn/english/) and municipal yearbooks: source-year service and population statistics.
- [Macrodatas China City Database](https://www.macrodatas.cn/article/1147473303), version 202603, original-data worksheet: independent 2018 city resources.
- [LandScan Global 2018](https://doi.org/10.48690/1524213): population supports.
- [Geofabrik](https://download.geofabrik.de/asia/china.html): OpenStreetMap roads.

Release years, population denominators and variable definitions follow the Supporting Methods. Data access follows each provider's terms.

## License

Copyright (c) 2026 Zhenchao Wu, Taikang Yao, Ning Shen.
Released under the [MIT License](LICENSE).
