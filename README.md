# Core analysis code

This repository contains the custom code behind the main quantitative results of *Nighttime urban infrastructure and inequalities in independent living in China*. It is a focused analysis release.

## Contents

- `src/02_prepare_health.py` and `src/03_fit_imach_likelihood.py`: observed-window health-state preparation and monthly multistate likelihood, for licensed participant records.
- `src/07_expectancy.py`, `src/08_native_imach.py` and `src/11_verify_native.py`: restricted state expectancies, exact four-channel allocation, official IMaCh conversion and numerical verification.
- `src/01_service_prediction.py`: nested municipality-held-out prediction, training-fold PCA sensitivity analysis, boosting and TreeSHAP.
- `src/05_benchmark_extensions.py` and `src/09_need_allocation.py`: service-preserving city references, care-need standardisation and uncertainty propagation.
- `data/`: selected analysis columns from the Article's aggregate city and cohort-composition Source Data, two fitted coefficient/covariance matrices and native IMaCh expectancy text files. No participant rows are included; `data/README.md` describes provenance and coding.

## Requirements and public reproduction

Use Python 3.12. Install the pinned packages with `python -m pip install -r requirements.txt`. Run the core public analysis from the repository root:

```bash
python run.py public --output outputs
```

The command recomputes both clinical endpoints from published parameters, verifies restricted expectancies against native IMaCh output, fits the held-out service prediction models, and calculates the observed and need-matched city comparisons. Outputs and a `verification.json` summary are written to `outputs/`. The run uses seed `20260921`; CPU time is dominated by the 2,000 clinical coefficient draws, grouped prediction and 2,000 city resamples. `outputs/` is excluded from Git.

The public input comprises the city panel used in held-out prediction and benchmarking, and aggregate eight-profile composition at each cohort support. Profile counts are reconstructed from published participant counts and shares; individual histories are neither needed nor supplied for this run. The fitted parameter files contain only coefficients and covariance matrices. The public command reproduces results conditional on these fitted parameters; refitting their likelihood needs the licensed inputs below.

## Refitting with licensed participant inputs

The originating CHARLS data provider controls access to individual records. After obtaining the authorized source files, provide their paths explicitly:

```bash
python run.py fit --output outputs --intervals INPUT_INTERVALS.csv.gz --windows INPUT_WINDOWS.csv --charls-core INPUT_CORE.dta --city-panel data/city_year_light_services.csv
```

`--intervals` is the previously derived visit-interval table; `--windows` supplies observed interview and death-report windows; `--charls-core` supplies education, birth timing and weights. The preparation step applies the manuscript's age, observed-state and complete-covariate rules, then refits both endpoints and projects the fitted transition laws. To run official IMaCh calculation mode on the resulting laws, also pass `--imach-exe PATH_TO_IMACH_EXECUTABLE`. [IMaCh](https://euroreves.ined.fr/imach/) is third-party software and is not redistributed here. The fit output includes participant-derived intervals and must remain outside any public commit; `outputs/` is ignored.

## Numerical checks

`run.py public` checks the independent-year differences (3.45 years for men and 3.66 for women, rounded), the economic held-out R² (0.705575), the training-fold PCA economic R² (0.715800), 74 need-matched targets among 167, the eligible-target median radiance gap (47.8%, rounded), and native IMaCh agreement within 0.0002 years. The PCA regression is fitted to its training-fold PCA outcome, including at the final outer-fold fit.

The city's service and radiance observations are aligned by stable support, municipality and year identifiers; no one service is traded off against another in observed-reference eligibility. The exact four-channel decomposition uses all 16 channel subsets and 2,000 joint coefficient draws.