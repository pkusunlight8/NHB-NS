# Released analysis inputs

`city_year_light_services.csv` contains only the columns read by the prediction and benchmark scripts. Its rows and values are selected from the Article's `city_year_light_services.csv` Source Data file. Site and municipality identifiers are stable pseudonyms across rows. The city panel is aggregate and contains no participant histories.

`cohort_standardised_care_need.csv` contains participant counts and eight profile shares for 84 cohort supports with at least ten participants, selected from the Article's corresponding Source Data file. Profiles are coded `female * 4 + rural * 2 + lower_education`, where each factor is 0 or 1. The analysis reconstructs exact integer profile counts by rounding each published share multiplied by the support's participant count, then divides by the count before multinomial draws. This preserves the original random-number sequence and reproduces the published uncertainty results.

`model_parameters/` contains only the 4 × 10 transition coefficient array and 40 × 40 municipal-sandwich covariance matrix for each endpoint. These are selected from the Article's released fitted parameter files; participant-level design rows are absent. `native_imach/` contains eight state-expectancy text outputs from official IMaCh 0.99s14 calculation mode, copied from the Article's Reproducibility material. They support independent verification of direct monthly integration.

The repository's MIT license covers the custom code. These selected study inputs retain the source and access descriptions in the Article's Data availability statement.
