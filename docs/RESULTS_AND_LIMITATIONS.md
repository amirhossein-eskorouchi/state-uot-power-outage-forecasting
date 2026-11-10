# Results and interpretation boundaries

## Observed labeled-data characteristics

The final presentation reports the following properties of the **labeled training data**:

| Descriptive statistic | Value |
| :--- | ---: |
| County-hour observations with OSI equal to zero | 44.1% |
| County-hour observations with OSI greater than 0.05 | 3.0% |
| Share of summed OSI carried by those high-severity county-hours | 60.5% |
| Approximate state-level summed severity carried by ~10% of counties | 40–59% |

These quantities explain the training objective's attention to geographic heterogeneity and heavy-tailed severity. They are **not** measures of predictive accuracy.

## Validation design

Five-fold evaluation holds out **whole counties** during model development. The evaluation uses horizon-specific RMSE and MAE. The uploaded final slide deck presents the validation structure but does not include a comprehensive numeric RMSE/MAE results table, so no such values are asserted here.

## Final held-out predictions

The final 63-county submission covers 144 forecast origins and four requested horizons. The official template contains 9,072 rows; out-of-window target cells remain missing.

Reported **predicted-output** summaries from the submitted ensemble:

| Prediction diagnostic | Value |
| :--- | ---: |
| Share of summed predicted OSI from highest seven held-out counties | 67.4% |
| Share of summed predicted OSI from first 24 target hours | 59.9% |
| Negative forecasts after validation | 0 |
| Maximum same-physical-target disagreement | 0 |

**Important:** Hidden future outage labels for the 63 held-out test counties were not supplied. The heatmap, forecast trajectories, and summaries above describe **predictions**, not their accuracy against observed outages. The five-model forecast range is not a calibrated confidence interval.

## Limitations and future directions

The results concern the March 2026 event and a fixed-history four-state benchmark. They do not establish generalization to unseen storms, regions outside the benchmark, different data quality regimes, or operational utility. The State-UOT term is **not** a physical power-grid simulator and does not model county adjacency directly. Independent-event validation and linking forecasts to restoration/resource-allocation decisions are appropriate next steps.

See the [full presentation](presentation/0Final%20Presentation%20%28Data%20Chalenge%29.pdf) for supporting figures and the full author/team context.
