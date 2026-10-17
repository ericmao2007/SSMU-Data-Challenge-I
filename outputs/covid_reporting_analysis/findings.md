# COVID versus reporting: what can the data distinguish?

**Conclusion:** There was a substantial decline before April 6. This rules out the April 6 citywide policy as the sole explanation for the entire decline. It does not isolate COVID as the cause, or allocate percentages of the decline to COVID and reporting.

## 1. Did crashes fall before the reporting policy?

Compare March 23–April 5 with March 9–22, then adjust that comparison for the corresponding same-weekday change in 2019 (364 days earlier). Each period has 14 days. PAUSE started at 8pm March 22, so March 23 is the first full day.

| Outcome | Adjusted early change | 95% interval |
|---|---:|---:|
| crashes | -54.3% | -65.6% to -39.2% |
| harm_crashes | -57.1% | -70.6% to -37.5% |
| non_harm_crashes | -53.5% | -64.3% to -39.5% |

Intervals use Poisson regression with weekday adjustment and a 7-lag HAC covariance. These are exploratory, model-conditional intervals. The short periods and single prior year limit inference. March 9–22 was already affected by the pandemic, so this does not estimate its full effect.

## 2. Was there an additional relative shift around April 6?

The main segmented model estimates an immediate **-19.5%** change in the ratio of no-reported-harm to harm crashes at April 6 (95% interval **-32.3% to -4.4%**, p=0.0135), conditional on the modeled trends and March 23 break.

This is a relative ratio change, not the percent of the total decline caused by reporting. The model also permits slopes to change after both dates. A falling ratio can reflect reduced recording, changed trip mix, speed, injury severity, or other pandemic conditions.

| Specification | April reporting step in relative ratio | 95% interval |
|---|---:|---:|
| Main | -19.5% | -32.3% to -4.4% |
| Short window | -22.9% | -35.7% to -7.4% |
| Long window | -30.2% | -43.0% to -14.5% |
| Earlier COVID break | -19.5% | -30.1% to -7.2% |
| 2019 placebo | +6.8% | -3.3% to +18.0% |

The 2019 placebo uses corresponding weekdays, not an actual 2019 citywide intervention. Sensitivity to the window or assumed COVID break signals weak identification. The 2019 Staten Island pilot also complicates the use of 2019 as an unaffected reference.

## Model and limits

Count models use Poisson log means; the comparative model uses OLS on log(non-harm/harm). Predictors are a linear weekly time trend, weekday indicators, an immediate step and slope change at each break. All use 7-lag HAC standard errors to allow heteroskedasticity and short-run serial correlation. These remain conditional on the mean model and bandwidth; no claim of a randomized experiment is made.
The two main breaks are only 14 days apart. COVID behavior began before PAUSE and changed continuously. Injury crashes are an imperfect comparison series because travel and crash severity affect them differently. No-harm crashes have known zero injuries and deaths; missing harm status is excluded from that classification. Fatalities alone are too sparse for this short daily comparison.
**Traffic adjustment is unavailable at the key dates:** the project traffic panel has zero observed traffic days in April, May and June 2020. We do not interpolate a denominator or treat annual traffic indices as a local causal control.

## Reproduce

Run `.venv/bin/python notebooks/covid_reporting_analysis.py`. Input: the existing aggregate `data/processed/daily_policy_panel.csv`, derived by the policy notebook from the local collision CSV. Raw collision records are not displayed. All tested specifications and exact period counts are exported.

## Sources

- [NYPD April 3 announcement: citywide policy effective April 6](https://www.nyc.gov/site/nypd/news/p0403b/nypd-expanding-successful-non-injury-collision-pilot-save-nypd-resources)
- [NYC Council notice: PAUSE effective March 22 at 8pm](https://council.nyc.gov/farah-louis/2020/03/21/coronavirus-covid-19-update-governor-cuomo-announces-new-york-state-on-p-a-u-s-e/)
- [NYC Motor Vehicle Collisions—Crashes](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes)