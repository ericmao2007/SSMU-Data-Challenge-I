# NYC crash policy analysis: findings from the executed notebook

**Data actually used:** 2,269,187 crashes, from July 01, 2012 to June 11, 2026. The street-context join matches 52.1% of all crashes; 30.5% have an unknown borough. Unmatched records remain in the citywide analysis. Partial endpoint months are excluded from monthly models.

**Missing-data finding:** months with more than 0.1% missing injury/fatality fields are excluded from the main monthly figures and models: Apr 2026, May 2026, Jun 2026. Their raw records remain in the exported panels with completeness flags. Missing deaths are never filled with zero.

**Traffic-adjusted COVID comparison (2020 vs 2019):** mean sampled-location traffic per observed day changed -20.2%. There were 316 matched dates in 2019 and 131 in 2020. The harm-crash index changed +9.0% and the fatality index changed +55.5%. These are descriptive sample-based indices. Changing roads, dates, travel mix, and reporting can affect the comparison; they do not isolate the effect of lockdown policy.

**2021 comparison:** the sampled-traffic-adjusted fatality index changed +49.5% relative to 2019. Check matched-date coverage and road composition before interpreting this difference.

**Vision Zero model:** the fatality contrast at 12 months is -33.9% (95% model interval -41.4% to -25.6%). Adding the recording-change indicator gives -34.0%. This is an association under a chosen pre-trend and seasonality model, not proof of causation. Intervals crossing zero are compatible with either direction under this model.

**Road-user differences, 2019 versus 2013:** pedestrian deaths -25.6%; cyclist deaths +181.8%; motorist deaths -25.5%. These changes concern counts, not exposure-adjusted safety; cycling and other travel volumes may have changed.

**24/7 camera time-group comparison:** the harm-crash ratio of rate ratios is 0.946 (95% approximate clustered interval 0.872–1.027). That is a -5.4% relative change in newly covered hours versus weekday daytime. The pre-period differential-trend check has p=0.8; it does not detect a differential trend, but cannot establish parallel trends. The comparison does not identify effects specifically at camera sites, and the pandemic recovery can affect time groups differently.

**Traffic coverage:** complete location-days overlap 3,672 of 5,094 crash dates. Section 7 normalizes only these dates using the combined file. The changing road sample does not provide a citywide per-vehicle or per-mile risk denominator.

## What to say in a five-minute presentation

1. Start with the distinction between crash reports and serious harm.
2. Show the COVID raw-versus-traffic-adjusted comparison and explain the April 2020 reporting change.
3. Show the Vision Zero contrast with its interval and specification sensitivity.
4. Use the 24/7 camera time-group comparison as a focused policy question; show pre-trend and placebo diagnostics.
5. Conclude with what the data support, what remains uncertain, and why a citywide causal risk estimate needs better exposure/control data.

## Claims to avoid

- Do not call every drop in police reports a fall in actual crashes.
- Do not label a conditional injury share or clock-hour rate as injury risk per vehicle.
- Do not use snapshot inventory SPEED as the historical legal speed limit.
- Do not label the all-Manhattan comparison as a toll-zone treatment effect.
- Do not interpret a policy-timed association as a proven causal effect.

The 2021 placebo estimate is visible in the sensitivity figure; inspect it before making a camera-effect claim. A low pre-trend p-value or unstable window estimates weakens the comparison.

## Outputs

- 01_long_run_frequency_and_harm.png
- 02_covid_traffic_adjusted_changes.png
- 03_covid_traffic_and_indices.png
- 04_vision_zero_model_sensitivity.png
- 05_road_user_fatalities.png
- 06_camera_time_groups.png
- 07_camera_window_sensitivity.png
- 08_street_context_descriptive.png
- 09_traffic_sample_coverage.png