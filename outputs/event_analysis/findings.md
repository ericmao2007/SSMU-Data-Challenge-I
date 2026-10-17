# Do major concerts coincide with more nearby NYC crashes?

**Main finding:** The two venues show opposite patterns: the 4 Citi Field nights had 13 crashes versus 3.9 expected; the 15 MSG nights had 23 versus 32.6 expected. The combined average hides this difference.

**Combined result:** Across 19 selected concert nights, crashes within 1 km during 16:00–02:00 averaged **1.89**, versus **1.92** on matched reference nights (**-1.3%**). This is an association in a small selected sample, not a causal concert effect.

There were **36 observed crashes**, compared with **36.5 matched expected crashes** (sum of event-specific reference averages). Injury/fatal crashes averaged 0.58 versus 0.67 (-13.5%).

## Results by artist

| Artist | Nights | Concert crashes/night | Reference crashes/night | Difference |
|---|---:|---:|---:|---:|
| Beyonce | 2 | 4.00 | 1.38 | +190.9% |
| BTS | 1 | 3.00 | 0.88 | +242.9% |
| Pink | 1 | 2.00 | 0.25 | +700.0% |
| Harry Styles | 15 | 1.53 | 2.17 | -29.5% |

The pooled result is dominated by the 15 MSG shows. Citi Field contributes only four nights. Large artist percentages arise from very small baseline counts and should not be treated as stable estimates.

## Design

The calendar contains Beyonce (Citi Field, June 7–8, 2016), BTS (Citi Field, October 6, 2018), Pink (Citi Field, August 3, 2023), and Harry Styles (15 MSG dates in August–September 2022). Event dates were transcribed from online sources accessed October 17, 2026; this is a curated pilot, not a complete NYC event database.
Each concert is compared with the same venue, clock window, and weekday at ±7, ±14, ±21 and ±28 days. Other listed concerts at that venue are excluded from reference nights. Each concert receives equal weight, even when reference-night counts differ. Midnight–02:00 crashes belong to the preceding event date. All zero-count venue windows are retained. No raw collision rows are displayed or exported.
Coordinates use approximate venue centers and haversine distance. Local results exclude missing/invalid coordinates; citywide counts retain them. These are all reported vehicle-involved crashes, not passenger-car-only crashes. Injury outcomes count people separately from crash frequency.

## Sensitivity and interpretation

- 0.5 km: +24.9% in crash counts.
- 1 km: -1.3% in crash counts.
- 2 km: +7.6% in crash counts.
- 2–5 km ring: +3.9% in crash counts.
- citywide: -1.1% in crash counts.
- Morning placebo (06:00–12:00, 1 km): -0.7%. A morning difference can flag residual date-level confounding; early arrivals may also affect it.
- Near-versus-ring ratio of count ratios: 0.95. Values above 1 mean the near-zone change exceeds the 2–5 km ring change; spillovers and different road environments limit this comparison.

**What this can support:** selected concert nights had the observed count differences shown above. It cannot establish whether a concert caused a crash, whether attendees were involved, or whether risk per driver increased.
**Limits:** Harry Styles supplies 15/19 nights; the remaining artist samples have one or two nights. References can include unlisted concerts, Mets games, US Open sessions, holidays and other events. Weather, construction, traffic volume, attendance and transit use are not controlled. Reference dates are reused, and consecutive shows are correlated. No significance tests or confidence intervals are claimed. Radius and time sensitivities are exploratory, not independent confirmations. Missing geocodes can bias local comparisons. The 2016–2023 dates span reporting and travel changes, so artist contrasts are especially confounded.
**Data audit:** 2,269,187 input rows; coverage 2012-07-01 to 2026-06-11; valid NYC-area coordinates in 90.3% of selected calendar-day records. All selected event/reference dates and following days lie inside dataset coverage. Duplicate IDs removed from selected records: 0.

## Reproduce

Run the six code cells in this notebook in order, from the repository root or notebooks folder. The concert calendar is embedded. The only data input required is the local `data/motor_vehicle_collisions.csv`; install numpy, pandas and matplotlib in the notebook environment. Cells rebuild comparison tables and PNG graphs without the companion script or precomputed outputs.

## Sources

- [beyonce_mlb](https://www.mlb.com/amp/news/beyonce-to-perform-at-citi-field-june-7-8-c163902796.html): Beyonce: June 7–8, 2016, Citi Field
- [bts_aeg](https://aegworldwide.com/press-center/press-releases/bts-confirms-highly-anticipated-details-their-first-stadium-show-us): BTS: October 6, 2018, Citi Field
- [pink_mlb](https://www.mlb.com/press-release/press-release-pink-summer-carnival-at-citi-field-in-2023): Pink: August 3, 2023, Citi Field
- [harry_tour](https://www.seatstories.com/news/harry-styles-five-dates-added-new-york-los-angeles-harrys-house-tickets): Harry Styles individual dates; syndicated TicketNews tour announcement. September 8 also cross-checked against the retrospective performance record.
- [harry_msg](https://www.msgentertainment.com/msg-is-harrys-house-harry-styles-makes-history-at-madison-square-garden-as-banner-is-raised-to-the-rafters-following-15-consecutive-nights-of-love-on-tour-at-the-garden/): Venue confirmation of 15 shows, August 20–September 21, 2022
- [harry_fader](https://www.thefader.com/2022/08/26/harry-styles-uk-eu-tour-dates-2023): Additional published tour schedule cross-check: August 28 and all nine September MSG dates
- [harry_sep8](https://www.setlist.fm/setlist/harry-styles/2022/madison-square-garden-new-york-ny-5bb0f7fc.html): Supplemental retrospective confirmation of September 8 performance
- [crashes](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes): City of New York police-reported motor vehicle collisions; local competition CSV

City of New York—Motor Vehicle Collisions—Crashes. (n.d.). Data.Gov. Retrieved October 14, 2026 (competition dataset acknowledgment).