# NYC policy and crash analysis

Open `nyc_policy_analysis.ipynb` in Jupyter or VS Code. It contains the executed analysis, explanations before each code block, charts, numerical interpretations, and source links.

The companion `nyc_policy_analysis.py` uses Jupyter `# %%` cells and is the editable source for the notebook. `tools/run_policy_notebook.py` builds and executes the notebook and exports a browser report with collapsible code.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python tools/run_policy_notebook.py
```

The inputs are the collision, street, and combined hourly traffic archives from GitHub commit `f92bea87dff7ee7a57e41b4af85535f571d1f5a1` (fetched October 17, 2026). The notebook verifies archive hashes and safely extracts the CSVs. Cached data permit offline execution. Kernel startup requires permission to open a local Jupyter connection.

Outputs:

- `outputs/policy_analysis/nyc_policy_analysis.html`: browser report with all executed results.
- `outputs/policy_analysis/findings.md`: computed findings and a suggested five-minute presentation outline.
- `outputs/policy_analysis/*.png`: individual figures for slides.
- `outputs/policy_analysis/*contrasts.csv`: estimates and uncertainty intervals.
- `data/processed/*.csv`: combined analysis panels, street-context lookup, and policy calendar.
- `outputs/policy_analysis/source_manifest.json` and `validation_checks.json`: source versions and integrity checks.

The study separates reported crashes from injuries and deaths, accounts for incomplete periods and seasonality, and checks model sensitivity. The 2020 reporting change, changing travel exposure, and lack of a randomized comparison limit causal claims. The street inventory's SPEED field is an average travel-speed attribute, not a historical legal-speed-limit record.

Both `src/timePattern.py` and the COVID notebook use `data/other/Combined_Traffic_Volume_Counts_hourly.csv` for normalization. Automated hours require four quarter-hour intervals; only complete location-days are retained. Repeated site-hours are averaged before averaging across locations. Outcomes and denominators use the same observed dates. Annual indices use ratios of sums divided by their pooled baseline (1), with matched-date coverage shown explicitly.

These are sample-based traffic-adjusted indices, not citywide per-mile or per-vehicle risks: sampled roads and dates change. The COVID comparison exports `annual_crashes_normalized_by_traffic.csv` and `covid_raw_vs_traffic_adjusted_changes.csv`. Previous VMT outputs are retained as older artifacts and are not inputs to the current analysis. `nyc_policy_analysis_normalized.ipynb` is an executed copy of the same notebook.

## Separate concert and crash pilot

Open `nyc_event_analysis.ipynb` or `outputs/event_analysis/nyc_event_analysis.html`. The separate study compares 19 sourced concert nights with same-weekday reference nights near MSG and Citi Field, including distance and time sensitivity checks. It is descriptive and uses a curated, incomplete event calendar. Rerun with `.venv/bin/python notebooks/nyc_event_analysis.py`. Source links and limitations are included in the report.

## COVID and reporting-change tests

`covid_reporting_analysis.py` compares crash types before and after the March 2020 lockdown and April 6 reporting change, with weekday-matched 2019 comparisons, segmented models, sensitivity checks and a placebo. Run `.venv/bin/python notebooks/covid_reporting_analysis.py`. The report and exported tables are in `outputs/covid_reporting_analysis/`; conclusions are exploratory, not a causal allocation of the decline.

## Animated crash map

Run `.venv/bin/python src/animateCrashHeatmap.py` to create `outputs/crash_timelapse/nyc_crashes_monthly.gif`. The GIF covers complete months from July 2012 to May 2026 using a fixed color scale and crashes per square kilometer per day. The cached official borough map is in `data/external/maps/`. Dependencies include Pillow from `requirements-analysis.txt`.
