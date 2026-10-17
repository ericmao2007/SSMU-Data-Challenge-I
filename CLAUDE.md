# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Entry for Duke SSMU Data Challenge I: one-day analysis of NYC Motor Vehicle Collisions (data.gov), delivered as a code notebook plus a 5-minute presentation. Judging weights: Creativity & Insight 40%, Visualization 25%, Communication 20%, Technical Soundness 10%, Reproducibility 5%.

Competition rules that affect how you work here:
- **"You may not put this data into an LLM."** Do not read, print, or paste raw rows of the collision/traffic CSVs into the conversation. Work through code, and inspect schemas, aggregates, and test fixtures rather than data contents.
- The APA citation for the dataset (see README.md) must appear at the end of the notebook. All outside resources, including AI chat links, must be cited in the presentation.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-analysis.txt   # includes requirements.txt (pandas/folium/plotly)

python -m unittest discover -s tests                 # all tests, run from repo root
python -m unittest tests.test_covid_exposure.ExposureTests.test_no_overlap_rejected   # single test

python tools/run_policy_notebook.py                  # rebuild .ipynb from .py, execute, export HTML
python tools/run_policy_notebook.py --no-execute     # rebuild .ipynb only
python tools/run_policy_notebook.py --export-only    # re-export HTML from the executed .ipynb

python src/timePattern.py --no-open                  # -> outputs/time_patterns.html
python src/locationHeatmap.py --no-open              # -> outputs/nyc_injury_heatmap.html
```

Tests import `src.*` as a package, so run them from the repo root.

## Architecture

**Notebook source of truth.** `notebooks/nyc_policy_analysis.py` (Jupyter `# %%` / `# %% [markdown]` cells) is the editable source. `tools/run_policy_notebook.py` splits it into cells, writes `notebooks/nyc_policy_analysis.ipynb`, executes it with a temporary kernelspec pointing at the current interpreter, fails if any cell errored, and exports `outputs/policy_analysis/nyc_policy_analysis.html` with code folded into `<details>`. Edit the `.py` file and rerun the tool; do not hand-edit the `.ipynb` file.

**Data acquisition.** CSVs are gitignored (`*.csv`). The notebook's setup cell downloads the three archives (collisions, street geospatial, combined hourly traffic) from a pinned upstream commit if missing, verifies their git-blob SHA-1, safely extracts CSVs next to the zip, and writes `source_manifest.json`. `src/` scripts expect the CSVs to already be extracted (`data/motor_vehicle_collisions.csv`, `data/other/Combined_Traffic_Volume_Counts_hourly.csv`), so run the notebook or unzip first.

**Shared traffic normalization (`src/timePattern.py`).** The standalone time-pattern chart and the notebook's COVID section both import `load_traffic` / `normalize_annual_by_traffic` from here, so changes affect both. Rules that the tests enforce:
- `load_traffic` reads the combined long format (`source, segment_id, direction, roadway_name, from_street, to_street, date, hour, volume, intervals`); a location is segment + direction + street endpoints.
- Automated counts need all four quarter-hour intervals per hour, and only complete 24-hour location-days are kept. Repeated site-hours are averaged before averaging across sites.
- `normalize_annual_by_traffic` uses only dates present in both outcomes and exposure. It computes ratio-of-sums indices relative to a pooled baseline (= 1) and rejects duplicate dates or no overlap.
- Results are sample-based traffic-adjusted indices, not citywide per-vehicle or per-mile risk. Keep that framing in prose and chart labels.
- Years have different numbers of matched dates, so compare raw counts per matched date, never matched-date sums. 2020 has no traffic-observed dates in April–July, and the sampled road segments barely overlap between 2019 and 2020; the notebook reports a same-calendar-month sensitivity alongside the main index.

**Outputs.** `outputs/policy_analysis/` holds the figures for slides (numbered by notebook section), `findings.md` (computed findings and a suggested presentation outline), JSON audit/validation files, and contrast CSVs.

**Analytical caveats already encoded in the notebook.** These are worth preserving when editing: the 2020 crash-reporting change, changing travel exposure, and the lack of a randomized comparison limit causal claims. The street inventory's `SPEED` field is average travel speed, not the legal speed limit.
