# NYC crash heatmap timelapse

Open `nyc_crashes_monthly.gif`. Each frame shows one complete calendar month. The fixed color scale measures spatially smoothed reported crashes per square kilometer per day; month length is accounted for, traffic exposure is not. All vehicle-involved crashes are included. The first and last months pause for readability.

Run `.venv/bin/python src/animateCrashHeatmap.py` to regenerate. Use `--frame-ms 300` for slower playback. Requires numpy, pandas, scipy, matplotlib and Pillow. The cached borough outlines allow offline reruns.

Frames exclude incomplete edge months and omit crashes missing usable coordinates. The mapped share varies over time; reporting rules and travel changed in 2020. Colors indicate burden, not risk per trip or a causal trend. Smoothing uses 250 m cells and a 100 m Gaussian sigma. Nearest-neighbor rendering keeps individual map cells sharp. The color scale uses a square-root transformation to keep lower concentrations visible.

The fixed color scale runs from 0 to 6+ crashes per km² per day. Values above 6 use the hottest color.

Sources: [NYC crashes](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes); [NYC DCP borough boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne). Boundaries retrieved October 17, 2026. See metadata.json for coverage and export checks.
