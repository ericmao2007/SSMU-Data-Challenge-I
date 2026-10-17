# Minimal NYC monthly crash GIFs

Each GIF shows only a map, a title with the month, and the monthly count. The crashes GIF counts reported collisions. The people GIF counts persons injured plus persons killed; the source does not count everyone involved. Counts include all reports, while maps require valid coordinates. Only complete edge months are included. Each map uses its own fixed color scale across time, with the same smoothing and daily density normalization as the original animation. Colors represent burden, not risk per trip. Reporting and travel changed in 2020.

Run `.venv/bin/python src/animateCrashMinimal.py` to regenerate. See monthly_counts.csv and metadata.json for counts and export checks.
