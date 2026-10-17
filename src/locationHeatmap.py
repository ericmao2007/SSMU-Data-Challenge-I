"""Create a smooth, interactive NYC traffic injury heatmap.

Run: python3 src/locationHeatmap.py (or add --no-open to save only).
Install: python3 -m pip install -r requirements.txt
The webpage needs internet for street-map tiles.
"""

import argparse
from html import escape
from math import cos, radians
from pathlib import Path
import webbrowser

import folium
from folium.plugins import HeatMap
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INJURIES = "NUMBER OF PERSONS INJURED"
CELL_METRES = 100
RADIUS = 20  # Hotspot size in screen pixels.
BLUR = 30    # Increase for a smoother heatmap.
WEST, SOUTH, EAST, NORTH = -74.26, 40.49, -73.68, 40.92
# Local distance approximation; cell sizes vary by less than 1% across NYC.
DX = CELL_METRES / (111_320 * cos(radians(40.7)))
DY = CELL_METRES / 111_320


def load_cells(csv_path):
    totals = None
    rows = located = 0
    first_date = last_date = None
    for chunk in pd.read_csv(
        csv_path, chunksize=200_000,
        usecols=["LATITUDE", "LONGITUDE", INJURIES, "CRASH DATE"],
    ):
        rows += len(chunk)
        dates = pd.to_datetime(chunk["CRASH DATE"], format="mixed", errors="coerce").dropna()
        if not dates.empty:
            first_date = dates.min() if first_date is None else min(first_date, dates.min())
            last_date = dates.max() if last_date is None else max(last_date, dates.max())
        numbers = chunk[["LATITUDE", "LONGITUDE", INJURIES]].apply(pd.to_numeric, errors="coerce")
        valid = numbers["LATITUDE"].between(SOUTH, NORTH) & numbers["LONGITUDE"].between(WEST, EAST)
        located += int(valid.sum())
        numbers = numbers.loc[valid & numbers[INJURIES].gt(0)].copy()
        numbers["x"] = ((numbers["LONGITUDE"] - WEST) // DX).astype(int)
        numbers["y"] = ((numbers["LATITUDE"] - SOUTH) // DY).astype(int)
        grouped = numbers.groupby(["x", "y"])[INJURIES].agg(injuries="sum", crashes="size")
        totals = grouped if totals is None else totals.add(grouped, fill_value=0)
    if totals is None or totals.empty:
        raise ValueError("No injury crashes with valid NYC coordinates found.")
    period = (f"{first_date:%b %Y} – {last_date:%b %Y}" if first_date is not None
              else "All dates in the input dataset")
    return totals, rows, located, period


def build_map(totals, rows, located, period):
    # Approximate locations by 100 m cell centres for a responsive webpage.
    # Normalize weights to the 0–1 range expected by the heat renderer.
    maximum = totals["injuries"].max()
    points = [
        [SOUTH + (y + 0.5) * DY, WEST + (x + 0.5) * DX, row["injuries"] / maximum]
        for (x, y), row in totals.iterrows()
    ]
    city_map = folium.Map(location=[40.71, -73.97], tiles="OpenStreetMap",
                          zoom_start=11, control_scale=True)
    city_map.fit_bounds([[SOUTH, WEST], [NORTH, EAST]])
    HeatMap(
        points, name="Injury concentration", radius=RADIUS, blur=BLUR,
        min_opacity=0.25, max_zoom=13,
        gradient={0.2: "#ffffb2", 0.4: "#fecc5c", 0.6: "#fd8d3c",
                  0.8: "#f03b20", 1.0: "#bd0026"},
    ).add_to(city_map)
    folium.LayerControl(collapsed=True).add_to(city_map)
    city_map.get_root().header.add_child(folium.Element("<title>NYC Traffic Injury Hotspots</title>"))
    city_map.get_root().html.add_child(folium.Element(f"""
      <div style="position:fixed;top:12px;left:55px;z-index:1000;background:white;
                  padding:12px 16px;border-radius:6px;font:14px/1.5 sans-serif;
                  box-shadow:0 1px 8px #0003;max-width:calc(100vw - 130px)">
        <b style="font-size:20px">Where do NYC traffic injuries cluster?</b><br>
        {escape(period)} · {int(totals['injuries'].sum()):,} people injured at mapped locations
      </div>
      <div style="position:fixed;bottom:28px;left:15px;z-index:1000;background:white;
                  padding:12px 16px;border-radius:6px;font:14px/1.5 sans-serif;
                  box-shadow:0 1px 8px #0003;width:260px;max-width:calc(100vw - 62px)">
        <b>Traffic injury concentration</b>
        <div style="height:14px;margin-top:8px;background:linear-gradient(
             to right,#ffffb2,#fecc5c,#fd8d3c,#f03b20,#bd0026)"></div>
        <div style="display:flex;justify-content:space-between">
          <span>Lower</span><span>Higher</span>
        </div>
        <div style="font-size:12px;color:#475569;margin-top:8px">
          Weighted by the number of people injured.<br>
          Smoothing highlights clusters; colors represent relative intensity
          at the current zoom, not exact counts.<br>
          {located / rows:.1%} of crash records have coordinates within the NYC extent;
          other records are omitted.<br>
          Shows injury burden, not risk per trip.<br>
          Source: NYC Motor Vehicle Collisions – Crashes.
        </div>
      </div>
    """))
    return city_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="Only save the webpage.")
    args = parser.parse_args()
    city_map = build_map(*load_cells(ROOT / "data/motor_vehicle_collisions.csv"))
    output = ROOT / "outputs/nyc_injury_heatmap.html"
    output.parent.mkdir(exist_ok=True)
    city_map.save(str(output))
    print(f"Saved {output}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
