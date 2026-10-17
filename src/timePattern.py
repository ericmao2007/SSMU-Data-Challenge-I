"""Create draggable time-pattern charts: python3 src/timePattern.py.

Install: python3 -m pip install -r requirements.txt
Saves a standalone webpage that works offline. Add --no-open to save only.
"""

import argparse
from pathlib import Path
import webbrowser

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
INJURIES = "NUMBER OF PERSONS INJURED"
BLUE = "#2563eb"


def load_patterns(csv_path):
    daily = hourly = None
    first = last = None
    missing_dates = missing_injuries = missing_times = 0
    for chunk in pd.read_csv(
        csv_path, usecols=["CRASH DATE", "CRASH TIME", INJURIES], chunksize=200_000,
    ):
        dates = pd.to_datetime(chunk["CRASH DATE"], format="mixed", errors="coerce").dt.normalize()
        injuries = pd.to_numeric(chunk[INJURIES], errors="coerce")
        hours = pd.to_datetime(chunk["CRASH TIME"], format="%H:%M", errors="coerce").dt.hour
        missing_dates += int(dates.isna().sum())
        missing_injuries += int((dates.notna() & ~injuries.ge(0)).sum())
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            first = valid_dates.min() if first is None else min(first, valid_dates.min())
            last = valid_dates.max() if last is None else max(last, valid_dates.max())
        valid = dates.notna() & injuries.ge(0)
        missing_times += int((valid & hours.isna()).sum())
        data = pd.DataFrame({"date": dates[valid], "hour": hours[valid], "injuries": injuries[valid]})
        by_date = data.groupby("date")["injuries"].sum()
        by_hour = data.groupby(["date", "hour"])["injuries"].sum()
        daily = by_date if daily is None else daily.add(by_date, fill_value=0)
        hourly = by_hour if hourly is None else hourly.add(by_hour, fill_value=0)
    if first is None or daily is None or daily.empty:
        raise ValueError("No dated records with valid injury counts found.")
    daily = daily.reindex(pd.date_range(first, last, freq="D"), fill_value=0).sort_index()
    exclusions = {"dates": missing_dates, "injuries": missing_injuries, "times": missing_times}
    return daily, hourly, exclusions


def hourly_chart(daily, hourly):
    """The year slider compares average injuries at each hour of a day."""
    years = sorted(daily.index.year.unique())
    options = [("All years", None)] + [(str(year), year) for year in years]
    steps = []
    averages = []
    for label, year in options:
        selected_days = daily if year is None else daily[daily.index.year == year]
        selected_hours = hourly if year is None else hourly[hourly.index.get_level_values("date").year == year]
        totals = selected_hours.groupby(level="hour").sum().reindex(range(24), fill_value=0)
        values = (totals / len(selected_days)).tolist()
        averages.append(values)
        steps.append(dict(
            label=label, method="update",
            args=[{"y": [values]}, {"title.text": f"When during the day do injuries happen? · {label}"}],
        ))
    fig = go.Figure(go.Bar(
        x=list(range(24)), y=averages[0], marker_color=BLUE,
        hovertemplate="Hour starting %{x:02d}:00<br>%{y:.2f} people injured per day<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", title="When during the day do injuries happen? · All years",
        height=470, margin=dict(t=65, b=115, l=75, r=25),
        xaxis=dict(title="Hour of day (NYC local time)", tickmode="array",
                   tickvals=list(range(0, 24, 2)), ticktext=[f"{hour:02d}:00" for hour in range(0, 24, 2)]),
        yaxis=dict(title="Average people injured per day<br>within each hour", rangemode="tozero"),
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="Year: "), pad=dict(t=45))],
        bargap=0.15,
    )
    return fig


def timeline_chart(daily):
    """Switch aggregation without confusing partial years with full-year totals."""
    fig = go.Figure()
    for index, (label, frequency) in enumerate([("Daily", "D"), ("Monthly", "MS"), ("Yearly", "YS")]):
        groups = daily.resample(frequency)
        average = groups.mean()
        days = groups.size()
        fig.add_trace(go.Scatter(
            x=average.index, y=average.values, customdata=days.values,
            name=label, visible=index == 1, mode="lines+markers" if index == 2 else "lines",
            line=dict(color=BLUE, width=2),
            hovertemplate="Period starting %{x|%b %d, %Y}<br>%{y:.2f} people injured per day"
                          "<br>%{customdata} calendar day(s) included<extra></extra>",
        ))
    buttons = [dict(label=label, method="update", args=[{"visible": [j == i for j in range(3)]}])
               for i, label in enumerate(["Daily", "Monthly", "Yearly"])]
    fig.update_layout(
        template="plotly_white", title="How have traffic injuries changed over time?",
        height=490, margin=dict(t=100, b=65, l=75, r=25), showlegend=False,
        yaxis=dict(title="People injured per day<br>(period average)", rangemode="tozero"),
        xaxis=dict(
            title="Date — drag the handles below to focus on a period", type="date",
            rangeslider=dict(visible=True, thickness=0.17),
            rangeselector=dict(buttons=[
                dict(count=1, label="1 year", step="year", stepmode="backward"),
                dict(count=5, label="5 years", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ]),
        ),
        updatemenus=[dict(type="dropdown", active=1, buttons=buttons, x=1, xanchor="right", y=1.21)],
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/motor_vehicle_collisions.csv")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    daily, hourly, excluded = load_patterns(args.input)
    config = {"responsive": True, "displaylogo": False, "toImageButtonOptions": {"scale": 2}}
    hour_html = hourly_chart(daily, hourly).to_html(full_html=False, include_plotlyjs=True, config=config)
    trend_html = timeline_chart(daily).to_html(full_html=False, include_plotlyjs=False, config=config)
    html = f"""<!doctype html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>NYC Traffic Injury Time Patterns</title>
    <style>
      body {{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.6 Arial,sans-serif}}
      main {{max-width:1150px;margin:auto;padding:24px}}
      section {{background:white;border-radius:10px;padding:12px;margin:20px 0}}
      h1 {{margin:0}} p {{margin:8px 0;color:#475569}} footer {{font-size:13px}}
    </style></head><body><main>
    <h1>NYC Traffic Injury Time Patterns</h1>
    <p>{daily.index.min():%b %d, %Y} – {daily.index.max():%b %d, %Y}
       · {int(daily.sum()):,} recorded people injured</p>
    <section><p>Drag the <b>year slider</b> to compare the 24-hour pattern across years.</p>{hour_html}</section>
    <section><p>Drag the <b>date-range handles</b> below the timeline. Use the dropdown to
      switch between daily, monthly, and yearly views. These controls affect this chart only.</p>{trend_html}</section>
    <footer>Daily averages use calendar days within the dataset date range, including days with
      no recorded injuries as zero. First and last periods may be partial; hover to see days included.
      Monthly and yearly values are averages per day, not totals. These charts describe reported injury
      burden, not risk per trip.<br>
      Excluded: {excluded['dates']:,} records with invalid dates;
      {excluded['injuries']:,} dated records with missing/invalid injury counts.
      The hourly chart additionally excludes {excluded['times']:,} records with invalid times.<br>
      Source: City of New York, Motor Vehicle Collisions – Crashes.</footer>
    </main></body></html>"""
    output = ROOT / "outputs/time_patterns.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved {output}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
