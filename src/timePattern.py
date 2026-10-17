"""Create traffic-adjusted time-pattern charts: python3 src/timePattern.py.

Install: python3 -m pip install -r requirements.txt
Saves a standalone webpage that works offline. Add --no-open to save only.
"""

import argparse
from pathlib import Path
import webbrowser

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
INJURIES = "NUMBER OF PERSONS INJURED"
BLUE = "#2563eb"
TRAFFIC_FILE = ROOT / "data/other/Combined_Traffic_Volume_Counts_hourly.csv"
TRAFFIC_COMMIT = "f92bea87dff7ee7a57e41b4af85535f571d1f5a1"
TRAFFIC_SOURCE = ("https://github.com/ericmao2007/SSMU-Data-Challenge-I/blob/"
                  f"{TRAFFIC_COMMIT}/data/other/Combined_Traffic_Volume_Counts_hourly.zip")
SITE_COLUMNS = ["SegmentID", "Direction", "Roadway Name", "From", "To"]


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


def load_traffic(csv_path):
    """Convert long hourly counts to complete site-days, then average sites per date.

    A location is a segment/direction/street-endpoints combination, so the
    result is per sampled location, not total sensors. Automated hours require
    all four quarter-hour intervals; historical rows already represent hours.
    Repeated observations of a location/hour are averaged, and only
    location-days with all 24 hours are kept so the sampled sites are
    consistent within each day. This does not correct changing site
    composition between days. Never add hist_volume to volume: it is a
    reference field for overlapping sources, not additional traffic.
    """
    raw = pd.read_csv(csv_path, dtype={"segment_id": "string"}, usecols=[
        "source", "segment_id", "direction", "roadway_name", "from_street",
        "to_street", "date", "hour", "volume", "intervals",
    ])
    raw = raw.rename(columns=dict(zip(
        ["segment_id", "direction", "roadway_name", "from_street", "to_street"], SITE_COLUMNS)))
    sites = raw[SITE_COLUMNS].astype("string").apply(
        lambda col: col.str.strip().str.upper().str.replace(r"\s+", " ", regex=True))
    dates = pd.to_datetime(raw["date"], format="mixed", errors="coerce").dt.normalize()
    hours = pd.to_numeric(raw["hour"], errors="coerce")
    volumes = pd.to_numeric(raw["volume"], errors="coerce")
    full_hour = (raw["source"].eq("historical") |
                 (raw["source"].eq("automated") & pd.to_numeric(raw["intervals"], errors="coerce").eq(4)))
    valid = (dates.notna() & sites.notna().all(axis=1) & sites.ne("").all(axis=1)
             & hours.between(0, 23) & hours.mod(1).eq(0) & volumes.ge(0) & full_hour)
    records = pd.concat([dates.rename("date"), sites, hours.rename("hour"),
                         volumes.rename("volume")], axis=1).loc[valid]
    site_hours = records.groupby(["date", *SITE_COLUMNS, "hour"])["volume"].mean()
    site_days = site_hours.unstack("hour").reindex(columns=range(24)).dropna()
    if site_days.empty:
        raise ValueError("No traffic observations with a valid date and 24 complete hourly counts.")
    exposure = site_days.groupby(level="date").mean()
    exposure["sites"] = site_days.groupby(level="date").size()
    site_years = site_days.reset_index()[["date", *SITE_COLUMNS]]
    site_years["year"] = site_years["date"].dt.year
    common_sites = site_years.groupby(SITE_COLUMNS)["year"].nunique().eq(site_years.year.nunique()).sum()
    report = {"input_rows": len(raw), "excluded_rows": int((~valid).sum()),
              "repeated_site_hours": len(records) - len(site_hours),
              "incomplete_site_days": len(site_hours.index.droplevel("hour").unique()) - len(site_days),
              "common_sites": int(common_sites)}
    return exposure, report


def match_traffic(daily, hourly, exposure):
    """Match calendar dates before calculating exploratory injury/traffic indices."""
    frame = pd.DataFrame({"injuries": daily})
    # Complete site-days guarantee all 24 hours, with no implicit missing-as-zero.
    frame["traffic"] = exposure[list(range(24))].sum(axis=1, min_count=24)
    frame["sites"] = exposure["sites"]
    supported = frame["traffic"].gt(0)
    frame["sites"] = frame["sites"].where(supported)
    frame["matched_injuries"] = frame["injuries"].where(supported)
    frame["traffic"] = frame["traffic"].where(supported)
    if not supported.any():
        raise ValueError("No overlapping crash dates with positive, complete traffic counts.")
    daily_baseline = frame["matched_injuries"].sum() / frame["traffic"].sum()

    sampled = exposure.loc[exposure.index.isin(daily.index), list(range(24))].copy()
    sampled.index.name = "date"
    sampled.columns.name = "hour"
    denominator = sampled.stack().rename("traffic").astype(float)
    matched = denominator.to_frame()
    matched["injuries"] = hourly.reindex(matched.index, fill_value=0)
    # A zero denominator is unavailable, never a zero injury rate.
    matched = matched.loc[matched["traffic"].gt(0)].copy()
    if matched.empty:
        raise ValueError("No overlapping date/hour observations with positive traffic counts.")
    hourly_baseline = matched["injuries"].sum() / matched["traffic"].sum()
    if daily_baseline <= 0 or hourly_baseline <= 0:
        raise ValueError("No positive injury baseline on traffic-observed dates.")
    frame["index"] = frame["matched_injuries"] / frame["traffic"] / daily_baseline
    return frame, matched, daily_baseline, hourly_baseline


def normalize_annual_by_traffic(daily_outcomes, exposure):
    """Annual outcome/traffic indices using the same observed dates on both sides."""
    if not daily_outcomes.index.is_unique or not exposure.index.is_unique:
        raise ValueError("Daily outcomes and traffic must have one row per date.")
    metrics = ["crashes", "harm_crashes", "injured", "killed"]
    traffic = exposure[list(range(24))].sum(axis=1, min_count=24)
    matched = daily_outcomes[metrics].join(traffic.rename("traffic"), how="inner")
    matched = matched.loc[matched.traffic.gt(0) & matched[metrics].notna().all(axis=1)].copy()
    if matched.empty:
        raise ValueError("No overlapping crash dates with positive, complete traffic counts.")
    matched["sites"] = exposure["sites"].reindex(matched.index)
    annual = matched[metrics + ["traffic"]].resample("YS").sum(min_count=1)
    annual["matched_days"] = matched.resample("YS").size()
    annual["mean_sites"] = matched.sites.resample("YS").mean()
    annual = annual.loc[annual.matched_days.gt(0)].copy()
    annual["mean_daily_traffic"] = annual.traffic / annual.matched_days
    for metric in metrics:
        baseline = matched[metric].sum() / matched.traffic.sum()
        annual[f"{metric}_index"] = annual[metric] / annual.traffic / baseline if baseline > 0 else float("nan")
    annual.index = annual.index.year
    annual.index.name = "year"
    return annual


def hourly_chart(matched, baseline):
    """Year slider; each hour uses injuries and exposure from identical dates."""
    years = sorted(matched.index.get_level_values("date").year.unique())
    options = [("All sampled years", None)] + [(str(year), year) for year in years]
    steps, values_by_year, details_by_year = [], [], []
    for label, year in options:
        selected = matched if year is None else matched[matched.index.get_level_values("date").year == year]
        grouped = selected.groupby(level="hour").sum().reindex(range(24))
        days = selected.groupby(level="hour").size().reindex(range(24), fill_value=0)
        values = (grouped["injuries"] / grouped["traffic"] / baseline).tolist()
        details = pd.concat([grouped, days.rename("days")], axis=1).values.tolist()
        values_by_year.append(values)
        details_by_year.append(details)
        steps.append(dict(
            label=label, method="update",
            args=[{"y": [values], "customdata": [details]},
                  {"title.text": f"Within a day · {label}"}],
        ))
    fig = go.Figure(go.Scatter(
        x=list(range(24)), y=values_by_year[0], customdata=details_by_year[0],
        mode="lines+markers", line=dict(color=BLUE, width=3), marker=dict(size=6),
        hovertemplate="Hour starting %{x:02d}:00<br>Adjusted index: %{y:.2f}"
                      "<br>Citywide injuries: %{customdata[1]:,.0f}"
                      "<br>Summed mean-site vehicle counts: %{customdata[0]:,.0f}"
                      "<br>Matched dates: %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", title="Within a day · All sampled years",
        height=470, margin=dict(t=65, b=115, l=75, r=25),
        xaxis=dict(title="Hour of day (NYC local time)", tickmode="array",
                   tickvals=list(range(0, 24, 2)), ticktext=[f"{hour:02d}:00" for hour in range(0, 24, 2)]),
        yaxis=dict(title="Traffic-adjusted index", range=[0, max(
            value for values in values_by_year for value in values if pd.notna(value)
        ) * 1.12], fixedrange=True),
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="Year: "), pad=dict(t=45))],
    )
    fig.add_hline(y=1, line_dash="dash", line_color="#64748b",
                  annotation_text="Overall baseline", annotation_position="top right")
    return fig


def timeline_chart(frame, baseline):
    """Annual summaries plus observed-day coverage; no daily interpolation."""
    groups = frame.resample("YS")
    annual = pd.DataFrame({
        "injuries": groups["matched_injuries"].sum(min_count=1),
        "traffic": groups["traffic"].sum(min_count=1),
        "days": groups["matched_injuries"].count(),
        "sites": groups["sites"].mean(),
    })
    # Keep gaps for wholly unsupported years inside the observed traffic period,
    # but do not stretch the chart into later years with no traffic information.
    observed = annual.loc[annual["days"].gt(0)].index
    annual = annual.loc[observed.min():observed.max()]
    values = annual["injuries"] / annual["traffic"] / baseline
    years = annual.index.year.tolist()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.76, 0.24], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(
        x=years, y=values, customdata=annual.values.tolist(),
        mode="lines+markers", connectgaps=False,
        line=dict(color=BLUE, width=3), marker=dict(size=9),
        hovertemplate="Year %{x:.0f}<br>Adjusted index: %{y:.2f}"
                      "<br>Injuries on sampled dates: %{customdata[0]:,.0f}"
                      "<br>Summed mean-site traffic: %{customdata[1]:,.0f}"
                      "<br>Sampled dates: %{customdata[2]:.0f}"
                      "<br>Mean locations/day: %{customdata[3]:.1f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=years, y=annual["days"], marker_color="#94a3b8",
        text=annual["days"], textposition="outside", cliponaxis=False,
        hovertemplate="%{x:.0f}: %{y} sampled dates<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="#64748b", row=1, col=1,
                  annotation_text="Overall baseline", annotation_position="top right")
    fig.update_layout(
        template="plotly_white", title="Across years · One summary per year",
        height=570, margin=dict(t=65, b=55, l=75, r=25), showlegend=False,
        hovermode="x unified", bargap=0.55,
    )
    fig.update_yaxes(title_text="Traffic-adjusted index", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Sampled<br>dates", rangemode="tozero", showgrid=False,
                     range=[0, annual["days"].max() * 1.4], fixedrange=True, row=2, col=1)
    fig.update_xaxes(type="linear", dtick=1, tickformat="d",
                     range=[min(years) - 0.4, max(years) + 0.4])
    fig.update_xaxes(title_text="Year — drag the range slider to zoom",
                     rangeslider=dict(visible=True, thickness=0.10), row=2, col=1)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/motor_vehicle_collisions.csv")
    parser.add_argument("--traffic", type=Path, default=TRAFFIC_FILE)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    daily, hourly, excluded = load_patterns(args.input)
    exposure, traffic_report = load_traffic(args.traffic)
    frame, matched, daily_baseline, hourly_baseline = match_traffic(daily, hourly, exposure)
    supported = frame["traffic"].dropna().index
    config = {"responsive": True, "displaylogo": False, "toImageButtonOptions": {"scale": 2}}
    hour_html = hourly_chart(matched, hourly_baseline).to_html(
        full_html=False, include_plotlyjs=True, config=config)
    trend_html = timeline_chart(frame, daily_baseline).to_html(
        full_html=False, include_plotlyjs=False, config=config)
    html = f"""<!doctype html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Traffic-Adjusted NYC Injury Patterns</title>
    <style>
      body {{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.6 Arial,sans-serif}}
      main {{max-width:1150px;margin:auto;padding:24px}}
      section {{background:white;border-radius:10px;padding:12px;margin:20px 0}}
      h1 {{margin:0}} p {{margin:8px 0;color:#475569}} footer {{font-size:13px}}
      .method {{background:#fff7ed;padding:14px 18px;border-left:4px solid #ea580c}}
    </style></head><body><main>
    <h1>NYC Injury Patterns, Adjusted for Traffic</h1>
    <p>Two views: the pattern within a day, and the comparison across years.</p>
    <p class="method"><b>How to read the index:</b> 1 = overall baseline;
      2 = twice the injury-to-traffic ratio. This is a sample-based comparison, not a per-vehicle injury rate.</p>
    <section><p>Each point summarizes one hour. Drag the <b>year slider</b> to compare years.
      The vertical scale stays fixed so the comparison is consistent.</p>{hour_html}</section>
    <section><p>Each point summarizes a year using its available traffic-observed dates.
      The bars underneath show how many dates support each point.
      Drag the <b>range slider</b> to focus on particular years.</p>{trend_html}</section>
    <p>{len(supported):,} matched dates from {supported.min():%b %d, %Y} to {supported.max():%b %d, %Y}.
      Traffic-adjusted charts end there; later injury data have no matching traffic counts.
      Lines join hourly or yearly summaries, not estimates for unsampled days.</p>
    <section><details><summary><b>Method and data limitations</b></summary>
      <ol><li>Keep traffic observations with a valid location, date, and all 24 hourly counts.
        Repeated observations for the same location/hour are averaged.</li>
      <li>For each date/hour, average vehicle counts across sampled locations.
        This avoids treating more counters as more traffic.</li>
      <li>Use citywide injuries on the same observed dates/hours. Divide summed injuries by
        summed mean-location vehicle counts, then divide by the pooled ratio to create an index.</li>
      <li>Yearly values are ratios of matched sums, not averages of daily ratios.
        Unsampled days are omitted from both sums; no traffic counts are interpolated.</li></ol>
      <p>The hourly and timeline views each use their pooled matched-observation baseline.
        The hourly view excludes invalid crash times; the timeline includes all dated injury counts.
        Zero traffic denominators are unavailable. Raw injury counts and sampled-location coverage
        appear on hover.</p>
      <p><b>Interpretation limit:</b> averaging locations adjusts for the number of counters, but
        does not make the changing road sample representative of NYC.
        {traffic_report['common_sites']:,} locations are observed in every traffic year in this file.
        A defensible citywide injury rate needs
        citywide exposure estimates or injuries matched to the actual counted road segments.
        Comparisons can also reflect changing sampled roads and seasons.</p>
    </details></section>
    <footer>Traffic cleaning: {traffic_report['excluded_rows']:,} of {traffic_report['input_rows']:,}
      rows excluded.<br>
      Crash exclusions: {excluded['dates']:,} invalid dates;
      {excluded['injuries']:,} dated records with missing/invalid injury counts.
      The hourly chart additionally excludes {excluded['times']:,} invalid crash times.<br>
      Partial site-days excluded: {traffic_report['incomplete_site_days']:,};
      repeated site-hour observations averaged: {traffic_report['repeated_site_hours']:,}.<br>
      Traffic input: {args.traffic.name}. Automated hours require four quarter-hour intervals.<br>
      Sources: City of New York, Motor Vehicle Collisions – Crashes;
      <a href="{TRAFFIC_SOURCE}">Combined Traffic Volume Counts (hourly)</a>.
    </footer></main></body></html>"""
    output = ROOT / "outputs/time_patterns.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(html, encoding="utf-8")
    frame.index.name = "date"
    frame.to_csv(output.with_name("time_normalization.csv"))
    print(f"Saved {output}")
    print(f"Normalized {len(supported):,} matched dates; unsupported dates remain missing.")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
