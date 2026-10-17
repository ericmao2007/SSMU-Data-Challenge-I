"""Checks for denominator alignment and aggregation: python3 -m unittest discover -s tests."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src import timePattern as patterns


class TrafficNormalizationTests(unittest.TestCase):
    def test_combined_format_excludes_partial_hours_and_partial_days(self):
        records = []
        for site, source, volume in [(1, "automated", 100), (2, "historical", 300)]:
            for hour in range(24):
                records.append(dict(source=source, segment_id=site, direction="NB",
                                    roadway_name="MAIN ST", from_street="A", to_street="B",
                                    date="2020-01-01", hour=hour, volume=volume,
                                    intervals=4 if source == "automated" else None,
                                    hist_volume=99999))
        # Average repeated site-hours, without adding the overlap reference.
        records.append({**records[0], "volume": 300})
        for hour in range(24):
            records.append({**records[0], "segment_id": 3, "hour": hour,
                            "intervals": 3 if hour == 23 else 4})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.csv"
            pd.DataFrame(records).to_csv(path, index=False)
            exposure, report = patterns.load_traffic(path)
        self.assertEqual(exposure.iloc[0][0], 250)
        self.assertEqual(exposure.iloc[0][1], 200)
        self.assertEqual(exposure.iloc[0]["sites"], 2)
        self.assertEqual(report["excluded_rows"], 1)
        self.assertEqual(report["incomplete_site_days"], 1)
        self.assertEqual(report["repeated_site_hours"], 1)

    def sample_data(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-01", "2021-01-01"])
        # Large injuries on an unobserved date must not leak into normalization.
        daily = pd.Series([10.0, 20.0, 9000.0, 10.0], index=dates)
        index = pd.MultiIndex.from_product([dates, range(24)], names=["date", "hour"])
        hourly = pd.Series(0.0, index=index)
        for date, count in daily.items():
            hourly.loc[(date, 0)] = count
        exposure = pd.DataFrame(0.0, index=dates[[0, 1, 3]], columns=range(24))
        exposure[0] = [100.0, 400.0, 200.0]
        exposure["sites"] = [2, 5, 1]
        return daily, hourly, exposure

    def test_matching_and_zero_traffic_never_create_zero_rates(self):
        daily, hourly, exposure = self.sample_data()
        frame, matched, daily_baseline, hourly_baseline = patterns.match_traffic(daily, hourly, exposure)
        self.assertTrue(pd.isna(frame.loc["2020-02-01", "index"]))
        self.assertTrue(pd.isna(frame.loc["2020-02-01", "matched_injuries"]))
        self.assertEqual(len(matched), 3)  # Zero-traffic hours excluded.
        self.assertAlmostEqual(daily_baseline, 40 / 700)
        self.assertAlmostEqual(hourly_baseline, 40 / 700)
        self.assertAlmostEqual((frame["index"] * frame["traffic"]).sum() / frame["traffic"].sum(), 1)

    def test_yearly_values_are_ratios_of_sums_with_visible_sample_coverage(self):
        frame, _, baseline, _ = patterns.match_traffic(*self.sample_data())
        chart = patterns.timeline_chart(frame, baseline)
        yearly = chart.data[0]
        self.assertAlmostEqual(yearly.y[0], (30 / 500) / baseline)
        self.assertEqual(list(yearly.x), [2020, 2021])
        self.assertEqual(list(chart.data[1].y), [2, 1])
        self.assertFalse(chart.data[0].connectgaps)
        self.assertTrue(chart.layout.xaxis2.rangeslider.visible)

    def test_annual_chart_retains_internal_missing_years_and_trims_unsupported_tail(self):
        frame, _, baseline, _ = patterns.match_traffic(*self.sample_data())
        frame.index = pd.to_datetime(["2020-01-01", "2020-01-02", "2021-01-01", "2022-01-01"])
        frame.loc[pd.Timestamp("2026-01-01")] = [0, float("nan"), float("nan"), float("nan"), float("nan")]
        chart = patterns.timeline_chart(frame, baseline)
        self.assertEqual(list(chart.data[0].x), [2020, 2021, 2022])
        self.assertTrue(pd.isna(chart.data[0].y[1]))
        self.assertEqual(chart.data[1].y[1], 0)

    def test_year_slider_updates_normalized_values_and_hover_details(self):
        _, matched, _, baseline = patterns.match_traffic(*self.sample_data())
        chart = patterns.hourly_chart(matched, baseline)
        self.assertEqual(chart.data[0].type, "scatter")
        self.assertEqual(chart.data[0].mode, "lines+markers")
        self.assertTrue(chart.layout.yaxis.fixedrange)
        step = chart.layout.sliders[0].steps[-1]
        self.assertEqual(step.label, "2021")
        self.assertAlmostEqual(step.args[0]["y"][0][0], (10 / 200) / baseline)
        self.assertEqual(step.args[0]["customdata"][0][0], [200, 10, 1])

    def test_disjoint_dates_fail_clearly(self):
        daily, hourly, exposure = self.sample_data()
        exposure.index = exposure.index + pd.DateOffset(years=10)
        with self.assertRaisesRegex(ValueError, "No overlapping crash dates"):
            patterns.match_traffic(daily, hourly, exposure)


if __name__ == "__main__":
    unittest.main()
