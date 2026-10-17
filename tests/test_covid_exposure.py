"""Check matching dates and ratio-of-sums normalization for combined counts."""
import unittest
import pandas as pd
from src.timePattern import normalize_annual_by_traffic


class ExposureTests(unittest.TestCase):
    def sample(self):
        dates = pd.to_datetime(['2019-01-01', '2019-01-02', '2020-01-01', '2020-01-02'])
        daily = pd.DataFrame({metric: [10, 20, 30, 9000]
                              for metric in ['crashes', 'harm_crashes', 'injured', 'killed']}, index=dates)
        exposure = pd.DataFrame(0., index=dates[:3], columns=range(24))
        exposure[0] = [100, 400, 1000]
        exposure['sites'] = [1, 10, 2]
        return daily, exposure

    def test_only_matched_dates_and_ratio_of_sums(self):
        result = normalize_annual_by_traffic(*self.sample())
        self.assertEqual(list(result.matched_days), [2, 1])
        self.assertEqual(result.loc[2020, 'killed'], 30)
        baseline = 60 / 1500
        self.assertAlmostEqual(result.loc[2019, 'killed_index'], (30 / 500) / baseline)
        self.assertAlmostEqual(result.loc[2020, 'killed_index'], (30 / 1000) / baseline)

    def test_zero_and_incomplete_exposure_omitted(self):
        daily, exposure = self.sample()
        exposure.loc[exposure.index[1], 0] = 0
        exposure.loc[exposure.index[2], 23] = float('nan')
        result = normalize_annual_by_traffic(daily, exposure)
        self.assertEqual(list(result.index), [2019])
        self.assertEqual(result.loc[2019, 'matched_days'], 1)

    def test_no_overlap_rejected(self):
        daily, exposure = self.sample()
        exposure.index += pd.DateOffset(years=10)
        with self.assertRaisesRegex(ValueError, 'No overlapping'):
            normalize_annual_by_traffic(daily, exposure)

    def test_duplicate_dates_rejected(self):
        daily, exposure = self.sample()
        with self.assertRaisesRegex(ValueError, 'one row per date'):
            normalize_annual_by_traffic(pd.concat([daily, daily.iloc[:1]]), exposure)


if __name__ == '__main__':
    unittest.main()
