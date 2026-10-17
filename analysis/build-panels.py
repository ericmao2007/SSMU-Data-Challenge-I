"""Download the raw inputs and build the daily and monthly crash panels the rest of the analysis reads.

Run this first, from the repository root: python analysis/build-panels.py

1. Downloads the collision and combined hourly traffic archives from a pinned
   upstream commit if missing, checks their git-blob SHA-1, and safely extracts
   the CSVs next to each zip.
2. Aggregates the collision file in chunks to one row per day (crashes, harm
   crashes, people injured/killed by road user, and how many records have a known
   injury/fatality status), then to months with completeness flags.
3. Joins sample-based traffic coverage (complete location-days from
   src/timePattern.load_traffic) to the daily panel.

Writes data/processed/daily_policy_panel.csv, monthly_policy_panel.csv and
daily_crashes_with_traffic_coverage.csv, read by newest/nyc_crash_graphs.ipynb and
analysis/covid-statistics.py and vision-zero-statistics.py.

Caveats: missing injury/fatality values stay missing (never filled with zero), and
months with more than 0.1% missing outcome fields are flagged complete_outcomes =
False so the models can drop them. The traffic columns describe a changing sample of
counted roads, not citywide vehicle travel.
"""
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.timePattern import TRAFFIC_FILE, load_traffic  # noqa: E402

DATA = ROOT / 'data'
PROCESSED = DATA / 'processed'
COMMIT = 'f92bea87dff7ee7a57e41b4af85535f571d1f5a1'
ARCHIVES = [
    ('other/Combined_Traffic_Volume_Counts_hourly.zip', 'b4d8087675f8c0a2816a5e82a2d24b7e0f10fb4f'),
    ('motor_vehicle_collisions.zip', 'f2dccb6531a04f13251f024a11d99298e3346139'),
]
OUTCOME_COLUMNS = {
    'NUMBER OF PERSONS INJURED': 'injured', 'NUMBER OF PERSONS KILLED': 'killed',
    'NUMBER OF PEDESTRIANS INJURED': 'ped_injured', 'NUMBER OF PEDESTRIANS KILLED': 'ped_killed',
    'NUMBER OF CYCLIST INJURED': 'cyclist_injured', 'NUMBER OF CYCLIST KILLED': 'cyclist_killed',
    'NUMBER OF MOTORIST INJURED': 'motorist_injured', 'NUMBER OF MOTORIST KILLED': 'motorist_killed',
}
MEASURES = ['crashes', 'harm_crashes', 'non_harm_crashes', 'known_harm',
            'known_injured', 'known_killed', *OUTCOME_COLUMNS.values()]


def fetch_archives():
    """Download each pinned archive if missing, verify its git blob hash, and extract its CSVs."""
    for relative, expected_blob in ARCHIVES:
        archive = DATA / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            urlretrieve(f'https://raw.githubusercontent.com/ericmao2007/SSMU-Data-Challenge-I/'
                        f'{COMMIT}/data/{relative}', archive)
        blob = hashlib.sha1(f'blob {archive.stat().st_size}\0'.encode())
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                blob.update(block)
        if blob.hexdigest() != expected_blob:
            raise ValueError(f'archive version differs: {relative}')
        with ZipFile(archive) as zipped:
            for item in zipped.infolist():
                if not item.filename.endswith('.csv') or item.filename.startswith('__MACOSX/'):
                    continue
                target = (archive.parent / item.filename).resolve()
                if not target.is_relative_to(archive.parent.resolve()):
                    raise ValueError(f'unsafe archive member: {item.filename}')
                if not target.exists() or target.stat().st_size != item.file_size:
                    zipped.extract(item, archive.parent)


def daily_panel():
    """Sum crash outcomes per day; harm status counts only records with known injury and fatality fields."""
    usecols = ['COLLISION_ID', 'CRASH DATE', *OUTCOME_COLUMNS]
    ids = pd.read_csv(DATA / 'motor_vehicle_collisions.csv', usecols=['COLLISION_ID'])
    if ids['COLLISION_ID'].duplicated().any():
        raise ValueError('duplicate collision IDs require an explicit deduplication decision')
    parts, raw, invalid = [], len(ids), 0
    for chunk in pd.read_csv(DATA / 'motor_vehicle_collisions.csv', usecols=usecols, chunksize=200_000):
        chunk['date'] = pd.to_datetime(chunk['CRASH DATE'], format='mixed', errors='coerce').dt.normalize()
        invalid += int(chunk.date.isna().sum())
        chunk = chunk.loc[chunk.date.notna()].copy()
        for original, short in OUTCOME_COLUMNS.items():
            values = pd.to_numeric(chunk[original], errors='coerce')
            if values.lt(0).any():
                raise ValueError(f'negative count in {original}')
            chunk[short] = values
        known = chunk['injured'].notna() & chunk['killed'].notna()
        chunk['harm_crashes'] = (chunk['injured'].gt(0) | chunk['killed'].gt(0)).astype(float).where(known)
        chunk['non_harm_crashes'] = (chunk['injured'].eq(0) & chunk['killed'].eq(0)).astype(float).where(known)
        chunk['crashes'] = 1
        chunk['known_harm'] = known.astype(int)
        chunk['known_injured'] = chunk.injured.notna().astype(int)
        chunk['known_killed'] = chunk.killed.notna().astype(int)
        parts.append(chunk.groupby('date')[MEASURES].sum(min_count=1))
    daily = pd.concat(parts).groupby(level=0).sum(min_count=1).sort_index()
    assert daily.crashes.sum() == raw - invalid
    # Empty-status days reconcile to zero *known* records; their outcomes stay NaN.
    assert (daily.harm_crashes.fillna(0) + daily.non_harm_crashes.fillna(0) == daily.known_harm).all()
    assert len(daily) == len(pd.date_range(daily.index.min(), daily.index.max())), 'missing crash dates'
    return daily, raw, invalid


def monthly_panel(daily):
    """Sum days to months and flag partial months and months with >0.1% missing outcome fields."""
    monthly = daily.resample('MS').sum(min_count=1)
    monthly['observed_days'] = daily['crashes'].resample('MS').size()
    monthly['calendar_days'] = monthly.index.days_in_month
    monthly['complete_month'] = monthly.observed_days.eq(monthly.calendar_days)
    monthly['injury_field_coverage'] = monthly.known_injured / monthly.crashes
    monthly['fatality_field_coverage'] = monthly.known_killed / monthly.crashes
    monthly['complete_outcomes'] = monthly.injury_field_coverage.ge(.999) & monthly.fatality_field_coverage.ge(.999)
    monthly['month_of_year'] = monthly.index.month
    return monthly


def traffic_coverage(daily):
    """Attach the number of sampled traffic locations and their mean daily count to each crash date."""
    exposure, _ = load_traffic(TRAFFIC_FILE)
    coverage = daily[['crashes', 'harm_crashes', 'injured', 'killed']].copy()
    coverage['sampled_locations'] = exposure['sites']
    coverage['mean_site_daily_count'] = exposure[list(range(24))].sum(axis=1, min_count=24)
    coverage['traffic_observed'] = coverage.mean_site_daily_count.gt(0).fillna(False).astype(int)
    return coverage


def main():
    fetch_archives()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    daily, raw, invalid = daily_panel()
    monthly = monthly_panel(daily)
    coverage = traffic_coverage(daily)
    daily.to_csv(PROCESSED / 'daily_policy_panel.csv')
    monthly.to_csv(PROCESSED / 'monthly_policy_panel.csv')
    coverage.to_csv(PROCESSED / 'daily_crashes_with_traffic_coverage.csv')

    print(f'{raw:,} crash records, {invalid:,} with an invalid date; '
          f'{daily.index.min():%B %d, %Y} to {daily.index.max():%B %d, %Y}')
    flagged = monthly.index[~monthly.complete_outcomes]
    print(f'Months flagged for >0.1% missing injury/fatality fields: {", ".join(flagged.strftime("%b %Y"))}')
    print(f'Crash dates with traffic observations: {int(coverage.traffic_observed.sum()):,} of {len(coverage):,}')
    print(f'Wrote {PROCESSED.relative_to(ROOT)}/daily_policy_panel.csv, monthly_policy_panel.csv '
          'and daily_crashes_with_traffic_coverage.csv')


if __name__ == '__main__':
    main()
