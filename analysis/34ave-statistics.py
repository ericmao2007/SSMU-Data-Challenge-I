"""Difference-in-differences test of whether crashes on 34th Avenue fell more than on a comparison avenue.

34th Avenue (69th St to Junction Blvd, Jackson Heights) became an Open Street in
May 2020 and was redesigned in 2022-2023. 35th Avenue, the parallel two-way avenue
over the same span, was not, so it serves as the primary comparison corridor.
   H0: the before-to-after change on 34th Ave equals the change on the comparison avenue.
   H1 (one-sided): crashes on 34th Ave fell more than on the comparison avenue.

Two versions of "equal change" are tested on monthly corridor counts:
1. Proportional (primary): Poisson DiD with month fixed effects. Conditioning on each
   month's two-corridor total turns it into a binomial model of 34th Ave's share, so
   citywide shocks that hit both corridors proportionally (seasonality, COVID, the
   April 2020 reporting change) drop out. exp(beta) is the ratio of 34th Ave's
   after/before ratio to the comparison's.
2. Additive: OLS on the monthly 34th-minus-comparison difference, i.e. the change in
   crashes per year on 34th Ave minus the change on the comparison avenue.
Both use HAC (Newey-West, 12 lags) standard errors for overdispersion and
autocorrelation. Before = 2017-2019 and after = 2024-2025 (post launch and redesign);
2020-2021 (launch) and 2022-2023 (redesign) get their own terms so they do not leak
into the main contrast and are reported as secondary contrasts.

Parallel-trends check: a linear trend in 34th Ave's share over 2013-2019 (H0: no
differential pre-trend) and a yearly event study relative to 2019.
Sensitivity: 37th Ave, 32nd Ave, and the three comparison avenues pooled.

Reads data/motor_vehicle_collisions.csv and newest/data/street_centerlines.json and
selects corridor crashes with the same rule as newest/nyc_crash_graphs.ipynb (named
on/cross street, within 35 m of the corridor centerline including intersections).
Writes outputs/34ave_statistics/.

Caveats: with one treated and one comparison corridor, the standard errors reflect
month-to-month noise, not the uncertainty of choosing the comparison avenue (hence
the sensitivity runs). Closing 34th Ave to through traffic can divert cars onto the
parallel avenues; that spillover raises comparison crashes and overstates the
relative drop. The Open Street also changed who uses 34th Ave (fewer cars, more
people walking and cycling), there is no corridor traffic denominator, and the
centerlines are current rather than historical. Results are associations, not
randomized causal effects.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
COLLISIONS = ROOT / 'data/motor_vehicle_collisions.csv'
CENTERLINES = ROOT / 'newest/data/street_centerlines.json'
REFERENCE_SELECTION = ROOT / 'newest/data/34ave_crashes.csv'
OUT = ROOT / 'outputs/34ave_statistics'
TREATED = '34 AVE'
PRIMARY = '35 AVE'
COMPARISONS = {'35 AVE': '35th Ave', '37 AVE': '37th Ave', '32 AVE': '32nd Ave'}
POOLED = 'POOLED'
LABELS = {TREATED: '34th Ave', **COMPARISONS, POOLED: 'Pooled 32nd/35th/37th Ave'}
ENDS = ['69 ST', 'JUNCTION BLVD']
CROSS_NUMBERS = (69, 94)  # numbered cross streets between 69th St and Junction Blvd
DISTANCE_M = 35
OUTCOMES = {'crashes': 'Reported crashes', 'harm_crashes': 'Injury/death crashes',
            'injured': 'People injured', 'ped_injured': 'Pedestrians injured',
            'cyclist_injured': 'Cyclists injured'}
# Calendar-year periods, matching the period comparison chart in newest/nyc_crash_graphs.ipynb.
BEFORE = ('before', 2017, 2019)
CONTRASTS = [('launch', 2020, 2021), ('redesign', 2022, 2023), ('after', 2024, 2025)]
MAIN = 'after'
CONTRAST_LABELS = {'launch': 'Launch 2020-21', 'redesign': 'Redesign 2022-23', 'after': 'After 2024-25'}
FIRST_YEAR, PRE_END, LAST_YEAR = 2013, 2019, 2025  # data begin July 2012, so 2013 is the first full year
REFERENCE_YEAR = 2019
HAC = {'maxlags': 12}
ALPHA = 0.05
SCALE = np.array([111320 * np.cos(np.deg2rad(40.754)), 111320])  # degrees -> metres near Jackson Heights


def road_lines(roads, name):
    return [np.asarray(line, float) for road in roads if road.get('full_street_name') == name
            for line in road['the_geom']['coordinates']]


def corridor_geometry(roads, name):
    """Centerline segments of `name` between its intersections with 69th St and Junction Blvd."""
    nodes = np.concatenate(road_lines(roads, name))
    endpoints = []
    for cross in ENDS:
        cross_nodes = np.concatenate(road_lines(roads, cross))
        distances = np.linalg.norm((nodes[:, None, :] - cross_nodes[None, :, :]) * SCALE, axis=2).min(axis=1)
        shared = np.unique(nodes[distances < 3], axis=0)
        if not len(shared):
            raise ValueError(f'{name} does not meet {cross} in the centerline extract')
        endpoints.append(shared.mean(axis=0))
    west, east = endpoints
    lines = [line for line in road_lines(roads, name) if west[0] <= line[:, 0].mean() <= east[0]]
    segments = np.concatenate([np.stack([line[:-1], line[1:]], axis=1) for line in lines])
    a, b = (segments[:, 0] - west) * SCALE, (segments[:, 1] - west) * SCALE
    return {'west': west, 'east': east, 'a': a, 'ab': b - a, 'segments': len(segments)}


def distance_to(geometry, points):
    a, ab = geometry['a'], geometry['ab']
    p = (points - geometry['west']) * SCALE
    t = np.clip(np.sum((p[:, None, :] - a) * ab, axis=2) / np.sum(ab * ab, axis=1), 0, 1)
    return np.linalg.norm(p[:, None, :] - (a + t[:, :, None] * ab), axis=2).min(axis=1)


def clean_street(series):
    return (series.fillna('').str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)
            .str.replace(r'\b(\d+)(?:ST|ND|RD|TH)\b', r'\1', regex=True)
            .str.replace(r'\bAVENUE\b', 'AVE', regex=True).str.replace(r'\bSTREET\b', 'ST', regex=True)
            .str.replace(r'\bBOULEVARD\b', 'BLVD', regex=True))


def select_corridor_crashes():
    """One pass over the collisions file, keeping crashes on each corridor."""
    roads = json.loads(CENTERLINES.read_text())
    geometries = {name: corridor_geometry(roads, name) for name in [TREATED, *COMPARISONS]}
    audit = {name: {'named_corridor_candidates': 0, 'spatial_outliers': 0, 'missing_coordinates_retained': 0,
                    'centerline_segments': g['segments'], 'west_endpoint': g['west'].tolist(),
                    'east_endpoint': g['east'].tolist()} for name, g in geometries.items()}
    columns = ['COLLISION_ID', 'CRASH DATE', 'BOROUGH', 'LATITUDE', 'LONGITUDE', 'ON STREET NAME',
               'CROSS STREET NAME', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
               'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF CYCLIST INJURED']
    parts = []
    for chunk in pd.read_csv(COLLISIONS, usecols=columns, chunksize=200000):
        on, cross = clean_street(chunk['ON STREET NAME']), clean_street(chunk['CROSS STREET NAME'])
        on_number = on.str.extract(r'^(\d+) ST$')[0].astype(float)
        cross_number = cross.str.extract(r'^(\d+) ST$')[0].astype(float)
        lat = pd.to_numeric(chunk.LATITUDE, errors='coerce')
        lon = pd.to_numeric(chunk.LONGITUDE, errors='coerce')
        usable = lat.between(40.49, 40.92) & lon.between(-74.26, -73.68)
        for name, g in geometries.items():
            named = on.eq(name) | cross.eq(name)
            bounded_name = ((on.eq(name) & (cross_number.between(*CROSS_NUMBERS) | cross.eq('JUNCTION BLVD'))) |
                            (cross.eq(name) & (on_number.between(*CROSS_NUMBERS) | on.eq('JUNCTION BLVD'))))
            low_lat, high_lat = sorted([g['west'][1], g['east'][1]])
            local = (usable & lon.between(g['west'][0] - .00035, g['east'][0] + .00035)
                     & lat.between(low_lat - .0005, high_lat + .0005))
            distance = pd.Series(np.inf, index=chunk.index)
            mask = local & named
            if mask.any():
                distance.loc[mask] = distance_to(g, np.column_stack([lon[mask], lat[mask]]))
            keep = named & ((local & distance.le(DISTANCE_M)) | (~usable & bounded_name & chunk.BOROUGH.eq('QUEENS')))
            audit[name]['named_corridor_candidates'] += int((named & bounded_name).sum())
            audit[name]['spatial_outliers'] += int((named & bounded_name & usable & ~keep).sum())
            audit[name]['missing_coordinates_retained'] += int((keep & ~usable).sum())
            sub = chunk.loc[keep, ['COLLISION_ID', 'CRASH DATE']].copy()
            sub['corridor'] = name
            for original, short in [('NUMBER OF PERSONS INJURED', 'injured'), ('NUMBER OF PERSONS KILLED', 'killed'),
                                    ('NUMBER OF PEDESTRIANS INJURED', 'ped_injured'),
                                    ('NUMBER OF CYCLIST INJURED', 'cyclist_injured')]:
                sub[short] = pd.to_numeric(chunk.loc[keep, original], errors='coerce')
            parts.append(sub)
    crashes = pd.concat(parts, ignore_index=True)
    crashes['date'] = pd.to_datetime(crashes.pop('CRASH DATE'), format='%m/%d/%Y', errors='coerce')
    crashes = crashes.dropna(subset=['date'])
    known = crashes.injured.notna() & crashes.killed.notna()
    crashes['harm_crashes'] = (crashes.injured.gt(0) | crashes.killed.gt(0)).astype(float).where(known)
    crashes['crashes'] = 1
    # A crash named on two corridors (impossible for parallel avenues, but cheap to check) is dropped from both.
    shared = crashes.COLLISION_ID[crashes.COLLISION_ID.duplicated()]
    crashes = crashes.loc[~crashes.COLLISION_ID.isin(shared)]
    for name in geometries:
        audit[name]['selected_crashes'] = int(crashes.corridor.eq(name).sum())
    audit['shared_between_corridors_dropped'] = int(shared.nunique())
    audit['last_crash_date'] = str(crashes.date.max().date())
    if REFERENCE_SELECTION.exists():
        reference = pd.read_csv(REFERENCE_SELECTION, usecols=['COLLISION_ID']).COLLISION_ID
        ours = crashes.COLLISION_ID[crashes.corridor.eq(TREATED)]
        audit[TREATED]['matches_newest_selection'] = bool(set(ours) == set(reference))
    return crashes, audit


def monthly_panel(crashes):
    """Wide month x (outcome, corridor) counts for complete calendar years, plus the pooled comparison."""
    if crashes.date.max() < pd.Timestamp(f'{LAST_YEAR}-12-31'):
        raise ValueError(f'collision data end before {LAST_YEAR} is complete')
    months = pd.date_range(f'{FIRST_YEAR}-01-01', f'{LAST_YEAR}-12-01', freq='MS')
    wide = (crashes.groupby(['corridor', pd.Grouper(key='date', freq='MS')])[list(OUTCOMES)].sum()
            .unstack('corridor').reindex(months, fill_value=0).fillna(0))
    wide.index.name = 'month'
    for outcome in OUTCOMES:
        wide[(outcome, POOLED)] = wide[outcome][list(COMPARISONS)].sum(axis=1)
    wide = wide.sort_index(axis=1)
    wide[('year', '')] = wide.index.year
    return wide


def period_of(years):
    out = pd.Series(pd.NA, index=years.index, dtype='object')
    for name, start, end in [BEFORE, *CONTRASTS]:
        out[years.between(start, end)] = name
    return out


def pct(x):
    return 100 * np.expm1(x)


def one_sided_drop_p(z):
    """P-value for H1: coefficient < 0 (34th Ave fell more), from a two-sided z statistic."""
    return float(stats.norm.cdf(z))


def did(wide, outcome, control):
    """Proportional and additive DiD of 34th Ave against `control` for every contrast period."""
    df = pd.DataFrame({'treated': wide[(outcome, TREATED)], 'control': wide[(outcome, control)],
                       'period': period_of(wide[('year', '')])}).dropna(subset=['period'])
    X = sm.add_constant(pd.DataFrame({p: df.period.eq(p).astype(float) for p, _, _ in CONTRASTS}))
    nonzero = (df.treated + df.control) > 0
    share = sm.GLM(np.column_stack([df.treated, df.control])[nonzero], X[nonzero],
                   family=sm.families.Binomial()).fit(cov_type='HAC', cov_kwds=HAC)
    gap = sm.OLS(df.treated - df.control, X).fit(cov_type='HAC', cov_kwds=HAC)

    # The conditional binomial must reproduce the Poisson panel with month and corridor fixed effects.
    long = df.reset_index().melt(id_vars=['month', 'period'], value_vars=['treated', 'control'],
                                 var_name='corridor', value_name='y')
    long['is_treated'] = long.corridor.eq('treated').astype(float)
    for p, _, _ in CONTRASTS:
        long[f'treated_{p}'] = long.is_treated * long.period.eq(p)
    poisson = smf.glm('y ~ C(month) + is_treated + ' + ' + '.join(f'treated_{p}' for p, _, _ in CONTRASTS),
                      data=long, family=sm.families.Poisson()).fit()
    for p, _, _ in CONTRASTS:
        assert np.isclose(poisson.params[f'treated_{p}'], share.params[p], atol=1e-6), (outcome, control, p)

    rows = []
    before = df.period.eq(BEFORE[0])
    share_ci, gap_ci = share.conf_int(ALPHA), gap.conf_int(ALPHA)
    for p, start, end in CONTRASTS:
        after = df.period.eq(p)
        per_year = lambda col, mask: 12 * df.loc[mask, col].mean()
        t0, t1, c0, c1 = (per_year('treated', before), per_year('treated', after),
                          per_year('control', before), per_year('control', after))
        assert np.isclose(12 * gap.params[p], (t1 - t0) - (c1 - c0))
        rows.append({
            'outcome': outcome, 'label': OUTCOMES[outcome], 'control': control, 'control_label': LABELS[control],
            'contrast': p, 'contrast_label': CONTRAST_LABELS[p], 'primary': control == PRIMARY and p == MAIN,
            'before_years': f'{BEFORE[1]}-{BEFORE[2]}', 'after_years': f'{start}-{end}',
            'treated_before_per_year': t0, 'treated_after_per_year': t1,
            'control_before_per_year': c0, 'control_after_per_year': c1,
            'treated_change_pct': 100 * (t1 / t0 - 1) if t0 else np.nan,
            'control_change_pct': 100 * (c1 / c0 - 1) if c0 else np.nan,
            'did_ratio_pct': pct(share.params[p]),
            'did_ratio_ci_low': pct(share_ci.loc[p, 0]), 'did_ratio_ci_high': pct(share_ci.loc[p, 1]),
            'did_ratio_z': share.tvalues[p], 'did_ratio_p': share.pvalues[p],
            'did_ratio_drop_p_one_sided': one_sided_drop_p(share.tvalues[p]),
            'did_additive_per_year': 12 * gap.params[p],
            'did_additive_ci_low': 12 * gap_ci.loc[p, 0], 'did_additive_ci_high': 12 * gap_ci.loc[p, 1],
            'did_additive_z': gap.tvalues[p], 'did_additive_p': gap.pvalues[p],
            'did_additive_drop_p_one_sided': one_sided_drop_p(gap.tvalues[p]),
            'pearson_dispersion': float(share.pearson_chi2 / share.df_resid),
            'months_with_crashes': int(nonzero.sum()),
        })
    return rows


def pretrends(wide, outcome, control):
    """Pre-launch differential trend and a yearly event study of 34th Ave's share."""
    df = pd.DataFrame({'treated': wide[(outcome, TREATED)], 'control': wide[(outcome, control)],
                       'year': wide[('year', '')]})
    df = df.loc[(df.treated + df.control) > 0]
    endog = lambda d: np.column_stack([d.treated, d.control])

    pre = df.loc[df.year <= PRE_END]
    t = (pre.index.year + (pre.index.month - 1) / 12) - REFERENCE_YEAR
    trend = sm.GLM(endog(pre), sm.add_constant(pd.DataFrame({'t': t}, index=pre.index)),
                   family=sm.families.Binomial()).fit(cov_type='HAC', cov_kwds=HAC)

    years = [y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y != REFERENCE_YEAR]
    X = sm.add_constant(pd.DataFrame({f'y{y}': df.year.eq(y).astype(float) for y in years}))
    event = sm.GLM(endog(df), X, family=sm.families.Binomial()).fit(cov_type='HAC', cov_kwds=HAC)
    pre_terms = [f'y{y}' for y in years if y < REFERENCE_YEAR]
    joint = event.wald_test(', '.join(f'{term} = 0' for term in pre_terms), scalar=True)
    ci = event.conf_int(ALPHA)
    study = [{'outcome': outcome, 'control': control, 'year': y,
              'ratio_vs_2019_pct': 0.0 if y == REFERENCE_YEAR else pct(event.params[f'y{y}']),
              'ci_low': 0.0 if y == REFERENCE_YEAR else pct(ci.loc[f'y{y}', 0]),
              'ci_high': 0.0 if y == REFERENCE_YEAR else pct(ci.loc[f'y{y}', 1]),
              'p': np.nan if y == REFERENCE_YEAR else event.pvalues[f'y{y}']}
             for y in range(FIRST_YEAR, LAST_YEAR + 1)]
    trend_ci = trend.conf_int(ALPHA).loc['t']
    summary = {'outcome': outcome, 'control': control, 'window': f'{FIRST_YEAR}-{PRE_END}',
               'pre_trend_pct_per_year': pct(trend.params['t']),
               'pre_trend_ci_low': pct(trend_ci[0]), 'pre_trend_ci_high': pct(trend_ci[1]),
               'pre_trend_p': trend.pvalues['t'],
               'event_study_pre_joint_chi2': float(joint.statistic), 'event_study_pre_joint_p': float(joint.pvalue)}
    return summary, study


def verdict(p):
    shown = 'p < 1e-300' if p == 0 else f'p = {p:.2g}'  # float underflow, not an exact zero
    return f'reject H0 ({shown})' if p < ALPHA else f'fail to reject H0 ({shown})'


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.parse_args()

    crashes, audit = select_corridor_crashes()
    wide = monthly_panel(crashes)
    OUT.mkdir(parents=True, exist_ok=True)
    level = int(100 * (1 - ALPHA))

    controls = [*COMPARISONS, POOLED]
    estimates = pd.DataFrame([row for o in OUTCOMES for c in controls for row in did(wide, o, c)])
    checks = [pretrends(wide, o, c) for o in OUTCOMES for c in controls]
    trend_tests = pd.DataFrame([summary for summary, _ in checks])
    event_study = pd.DataFrame([row for _, study in checks for row in study])

    counts = wide.drop(columns='year', level=0).stack('corridor', future_stack=True).reset_index()
    counts.to_csv(OUT / 'corridor_monthly_counts.csv', index=False)
    estimates.to_csv(OUT / 'did_estimates.csv', index=False)
    trend_tests.to_csv(OUT / 'pretrend_tests.csv', index=False)
    event_study.to_csv(OUT / 'event_study.csv', index=False)
    (OUT / 'selection_audit.json').write_text(json.dumps(audit, indent=2))

    print(f'34th Ave difference-in-differences: 69th St to Junction Blvd, '
          f'{BEFORE[1]}-{BEFORE[2]} vs {CONTRASTS[-1][1]}-{CONTRASTS[-1][2]}')
    print(f'Corridor crashes selected: ' + ', '.join(f'{LABELS[n]} {audit[n]["selected_crashes"]:,}'
                                                     for n in [TREATED, *COMPARISONS]))
    if 'matches_newest_selection' in audit[TREATED]:
        print(f'34th Ave selection matches newest/data/34ave_crashes.csv: {audit[TREATED]["matches_newest_selection"]}')
    print('Proportional: binomial share of the two-corridor monthly total (= Poisson with month fixed effects).')
    print('Additive: OLS on the monthly 34th-minus-comparison gap. HAC(12) standard errors throughout.')
    print('One-sided p-values test H1: 34th Ave fell more than the comparison.\n')

    print(f'Primary comparison: {LABELS[PRIMARY]}')
    main_rows = estimates.loc[estimates.primary]
    for r in main_rows.to_dict('records'):
        pre = trend_tests.loc[trend_tests.outcome.eq(r['outcome']) & trend_tests.control.eq(PRIMARY)].iloc[0]
        print(f'{r["label"]}')
        print(f'  per year: 34th Ave {r["treated_before_per_year"]:.1f} -> {r["treated_after_per_year"]:.1f} '
              f'({r["treated_change_pct"]:+.0f}%), {r["control_label"]} {r["control_before_per_year"]:.1f} -> '
              f'{r["control_after_per_year"]:.1f} ({r["control_change_pct"]:+.0f}%)')
        print(f'  proportional DiD: {r["did_ratio_pct"]:+.1f}% '
              f'[{level}% CI {r["did_ratio_ci_low"]:+.1f}, {r["did_ratio_ci_high"]:+.1f}], '
              f'{verdict(r["did_ratio_p"])}; fell more: {verdict(r["did_ratio_drop_p_one_sided"])}')
        print(f'  additive DiD: {r["did_additive_per_year"]:+.1f}/year '
              f'[{level}% CI {r["did_additive_ci_low"]:+.1f}, {r["did_additive_ci_high"]:+.1f}], '
              f'{verdict(r["did_additive_p"])}; fell more: {verdict(r["did_additive_drop_p_one_sided"])}')
        print(f'  pre-trend {pre["window"]}: {pre["pre_trend_pct_per_year"]:+.1f}%/year in 34th Ave share, '
              f'{verdict(pre["pre_trend_p"])}; event-study pre-years joint {verdict(pre["event_study_pre_joint_p"])}; '
              f'Pearson dispersion {r["pearson_dispersion"]:.1f}\n')

    print('Proportional DiD by comparison and period (% change in 34th Ave relative to comparison, p two-sided)')
    table = estimates.assign(cell=[f'{d:+.0f}% (p={p:.2g})' for d, p in zip(estimates.did_ratio_pct, estimates.did_ratio_p)])
    table = table.pivot_table(index=['label', 'control_label'], columns='contrast_label', values='cell', aggfunc='first',
                              sort=False)[[CONTRAST_LABELS[p] for p, _, _ in CONTRASTS]]
    print(table.to_string(), '\n')

    print('Note: one treated and one comparison corridor, possible traffic diversion onto the comparison avenues,\n'
          'changed walking/cycling exposure on the Open Street, and the 2020 COVID and reporting changes mean these\n'
          'are associations, not randomized causal effects.')
    print(f'\nWrote {OUT.relative_to(ROOT)}/did_estimates.csv, pretrend_tests.csv, event_study.csv, '
          'corridor_monthly_counts.csv and selection_audit.json')


if __name__ == '__main__':
    main()
