"""Hypothesis tests for the COVID-19 break in NYC reported crashes.

1. Segmented (interrupted time-series) regression on monthly counts.
   H0 (level): no immediate change in crashes when COVID hit.
   H0 (slope): no change in the monthly trend after COVID hit.
2. Two-proportion z test on the injury/death share of crashes.
   H0: P(injury or death | crash) before = after.

Reads data/processed/monthly_policy_panel.csv, which analysis/build-panels.py
builds. Writes outputs/covid_statistics/.

Caveats: NYPD stopped requiring police reports for property-only collisions on
April 6, 2020, two weeks after PAUSE began. The all-crash level drop and the
injury share are both inflated by that reporting change, and travel volume also
fell, so these tests describe reported crashes, not per-vehicle risk, and the
breaks are not causal effects of COVID alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'data/processed/monthly_policy_panel.csv'
OUT = ROOT / 'outputs/covid_statistics'
# PAUSE began March 22, 2020, so March is a partial-exposure transition month and is left out.
TRANSITION = pd.Timestamp('2020-03-01')
EVENT = pd.Timestamp('2020-04-01')
OUTCOMES = {'crashes': 'Reported crashes', 'harm_crashes': 'Injury/death crashes',
            'injured': 'People injured', 'killed': 'People killed'}
ALPHA = 0.05


def load_panel(start, end):
    monthly = pd.read_csv(PANEL, parse_dates=['date']).set_index('date').sort_index()
    monthly = monthly.loc[start:end]
    incomplete = monthly.index[~(monthly.complete_month & monthly.complete_outcomes)]
    if len(incomplete):
        raise ValueError(f'incomplete months in window: {[d.strftime("%Y-%m") for d in incomplete]}')
    if not (monthly.index.min() < TRANSITION < monthly.index.max()):
        raise ValueError('window must span March 2020')
    monthly['t'] = np.arange(len(monthly))
    monthly['post'] = (monthly.index >= EVENT).astype(int)
    monthly['after_t'] = np.maximum(monthly.t - monthly.loc[EVENT, 't'], 0)
    return monthly.loc[monthly.index != TRANSITION]


def segmented_regression(monthly):
    """Poisson GLM with a calendar-day offset, month-of-year effects and HAC errors.

    exp(post) is the immediate level ratio at April 2020 versus the extrapolated
    pre-COVID trend; exp(after_t) is the ratio of post- to pre-COVID monthly growth.
    """
    rows = []
    for outcome, label in OUTCOMES.items():
        model = smf.glm(f'{outcome} ~ t + post + after_t + C(month_of_year)', data=monthly,
                        family=sm.families.Poisson(), offset=np.log(monthly.calendar_days)).fit(
                            cov_type='HAC', cov_kwds={'maxlags': 12})
        ci = model.conf_int(ALPHA)
        pct = lambda x: 100 * np.expm1(x)
        joint = model.wald_test('post = 0, after_t = 0', scalar=True)
        rows.append({
            'outcome': outcome, 'label': label,
            'pre_months': int((monthly.post == 0).sum()), 'post_months': int(monthly.post.sum()),
            'pre_trend_pct_per_month': pct(model.params['t']),
            'post_trend_pct_per_month': pct(model.params['t'] + model.params['after_t']),
            'level_change_pct': pct(model.params['post']),
            'level_change_ci_low': pct(ci.loc['post', 0]), 'level_change_ci_high': pct(ci.loc['post', 1]),
            'level_change_z': model.tvalues['post'], 'level_change_p': model.pvalues['post'],
            'slope_change_pct_per_month': pct(model.params['after_t']),
            'slope_change_ci_low': pct(ci.loc['after_t', 0]), 'slope_change_ci_high': pct(ci.loc['after_t', 1]),
            'slope_change_z': model.tvalues['after_t'], 'slope_change_p': model.pvalues['after_t'],
            'joint_wald_chi2': float(joint.statistic), 'joint_wald_p': float(joint.pvalue),
        })
    return pd.DataFrame(rows)


def injury_share_test(monthly):
    """Pooled two-proportion z test of harm crashes / crashes with known harm status.

    The z test treats every crash as independent. A quasi-binomial GLM on monthly
    counts (Pearson-scaled, with month-of-year effects) checks the same contrast
    while allowing for month-to-month overdispersion.
    """
    before, after = monthly.loc[monthly.post == 0], monthly.loc[monthly.post == 1]
    counts = np.array([after.harm_crashes.sum(), before.harm_crashes.sum()])
    nobs = np.array([after.known_harm.sum(), before.known_harm.sum()])
    z, p = proportions_ztest(counts, nobs, alternative='two-sided')
    diff_low, diff_high = confint_proportions_2indep(counts[0], nobs[0], counts[1], nobs[1],
                                                     method='wald', compare='diff', alpha=ALPHA)
    ratio_low, ratio_high = confint_proportions_2indep(counts[0], nobs[0], counts[1], nobs[1],
                                                       method='log', compare='ratio', alpha=ALPHA)
    p_after, p_before = counts / nobs

    monthly = monthly.assign(non_harm=monthly.known_harm - monthly.harm_crashes)
    formula = 'harm_crashes + non_harm ~ post + C(month_of_year)'
    binomial = smf.glm(formula, data=monthly, family=sm.families.Binomial()).fit()
    dispersion = binomial.pearson_chi2 / binomial.df_resid
    quasi = smf.glm(formula, data=monthly, family=sm.families.Binomial()).fit(scale=dispersion)
    or_low, or_high = np.exp(quasi.conf_int(ALPHA).loc['post'])
    return {
        'before_months': f'{before.index.min():%Y-%m} to {before.index.max():%Y-%m}',
        'after_months': f'{after.index.min():%Y-%m} to {after.index.max():%Y-%m}',
        'before_harm_crashes': int(counts[1]), 'before_known_crashes': int(nobs[1]), 'p_before': p_before,
        'after_harm_crashes': int(counts[0]), 'after_known_crashes': int(nobs[0]), 'p_after': p_after,
        'difference_pp': 100 * (p_after - p_before),
        'difference_ci_pp': [100 * diff_low, 100 * diff_high],
        'risk_ratio': p_after / p_before, 'risk_ratio_ci': [ratio_low, ratio_high],
        'z': float(z), 'p_value': float(p),
        'quasi_binomial_odds_ratio': float(np.exp(quasi.params['post'])),
        'quasi_binomial_odds_ratio_ci': [float(or_low), float(or_high)],
        'quasi_binomial_p': float(quasi.pvalues['post']),
        'quasi_binomial_dispersion': float(quasi.scale),
    }


def verdict(p):
    shown = 'p < 1e-300' if p == 0 else f'p = {p:.2g}'  # float underflow, not an exact zero
    return f'reject H0 ({shown})' if p < ALPHA else f'fail to reject H0 ({shown})'


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--start', default='2017-01-01', help='first month of the window (default after the 2016 recording change)')
    parser.add_argument('--end', default='2022-12-01', help='last month of the window')
    args = parser.parse_args()

    monthly = load_panel(args.start, args.end)
    OUT.mkdir(parents=True, exist_ok=True)
    level = int(100 * (1 - ALPHA))

    seg = segmented_regression(monthly)
    seg.to_csv(OUT / 'segmented_regression.csv', index=False)
    print(f'Segmented regression, {args.start[:7]} to {args.end[:7]} '
          f'({seg.pre_months[0]} pre / {seg.post_months[0]} post months, March 2020 excluded)')
    print('Poisson GLM, calendar-day offset, month-of-year effects, HAC(12) standard errors\n')
    for r in seg.to_dict('records'):
        print(f'{r["label"]}')
        print(f'  trend: {r["pre_trend_pct_per_month"]:+.2f}%/month before, {r["post_trend_pct_per_month"]:+.2f}%/month after')
        print(f'  level change at Apr 2020: {r["level_change_pct"]:+.1f}% '
              f'[{level}% CI {r["level_change_ci_low"]:+.1f}, {r["level_change_ci_high"]:+.1f}], {verdict(r["level_change_p"])}')
        print(f'  slope change: {r["slope_change_pct_per_month"]:+.2f} pts/month '
              f'[{level}% CI {r["slope_change_ci_low"]:+.2f}, {r["slope_change_ci_high"]:+.2f}], {verdict(r["slope_change_p"])}')
        print(f'  joint (level and slope): chi2 = {r["joint_wald_chi2"]:.1f}, {verdict(r["joint_wald_p"])}\n')

    share = injury_share_test(monthly)
    (OUT / 'injury_share_ztest.json').write_text(json.dumps(share, indent=2) + '\n')
    print('Two-proportion z test: share of crashes (known harm status) with an injury or death')
    print(f'  before ({share["before_months"]}): {share["before_harm_crashes"]:,} / {share["before_known_crashes"]:,} '
          f'= {100 * share["p_before"]:.2f}%')
    print(f'  after  ({share["after_months"]}): {share["after_harm_crashes"]:,} / {share["after_known_crashes"]:,} '
          f'= {100 * share["p_after"]:.2f}%')
    lo, hi = share['difference_ci_pp']
    rlo, rhi = share['risk_ratio_ci']
    print(f'  difference: {share["difference_pp"]:+.2f} pp [{level}% CI {lo:+.2f}, {hi:+.2f}]; '
          f'ratio {share["risk_ratio"]:.3f} [{rlo:.3f}, {rhi:.3f}]')
    print(f'  z = {share["z"]:.1f}, {verdict(share["p_value"])}')
    olo, ohi = share['quasi_binomial_odds_ratio_ci']
    print(f'  overdispersion check (quasi-binomial, dispersion {share["quasi_binomial_dispersion"]:.1f}): '
          f'OR {share["quasi_binomial_odds_ratio"]:.2f} [{olo:.2f}, {ohi:.2f}], {verdict(share["quasi_binomial_p"])}')
    print('\nNote: the April 6, 2020 end of mandatory police reports for property-only crashes coincides with '
          'the break,\nso the all-crash drop and the injury-share rise partly reflect what gets reported.')
    print(f'\nWrote {OUT.relative_to(ROOT)}/segmented_regression.csv and injury_share_ztest.json')


if __name__ == '__main__':
    main()
