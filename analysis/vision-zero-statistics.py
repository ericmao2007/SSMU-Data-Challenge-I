"""Segmented regression test of whether monthly injuries and deaths dropped after Vision Zero.

Interrupted time-series model on monthly counts, one per outcome:
   H0 (level): no immediate change in the monthly count when Vision Zero began.
   H0 (slope): no change in the monthly trend after Vision Zero began.
   H0 (12 months): no difference 12 months in versus the extrapolated pre-launch trend.
The "did it drop" question is answered with one-sided tests (H1: change < 0)
alongside the usual two-sided ones.

Reads data/processed/monthly_policy_panel.csv, which analysis/build-panels.py
builds. Writes outputs/vision_zero_statistics/.

Caveats: the collision data begin in July 2012, so there are only 18 pre-launch
months. Vision Zero is a package (the 25 mph default limit followed in November
2014, plus cameras and street redesigns), there is no comparison city, and the
crash recording system changed in March 2016. Each model is fit with and without
an indicator for that recording change; neither version is a randomized estimate
of the policy's causal effect.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'data/processed/monthly_policy_panel.csv'
OUT = ROOT / 'outputs/vision_zero_statistics'
# Vision Zero was announced January 15, 2014, so January is a partial-exposure
# transition month and is left out; February is the first full post-launch month.
TRANSITION = pd.Timestamp('2014-01-01')
EVENT = pd.Timestamp('2014-02-01')
RECORDING_CHANGE = pd.Timestamp('2016-03-01')
OUTCOMES = {'injured': 'People injured', 'killed': 'People killed',
            'ped_injured': 'Pedestrians injured', 'ped_killed': 'Pedestrians killed'}
SPECS = {'base': 'Trend + seasonality', 'recording': '+ March 2016 recording-change indicator'}
HORIZON = 12
ALPHA = 0.05


def load_panel(start, end):
    monthly = pd.read_csv(PANEL, parse_dates=['date']).set_index('date').sort_index()
    monthly = monthly.loc[start:end]
    incomplete = monthly.index[~(monthly.complete_month & monthly.complete_outcomes)]
    if len(incomplete):
        raise ValueError(f'incomplete months in window: {[d.strftime("%Y-%m") for d in incomplete]}')
    if not (monthly.index.min() < TRANSITION < monthly.index.max()):
        raise ValueError('window must span January 2014')
    monthly['t'] = np.arange(len(monthly))
    monthly['post'] = (monthly.index >= EVENT).astype(int)
    monthly['after_t'] = np.maximum(monthly.t - monthly.loc[EVENT, 't'], 0)
    monthly['recording'] = (monthly.index >= RECORDING_CHANGE).astype(int)
    return monthly.loc[monthly.index != TRANSITION]


def one_sided_drop_p(z):
    """P-value for H1: coefficient < 0 (a drop), from a two-sided z statistic."""
    return float(stats.norm.cdf(z))


def fit(monthly, outcome, spec):
    formula = f'{outcome} ~ t + post + after_t + C(month_of_year)'
    if spec == 'recording':
        if not monthly.recording.any():
            return None
        formula += ' + recording'
    return smf.glm(formula, data=monthly, family=sm.families.Poisson(),
                   offset=np.log(monthly.calendar_days)).fit(cov_type='HAC', cov_kwds={'maxlags': 12})


def segmented_regression(monthly):
    """Poisson GLM with a calendar-day offset, month-of-year effects and HAC errors.

    exp(post) is the immediate level ratio at February 2014 versus the extrapolated
    pre-launch trend; exp(after_t) is the ratio of post- to pre-launch monthly growth;
    exp(post + 12 * after_t) is the ratio 12 months after February 2014. HAC errors
    cover overdispersion and autocorrelation; Pearson dispersion is reported as a check.
    """
    rows, curves = [], []
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    pct = lambda x: 100 * np.expm1(x)
    for outcome, label in OUTCOMES.items():
        for spec, spec_label in SPECS.items():
            model = fit(monthly, outcome, spec)
            if model is None:
                continue
            ci = model.conf_int(ALPHA)
            weights = pd.Series(0.0, index=model.params.index)
            weights['post'], weights['after_t'] = 1, HORIZON
            contrast = float(weights @ model.params)
            contrast_se = float(np.sqrt(weights @ model.cov_params() @ weights))
            contrast_z = contrast / contrast_se
            joint = model.wald_test('post = 0, after_t = 0', scalar=True)
            rows.append({
                'outcome': outcome, 'label': label, 'spec': spec, 'spec_label': spec_label,
                'pre_months': int((monthly.post == 0).sum()), 'post_months': int(monthly.post.sum()),
                'pre_mean_per_month': monthly.loc[monthly.post == 0, outcome].mean(),
                'post_mean_per_month': monthly.loc[monthly.post == 1, outcome].mean(),
                'pre_trend_pct_per_month': pct(model.params['t']),
                'post_trend_pct_per_month': pct(model.params['t'] + model.params['after_t']),
                'level_change_pct': pct(model.params['post']),
                'level_change_ci_low': pct(ci.loc['post', 0]), 'level_change_ci_high': pct(ci.loc['post', 1]),
                'level_change_z': model.tvalues['post'], 'level_change_p': model.pvalues['post'],
                'level_drop_p_one_sided': one_sided_drop_p(model.tvalues['post']),
                'slope_change_pct_per_month': pct(model.params['after_t']),
                'slope_change_ci_low': pct(ci.loc['after_t', 0]), 'slope_change_ci_high': pct(ci.loc['after_t', 1]),
                'slope_change_z': model.tvalues['after_t'], 'slope_change_p': model.pvalues['after_t'],
                'slope_drop_p_one_sided': one_sided_drop_p(model.tvalues['after_t']),
                f'change_at_{HORIZON}m_pct': pct(contrast),
                f'change_at_{HORIZON}m_ci_low': pct(contrast - z_crit * contrast_se),
                f'change_at_{HORIZON}m_ci_high': pct(contrast + z_crit * contrast_se),
                f'change_at_{HORIZON}m_z': contrast_z,
                f'change_at_{HORIZON}m_p': float(2 * stats.norm.sf(abs(contrast_z))),
                f'change_at_{HORIZON}m_drop_p_one_sided': one_sided_drop_p(contrast_z),
                'joint_wald_chi2': float(joint.statistic), 'joint_wald_p': float(joint.pvalue),
                'pearson_dispersion': float(model.pearson_chi2 / model.df_resid),
            })
            # Counterfactual: same fit with the Vision Zero terms switched off.
            counterfactual = monthly.assign(post=0, after_t=0)
            curves.append(pd.DataFrame({
                'date': monthly.index, 'outcome': outcome, 'spec': spec, 'observed': monthly[outcome].to_numpy(),
                'fitted': model.predict(monthly, offset=np.log(monthly.calendar_days)).to_numpy(),
                'counterfactual': model.predict(counterfactual, offset=np.log(monthly.calendar_days)).to_numpy(),
            }))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def verdict(p):
    shown = 'p < 1e-300' if p == 0 else f'p = {p:.2g}'  # float underflow, not an exact zero
    return f'reject H0 ({shown})' if p < ALPHA else f'fail to reject H0 ({shown})'


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--start', default='2012-07-01', help='first month of the window (default: first month of data)')
    parser.add_argument('--end', default='2019-12-01', help='last month of the window (default stops before COVID)')
    args = parser.parse_args()

    monthly = load_panel(args.start, args.end)
    OUT.mkdir(parents=True, exist_ok=True)
    level = int(100 * (1 - ALPHA))

    seg, curves = segmented_regression(monthly)
    seg.to_csv(OUT / 'segmented_regression.csv', index=False)
    curves.to_csv(OUT / 'fitted_and_counterfactual.csv', index=False)
    print(f'Vision Zero segmented regression, {args.start[:7]} to {args.end[:7]} '
          f'({seg.pre_months[0]} pre / {seg.post_months[0]} post months, January 2014 excluded)')
    print('Poisson GLM, calendar-day offset, month-of-year effects, HAC(12) standard errors')
    print('One-sided p-values test H1: drop (change < 0)\n')
    for r in seg.to_dict('records'):
        print(f'{r["label"]} [{r["spec_label"]}]')
        print(f'  mean per month: {r["pre_mean_per_month"]:,.1f} before, {r["post_mean_per_month"]:,.1f} after')
        print(f'  trend: {r["pre_trend_pct_per_month"]:+.2f}%/month before, {r["post_trend_pct_per_month"]:+.2f}%/month after')
        print(f'  level change at Feb 2014: {r["level_change_pct"]:+.1f}% '
              f'[{level}% CI {r["level_change_ci_low"]:+.1f}, {r["level_change_ci_high"]:+.1f}], '
              f'{verdict(r["level_change_p"])}; drop: {verdict(r["level_drop_p_one_sided"])}')
        print(f'  slope change: {r["slope_change_pct_per_month"]:+.2f} pts/month '
              f'[{level}% CI {r["slope_change_ci_low"]:+.2f}, {r["slope_change_ci_high"]:+.2f}], '
              f'{verdict(r["slope_change_p"])}; drop: {verdict(r["slope_drop_p_one_sided"])}')
        print(f'  change {HORIZON} months in vs. pre-trend: {r[f"change_at_{HORIZON}m_pct"]:+.1f}% '
              f'[{level}% CI {r[f"change_at_{HORIZON}m_ci_low"]:+.1f}, {r[f"change_at_{HORIZON}m_ci_high"]:+.1f}], '
              f'{verdict(r[f"change_at_{HORIZON}m_p"])}; drop: {verdict(r[f"change_at_{HORIZON}m_drop_p_one_sided"])}')
        print(f'  joint (level and slope): chi2 = {r["joint_wald_chi2"]:.1f}, {verdict(r["joint_wald_p"])}; '
              f'Pearson dispersion {r["pearson_dispersion"]:.1f}\n')

    print('Note: only 18 pre-launch months are available, Vision Zero bundles several measures, and the\n'
          'March 2016 recording change shifts reported counts, so these are associations, not causal effects.')
    print(f'\nWrote {OUT.relative_to(ROOT)}/segmented_regression.csv and fitted_and_counterfactual.csv')


if __name__ == '__main__':
    main()
