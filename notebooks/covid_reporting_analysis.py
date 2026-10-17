"""Exploratory tests of the overlapping COVID and NYPD reporting breaks."""
from pathlib import Path
import os, json, hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/nyc-event-mpl')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/covid_reporting_analysis'
OUT.mkdir(parents=True,exist_ok=True)
INPUT=ROOT/'data/processed/daily_policy_panel.csv'
daily=pd.read_csv(INPUT,parse_dates=['date']).set_index('date').sort_index()
assert daily.index.is_unique
assert np.allclose((daily.harm_crashes+daily.non_harm_crashes).dropna(),daily.known_harm.loc[(daily.harm_crashes+daily.non_harm_crashes).notna()])

# Three equal-length, whole-week periods. References are 364 days earlier,
# matching the weekday rather than assuming the same calendar date matches it.
periods={'Before lockdown':('2020-03-09','2020-03-22'),
         'After lockdown, before reporting change':('2020-03-23','2020-04-05'),
         'After reporting change':('2020-04-06','2020-04-19')}
rows=[]
for label,(start,end) in periods.items():
    dates=pd.date_range(start,end); ref=dates-pd.Timedelta(days=364)
    for outcome in ['crashes','harm_crashes','non_harm_crashes']:
        now=daily.loc[dates,outcome]; previous=daily.loc[ref,outcome]
        assert len(now)==14 and now.notna().all() and previous.notna().all()
        rows.append({'period':label,'start':start,'end':end,'outcome':outcome,
                     'total_2020':now.sum(),'daily_mean_2020':now.mean(),
                     'total_reference_2019':previous.sum(),'daily_mean_reference_2019':previous.mean(),
                     'year_over_year_percent':100*(now.sum()/previous.sum()-1)})
period_table=pd.DataFrame(rows)
period_table.to_csv(OUT/'matched_period_counts.csv',index=False)

# A daily Poisson model adjusts the whole-week period comparison for weekday.
# The year × period coefficient compares the change across 2020 periods with
# the change across the same weekdays in 2019. HAC inference allows serial dependence.
matched=[]
for stage,(start,end) in enumerate(periods.values()):
    for date in pd.date_range(start,end):
        for year,observed in [(2020,date),(2019,date-pd.Timedelta(days=364))]:
            row={'date':observed,'year2020':int(year==2020),'stage':stage,'dow':observed.dayofweek}
            row.update(daily.loc[observed,['crashes','harm_crashes','non_harm_crashes']].to_dict())
            matched.append(row)
matched=pd.DataFrame(matched).sort_values('date')
D=patsy.dmatrix('year2020*C(stage)+C(dow)',matched,return_type='dataframe')
contrasts=[]
for outcome in ['crashes','harm_crashes','non_harm_crashes']:
    fit=sm.GLM(matched[outcome],D,family=sm.families.Poisson()).fit(cov_type='HAC',cov_kwds={'maxlags':7})
    # Stage 1 vs stage 0: change before the citywide reporting policy began.
    for name,weights in [('Early change relative to 2019',{'year2020:C(stage)[T.1]':1}),
                         ('Later vs early change relative to 2019',{'year2020:C(stage)[T.2]':1,'year2020:C(stage)[T.1]':-1})]:
        c=pd.Series(0.,index=D.columns)
        for term,value in weights.items(): c[term]=value
        b=float(c@fit.params); se=float(np.sqrt(c@fit.cov_params()@c))
        contrasts.append({'outcome':outcome,'comparison':name,'percent':100*np.expm1(b),
                          'ci_low':100*np.expm1(b-1.96*se),'ci_high':100*np.expm1(b+1.96*se),
                          'p_value':2*norm.sf(abs(b/se))})
contrast=pd.DataFrame(contrasts); contrast.to_csv(OUT/'period_model_contrasts.csv',index=False)

# Segmented daily time series: linear log trends, two steps and slope changes.
# March 23 is the first full day after PAUSE took effect at 8pm March 22.
# The response comparison is no-reported-harm / injury-or-death crashes.
# This is not a randomized control: COVID can change the two types differently.
def fit_breaks(start,end,covid='2020-03-23',report='2020-04-06',label='Main'):
    d=daily.loc[start:end].copy()
    assert (d[['harm_crashes','non_harm_crashes']]>0).all().all()
    d['t']=(d.index-pd.Timestamp(start)).days/7
    for name,date in [('covid',covid),('report',report)]:
        elapsed=(d.index-pd.Timestamp(date)).days
        d[name]=(elapsed>=0).astype(int)
        d[name+'_slope']=np.maximum(elapsed,0)/7
    d['dow']=d.index.dayofweek
    X=patsy.dmatrix('t+covid+covid_slope+report+report_slope+C(dow)',d,return_type='dataframe')
    fits={}; rows=[]
    for outcome in ['harm_crashes','non_harm_crashes','relative_non_harm']:
        if outcome=='relative_non_harm':
            fit=sm.OLS(np.log(d.non_harm_crashes/d.harm_crashes),X).fit(cov_type='HAC',cov_kwds={'maxlags':7})
        else:
            fit=sm.GLM(d[outcome],X,family=sm.families.Poisson()).fit(cov_type='HAC',cov_kwds={'maxlags':7})
        fits[outcome]=fit
        for term in ['covid','report']:
            b,se=fit.params[term],fit.bse[term]
            rows.append({'specification':label,'start':start,'end':end,'covid_break':covid,'report_break':report,
                         'outcome':outcome,'term':term,'percent':100*np.expm1(b),
                         'ci_low':100*np.expm1(b-1.96*se),'ci_high':100*np.expm1(b+1.96*se),
                         'p_value':fit.pvalues[term]})
    return pd.DataFrame(rows),d,X,fits
specifications=[('2020-01-06','2020-06-28','2020-03-23','2020-04-06','Main'),
                ('2020-02-03','2020-05-31','2020-03-23','2020-04-06','Short window'),
                ('2020-01-06','2020-08-30','2020-03-23','2020-04-06','Long window'),
                ('2020-01-06','2020-06-28','2020-03-16','2020-04-06','Earlier COVID break'),
                ('2019-01-07','2019-06-30','2019-03-25','2019-04-08','2019 placebo')]
results=[]
for spec in specifications:
    r,d,X,fits=fit_breaks(*spec); results.append(r)
    if spec[-1]=='Main': main_d,main_X,main_fits=d,X,fits
estimates=pd.concat(results,ignore_index=True)
estimates.to_csv(OUT/'segmented_model_estimates.csv',index=False)

traffic=pd.read_csv(ROOT/'data/processed/daily_crashes_with_traffic_coverage.csv',parse_dates=['date'])
coverage=traffic[traffic.date.between('2020-01-01','2020-06-30')].groupby(traffic.date.dt.to_period('M')).agg(days=('date','size'),traffic_days=('traffic_observed','sum'))
coverage.to_csv(OUT/'traffic_coverage.csv')
assert coverage.loc[pd.Period('2020-04'),'traffic_days']==0

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
colors={'non_harm_crashes':'#d97706','harm_crashes':'#2563b8'}
names={'non_harm_crashes':'No reported injury or death','harm_crashes':'At least one injury or death'}
for outcome in colors:
    baseline=main_d.loc['2020-01-06':'2020-02-02',outcome].mean()
    axes[0].plot(main_d.index,100*main_d[outcome].rolling(7,min_periods=7).mean()/baseline,label=names[outcome],color=colors[outcome],lw=2.5)
axes[0].axhline(100,color='#94a3b8',ls=':',lw=1)
axes[0].set_ylabel('Index: January average = 100')
axes[0].set_title('Both types fell; compare their timing and relative change',loc='left',fontweight='bold')
axes[0].legend(frameon=False,loc='upper right')
ratio=main_d.non_harm_crashes.rolling(7).sum()/main_d.harm_crashes.rolling(7).sum()
axes[1].plot(main_d.index,ratio,label='Observed ratio (7-day totals)',color='#475569',lw=2)
axes[1].plot(main_d.index,np.exp(main_fits['relative_non_harm'].predict(main_X)),label='Segmented model',color='#9333ea',alpha=.75)
axes[1].set_ylabel('No-harm crashes per harm crash')
axes[1].set_title('A relative decline is consistent with reporting changes, but has other explanations',loc='left',fontsize=11)
axes[1].legend(frameon=False)
for ax in axes:
    ax.axvline(pd.Timestamp('2020-03-23'),color='#2563b8',ls='--',lw=1)
    ax.axvline(pd.Timestamp('2020-04-06'),color='#d97706',ls='--',lw=1)
    ax.grid(alpha=.15)
axes[0].text(.02,.08,'Blue dashed line: March 23, first full PAUSE day\nOrange dashed line: April 6, reporting policy',transform=axes[0].transAxes,fontsize=10)
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b 2020'))
fig.suptitle('Why did recorded NYC crashes drop in 2020?',fontsize=19,fontweight='bold',y=.98)
fig.text(.08,.025,'Daily citywide reports. No-harm means no recorded injuries/deaths; it is a proxy for property-damage-only crashes.',fontsize=10,color='#475569')
fig.tight_layout(rect=[0,.05,1,.95]); fig.savefig(OUT/'covid_reporting_comparison.png',dpi=170); plt.close(fig)

m=estimates.query("specification == 'Main' and outcome == 'relative_non_harm' and term == 'report'").iloc[0]
early=contrast[contrast.comparison=='Early change relative to 2019']
lines=['# COVID versus reporting: what can the data distinguish?',
       '', '**Conclusion:** There was a substantial decline before April 6. This rules out the April 6 citywide policy as the sole explanation for the entire decline. It does not isolate COVID as the cause, or allocate percentages of the decline to COVID and reporting.',
       '', '## 1. Did crashes fall before the reporting policy?',
       '', 'Compare March 23–April 5 with March 9–22, then adjust that comparison for the corresponding same-weekday change in 2019 (364 days earlier). Each period has 14 days. PAUSE started at 8pm March 22, so March 23 is the first full day.',
       '', '| Outcome | Adjusted early change | 95% interval |', '|---|---:|---:|']
for r in early.itertuples(): lines.append(f'| {r.outcome} | {r.percent:+.1f}% | {r.ci_low:+.1f}% to {r.ci_high:+.1f}% |')
lines += ['', 'Intervals use Poisson regression with weekday adjustment and a 7-lag HAC covariance. These are exploratory, model-conditional intervals. The short periods and single prior year limit inference. March 9–22 was already affected by the pandemic, so this does not estimate its full effect.',
          '', '## 2. Was there an additional relative shift around April 6?',
          '', f'The main segmented model estimates an immediate **{m.percent:+.1f}%** change in the ratio of no-reported-harm to harm crashes at April 6 (95% interval **{m.ci_low:+.1f}% to {m.ci_high:+.1f}%**, p={m.p_value:.3g}), conditional on the modeled trends and March 23 break.',
          '', 'This is a relative ratio change, not the percent of the total decline caused by reporting. The model also permits slopes to change after both dates. A falling ratio can reflect reduced recording, changed trip mix, speed, injury severity, or other pandemic conditions.',
          '', '| Specification | April reporting step in relative ratio | 95% interval |','|---|---:|---:|']
for r in estimates.query("outcome == 'relative_non_harm' and term == 'report'").itertuples():
    lines.append(f'| {r.specification} | {r.percent:+.1f}% | {r.ci_low:+.1f}% to {r.ci_high:+.1f}% |')
lines += ['', 'The 2019 placebo uses corresponding weekdays, not an actual 2019 citywide intervention. Sensitivity to the window or assumed COVID break signals weak identification. The 2019 Staten Island pilot also complicates the use of 2019 as an unaffected reference.',
          '', '## Model and limits', '',
          'Count models use Poisson log means; the comparative model uses OLS on log(non-harm/harm). Predictors are a linear weekly time trend, weekday indicators, an immediate step and slope change at each break. All use 7-lag HAC standard errors to allow heteroskedasticity and short-run serial correlation. These remain conditional on the mean model and bandwidth; no claim of a randomized experiment is made.',
          'The two main breaks are only 14 days apart. COVID behavior began before PAUSE and changed continuously. Injury crashes are an imperfect comparison series because travel and crash severity affect them differently. No-harm crashes have known zero injuries and deaths; missing harm status is excluded from that classification. Fatalities alone are too sparse for this short daily comparison.',
          '**Traffic adjustment is unavailable at the key dates:** the project traffic panel has zero observed traffic days in April, May and June 2020. We do not interpolate a denominator or treat annual traffic indices as a local causal control.',
          '', '## Reproduce', '', 'Run `.venv/bin/python notebooks/covid_reporting_analysis.py`. Input: the existing aggregate `data/processed/daily_policy_panel.csv`, derived by the policy notebook from the local collision CSV. Raw collision records are not displayed. All tested specifications and exact period counts are exported.',
          '', '## Sources', '',
          '- [NYPD April 3 announcement: citywide policy effective April 6](https://www.nyc.gov/site/nypd/news/p0403b/nypd-expanding-successful-non-injury-collision-pilot-save-nypd-resources)',
          '- [NYC Council notice: PAUSE effective March 22 at 8pm](https://council.nyc.gov/farah-louis/2020/03/21/coronavirus-covid-19-update-governor-cuomo-announces-new-york-state-on-p-a-u-s-e/)',
          '- [NYC Motor Vehicle Collisions—Crashes](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes)']
report='\n'.join(lines); (OUT/'findings.md').write_text(report)
(OUT/'validation.json').write_text(json.dumps({'daily_input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
 'classification_reconciles':True,'equal_14_day_periods':True,'reference_offset_days':364,
 'main_date_count':len(main_d),'specifications':len(specifications),
 'unknown_harm_2020_Jan_June':float((main_d.crashes-main_d.known_harm).sum()),
 'April_to_June_traffic_observed_days':int(coverage.iloc[3:].traffic_days.sum())},indent=2))
# Lightweight browser report includes the full conclusions and figure.
import nbformat,base64
from nbconvert import HTMLExporter
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(report),
    nbformat.v4.new_markdown_cell('![Comparison](attachment:comparison.png)',attachments={'comparison.png':{'image/png':base64.b64encode((OUT/'covid_reporting_comparison.png').read_bytes()).decode()}})])
html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
(OUT/'covid_reporting_analysis.html').write_text(html)
print(early.to_string(index=False)); print(estimates.query("outcome == 'relative_non_harm' and term == 'report'").to_string(index=False))
