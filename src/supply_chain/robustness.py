"""Phase 1.6 descriptive diagnostics. No inference of listings or stockouts.

The accepted Phase 1.5 classifier is reused unchanged. Pure functions below never
write to the accepted database or discard zeros from the baseline.
"""
import numpy as np
import pandas as pd
from .segmentation import classify

KEYS=['item_id','store_id']
DEFAULT_THRESHOLDS=dict(a=.8,b=.95,x=.5,y=1.,intermittent=1.32)

def validate_daily(frame):
    needed=set(KEYS+['date','units','sell_price','revenue','cat_id'])
    if not needed.issubset(frame): raise ValueError(f'Missing fields: {needed-set(frame)}')
    if frame.empty: raise ValueError('Empty portfolio')
    d=frame.copy();d['date']=pd.to_datetime(d.date,errors='raise')
    if d[KEYS+['date','cat_id']].isna().any().any() or d.duplicated(KEYS+['date']).any():
        raise ValueError('Null or duplicate daily keys')
    v=d.units.dropna().astype(float)
    if not (np.isfinite(v)&(v>=0)&(v==np.floor(v))).all(): raise ValueError('Invalid unit observations')
    p=d.sell_price.dropna().astype(float)
    if not (np.isfinite(p)&(p>0)).all(): raise ValueError('Invalid prices')
    expected=(d.units*d.sell_price).where(d.units.ne(0),0)
    if not np.allclose(expected.to_numpy(float),d.revenue.to_numpy(float),equal_nan=True,rtol=1e-12,atol=1e-8):
        raise ValueError('Revenue inconsistent with units and weekly price')
    d=d.sort_values(KEYS+['date']).reset_index(drop=True)
    for _,g in d.groupby(KEYS):
        if not g.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError('Daily grid must be contiguous; represent unknown units explicitly as null')
    return d

def lifecycle_metrics(frame):
    d=validate_daily(frame);rows=[]
    for key,g in d.groupby(KEYS,sort=True):
        u=g.units;positive=g.loc[u>0,'date'];first=positive.min();last=positive.max()
        span_defined=not positive.empty;observed=int(u.count());zero=u.eq(0);outside=(g.date<first)|(g.date>last)
        run=longest=0
        for is_zero in zero.fillna(False):
            run=run+1 if is_zero else 0;longest=max(longest,run)
        mean=u.mean();std=u.std(ddof=0);npositive=int(u.gt(0).sum())
        rows.append(dict(zip(KEYS,key))|dict(cat_id=g.cat_id.iloc[0],first_positive_date=first,last_positive_date=last,
            span_state='defined' if span_defined else ('all_zero' if observed==len(g) and zero.all() else 'no_observed_positive_sales'),
            total_calendar_days=len(g),total_observed_days=observed,missing_sales_days=int(u.isna().sum()),
            positive_sales_days=npositive,zero_sales_days=int(zero.sum()),
            leading_zero_days=int((zero&(g.date<first)).sum()) if span_defined else None,
            trailing_zero_days=int((zero&(g.date>last)).sum()) if span_defined else None,
            internal_zero_days=int((zero&g.date.between(first,last)).sum()) if span_defined else None,
            longest_zero_run=longest,full_mean_units=mean,full_std_units=std,
            full_cv=std/mean if pd.notna(mean) and mean>0 else np.nan,
            full_adi=observed/npositive if npositive else np.nan,
            positive_span_days=(last-first).days+1 if span_defined else None,
            outside_span_observed_days=int((outside&u.notna()).sum()) if span_defined else None,
            outside_span_observed_share=float((outside&u.notna()).sum()/observed) if span_defined and observed else np.nan))
    return pd.DataFrame(rows)

def select_window(g,window,end):
    if window=='full':return g
    if window=='last365':return g[g.date>=end-pd.Timedelta(days=364)]
    if window=='positive_span':
        pos=g.loc[g.units>0,'date']
        return g[g.date.between(pos.min(),pos.max())] if not pos.empty else g.iloc[:0]
    if window=='priced_days':return g[g.sell_price.notna()]
    raise ValueError(f'Unknown window: {window}')

def summarize_window(d,window):
    rows=[];end=d.date.max()
    for key,g in d.groupby(KEYS,sort=True):
        s=select_window(g,window,end);u=s.units
        rows.append(dict(zip(KEYS,key))|dict(cat_id=g.cat_id.iloc[0],window=window,
            window_start=s.date.min(),window_end=s.date.max(),calendar_rows=len(s),
            excluded_calendar_rows=len(g)-len(s),observed_days=int(u.count()),missing_sales=int(u.isna().sum()),
            positive_days=int(u.gt(0).sum()),zero_days=int(u.eq(0).sum()),
            units=u.sum(min_count=1),revenue=s.revenue.sum(min_count=1),
            unpriced_positive_days=int(((u>0)&s.sell_price.isna()).sum()),
            mean_units=u.mean(),std_units=u.std(ddof=0)))
    return pd.DataFrame(rows)

def classify_scenario(summary,scenario):
    settings=DEFAULT_THRESHOLDS|scenario.get('thresholds',{})
    s=classify(summary,**settings);s['scenario']=scenario['name']
    usable=(s.missing_sales==0)&(s.observed_days>=28)&(s.mean_units>0)
    # Stabilize only rule comparisons at 12 decimal places; preserve unrounded measures.
    valid=s.abc.ne('U');prior=(s.loc[valid,'eligible_revenue_share'].cumsum()-s.loc[valid,'eligible_revenue_share']).round(12)
    s.loc[valid,'abc']=np.select([prior<round(settings['a'],12),prior<round(settings['b'],12)],['A','B'],default='C')
    s['cv_class']=np.where(usable,np.select([s.cv.round(12)<=round(settings['x'],12),s.cv.round(12)<=round(settings['y'],12)],['X','Y'],default='Z'),'U')
    s['adi_rule_triggered']=usable & (s.adi.round(12)>=round(settings['intermittent'],12))
    s['xyz']=np.where(s.adi_rule_triggered,'Z',s.cv_class)
    s['segment']=s.abc+s.xyz
    s['adi_override_effective']=s.adi_rule_triggered & s.cv_class.ne('Z')
    s['abc_eligible']=s.abc.ne('U');s['xyz_eligible']=s.xyz.ne('U');s['classification_eligible']=s.abc_eligible&s.xyz_eligible
    s['abc_exclusion_reason']=np.where(s.abc_eligible,'',s.abc_reason)
    s['xyz_exclusion_reason']=np.where(s.xyz_eligible,'',s.xyz_reason)
    s.loc[s.calendar_rows==0,['abc_exclusion_reason','xyz_exclusion_reason']]='No observations in scenario window'
    s['positive_sales_frequency']=s.positive_days/s.observed_days.replace(0,np.nan)
    total=s.revenue.sum(min_count=1)
    s['portfolio_revenue_share']=s.revenue/total if pd.notna(total) and total>0 else np.nan
    s['revenue_rank']=pd.Series(pd.NA,index=s.index,dtype='Int64')
    s.loc[s.abc_eligible,'revenue_rank']=range(1,int(s.abc_eligible.sum())+1)
    for k,v in settings.items():s['threshold_'+k]=v
    return s

def scenario_metrics(frame,scenarios):
    d=validate_daily(frame);cache={};out=[]
    for spec in scenarios:
        window=spec['window']
        if window not in cache:cache[window]=summarize_window(d,window)
        out.append(classify_scenario(cache[window],spec))
    return pd.concat(out,ignore_index=True)

def transition_register(metrics,lifecycle):
    base=metrics[metrics.scenario=='baseline'].drop(columns='scenario')
    if len(base)!=len(lifecycle):raise ValueError('Baseline must contain each series once')
    t=metrics.merge(base,on=KEYS,suffixes=('','_baseline'),validate='many_to_one')
    t=t.merge(lifecycle[KEYS+['leading_zero_days','trailing_zero_days','internal_zero_days','span_state','outside_span_observed_share']],on=KEYS,validate='many_to_one')
    def changed(a,b):return ~(a.eq(b).fillna(False)|(a.isna()&b.isna()))
    for name in ['abc','xyz','segment','cv_class','adi_rule_triggered','adi_override_effective','abc_eligible','xyz_eligible','revenue_rank']:
        t[name+'_changed']=changed(t[name],t[name+'_baseline'])
    t['observation_coverage_changed']=t.calendar_rows.ne(t.calendar_rows_baseline)|t.observed_days.ne(t.observed_days_baseline)|changed(t.window_start,t.window_start_baseline)|changed(t.window_end,t.window_end_baseline)
    t['eligibility_changed']=t.abc_eligible_changed|t.xyz_eligible_changed
    t['abc_thresholds_changed']=t.threshold_a.ne(t.threshold_a_baseline)|t.threshold_b.ne(t.threshold_b_baseline)
    t['cv_thresholds_changed']=t.threshold_x.ne(t.threshold_x_baseline)|t.threshold_y.ne(t.threshold_y_baseline)
    t['adi_threshold_changed']=t.threshold_intermittent.ne(t.threshold_intermittent_baseline)
    t['revenue_share_changed']=~np.isclose(t.eligible_revenue_share,t.eligible_revenue_share_baseline,equal_nan=True,rtol=1e-12,atol=1e-12)
    t['possible_lifecycle_boundary_influence']=t.window.eq('positive_span')&t.observation_coverage_changed&((t.leading_zero_days.fillna(0)+t.trailing_zero_days.fillna(0))>0)
    categories=[];interpretations=[]
    for r in t.itertuples():
        labels=[]
        if r.abc_changed or r.revenue_rank_changed:labels.append('Revenue-rank sensitivity')
        if r.cv_class_changed:labels.append('Variability-threshold sensitivity' if r.cv_thresholds_changed else 'CV-class change under observation-window change')
        if r.adi_rule_triggered_changed or r.adi_override_effective_changed:labels.append('Intermittency sensitivity')
        if r.observation_coverage_changed:labels.append('Observation-window sensitivity')
        if r.possible_lifecycle_boundary_influence:labels.append('Possible lifecycle-boundary influence')
        if r.eligibility_changed or not r.classification_eligible:labels.append('Coverage limitation')
        mechanisms=sum([r.abc_changed,r.cv_class_changed,(r.adi_rule_triggered_changed or r.adi_override_effective_changed),r.eligibility_changed])
        if r.segment_changed and mechanisms>1:labels.append('Multiple interacting changes')
        if not r.segment_changed:labels.append('No observed label change')
        categories.append('; '.join(labels))
        interpretations.append('Multiple classification components change; unique attribution is not identified.' if r.segment_changed and mechanisms>1 else ('Observed rule/component change under stated assumptions; no operational cause identified.' if r.segment_changed else 'Final label is unchanged; inspect numerical and coverage differences separately.'))
    t['diagnostic_categories']=categories;t['diagnostic_interpretation']=interpretations
    t['evidence_status']='Computed from accepted observed sales; requires independent validation'
    t['limitations']='Positive-sales span is a retrospective proxy, not verified listing/availability. Windows and thresholds are assumptions; no stockout, promotion or supplier cause inferred.'
    return t

def scenario_summaries(t):
    rows=[]
    for name,s in t.groupby('scenario',sort=False):
        total=s.revenue.sum(min_count=1);baseline_total=s.revenue_baseline.sum(min_count=1)
        row=dict(scenario=name,window=s.window.iloc[0],series=len(s),observed_days=int(s.observed_days.sum()),
            calendar_rows=int(s.calendar_rows.sum()),excluded_calendar_rows=int(s.excluded_calendar_rows.sum()),
            revenue=float(total),eligible_series=int(s.classification_eligible.sum()),changed_labels=int(s.segment_changed.sum()),
            abc_changes=int(s.abc_changed.sum()),xyz_changes=int(s.xyz_changed.sum()),cv_class_changes=int(s.cv_class_changed.sum()),
            adi_trigger_changes=int(s.adi_rule_triggered_changed.sum()),adi_effective_changes=int(s.adi_override_effective_changed.sum()),
            coverage_changes=int(s.observation_coverage_changed.sum()),eligibility_changes=int(s.eligibility_changed.sum()))
        for label in ['X','Y','Z','U']:
            mask=s.xyz.eq(label);row[label+'_count']=int(mask.sum());row[label+'_revenue_share']=float(s.loc[mask,'revenue'].sum()/total) if total>0 else None
        row['AZ_count']=int(s.segment.eq('AZ').sum());row['AZ_revenue_share']=float(s.loc[s.segment.eq('AZ'),'revenue'].sum()/total) if total>0 else None
        for label in ['X','Y','Z','U']:
            mask=s.xyz_baseline.eq('Z')&s.xyz.eq(label)
            row['baseline_Z_to_'+label+'_count']=int(mask.sum())
            row['baseline_Z_to_'+label+'_baseline_revenue_share']=float(s.loc[mask,'revenue_baseline'].sum()/baseline_total) if baseline_total>0 else None
            row['baseline_Z_to_'+label+'_scenario_revenue_share']=float(s.loc[mask,'revenue'].sum()/total) if total>0 else None
        rows.append(row)
    return pd.DataFrame(rows)
