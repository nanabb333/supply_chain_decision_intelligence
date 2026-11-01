"""Week 2 A guardrails only. No model fitting or empirical forecast scoring."""
from datetime import date,timedelta
import hashlib
import numpy as np
import pandas as pd
from .robustness import scenario_metrics
KEYS=['item_id','store_id']

def splits(config):
    h=config['horizon'];origins=config['origins'];roles=config['roles']
    if not isinstance(h,int) or isinstance(h,bool) or h<1:raise ValueError('Invalid horizon')
    if len(origins)!=len(roles) or roles!=['development']*(len(roles)-1)+['holdout']:raise ValueError('One final holdout required')
    if not origins or origins!=sorted(set(origins)):raise ValueError('Origins must increase uniquely')
    start=config['train_start'];minimum=config['min_history']
    if start<1 or minimum<1:raise ValueError('Invalid training bounds')
    anchor=date.fromisoformat(config['anchor']);result=[]
    for i,o in enumerate(origins):
        if not isinstance(o,int) or o-start+1<minimum or o+h>config['last_observed_day']:raise ValueError('Unavailable history/horizon')
        if i and o<origins[i-1]+h:raise ValueError('Evaluation blocks overlap')
        row=dict(role=roles[i],train_start=start,train_end=o,origin=o,target_start=o+1,target_end=o+h,horizon=h)
        for k in ['train_start','train_end','target_start','target_end']:row[k+'_date']=(anchor+timedelta(days=row[k]-1)).isoformat()
        result.append(row)
    return result

def asof_history(frame,origin):
    """Strict boundary: caller must project/slice raw input before this guard."""
    d=frame.copy();d['date']=pd.to_datetime(d.date);o=pd.Timestamp(origin)
    if d.empty or d.date.isna().any() or d.date.gt(o).any():raise ValueError('Empty, missing or future history')
    if d[KEYS].isna().any().any() or d.duplicated(KEYS+['date']).any():raise ValueError('Invalid keys')
    for _,g in d.groupby(KEYS):
        dates=g.date.sort_values()
        if dates.max()!=o or not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():raise ValueError('History must end at origin and be contiguous')
    u=d.units.dropna().to_numpy(float)
    if not (np.isfinite(u)&(u>=0)&(u==np.floor(u))).all():raise ValueError('Invalid sales')
    return d.sort_values(KEYS+['date']).reset_index(drop=True)

def check_lineage(rows,origin):
    """Derived-feature source max and fitted-at dates, not just feature row date."""
    o=pd.Timestamp(origin)
    for r in rows:
        if r['kind'] not in ['observed','calendar_arithmetic']:raise ValueError('Unapproved availability kind')
        for key in ['available_at','fit_cutoff']:
            v=pd.Timestamp(r[key])
            if pd.isna(v) or v>o:raise ValueError('Feature not available at origin')
        source=pd.Timestamp(r['source_max'])
        if pd.isna(source) or (r['kind']!='calendar_arithmetic' and source>o):raise ValueError('Future feature source')
        if r['kind']=='calendar_arithmetic' and r['name'] not in ['weekday','month','day_of_year','horizon']:raise ValueError('Future feature not deterministic calendar')
    return True

def cohort(history,origin,minimum=365,limit=None,seed='week2-a-v1'):
    d=asof_history(history,origin);rows=[]
    for key,g in d.groupby(KEYS,sort=True):
        if len(g)>=minimum and g.units.notna().all():rows.append(key)
    # No activity threshold: all-zero histories retained. No post-origin metadata argument.
    ranked=sorted(rows,key=lambda k:(hashlib.sha256((seed+'|'+repr(k)).encode()).hexdigest(),k))
    if limit is not None:
        if not isinstance(limit,int) or limit<1:raise ValueError('Invalid cohort limit')
        ranked=ranked[:limit]
    return sorted(ranked)

def origin_diagnostics(history,origin):
    """Synthetic-tested adapter: origin-cropped observations + explicit price availability.
    Production weekly prices must first receive week-end+7 availability timestamps.
    """
    d=asof_history(history,origin);o=pd.Timestamp(origin)
    if 'price_available_at' not in d:raise ValueError('Explicit price availability required')
    if 'price_week_end' not in d:raise ValueError('Calendar-derived week end required')
    end=pd.to_datetime(d.price_week_end)
    if end.isna().any() or (end<d.date).any() or (end>d.date+pd.Timedelta(days=6)).any():raise ValueError('Invalid week end')
    release=end+pd.Timedelta(days=7)
    availability=pd.to_datetime(d.price_available_at)
    released=release.le(o)
    d['sell_price']=d.sell_price.where(released&availability.notna()&availability.le(o))
    d['revenue']=(d.units*d.sell_price).where(d.units.ne(0),0)
    result=scenario_metrics(d,[{'name':'baseline','window':'full'}])
    # ABC excludes unreleased weeks across the entire frame, while XYZ retains daily history.
    abcdata=d.loc[released].copy()
    fields=['abc','abc_eligible','abc_exclusion_reason','eligible_revenue_share','cumulative_share','revenue_rank']
    if abcdata.empty:
        result['abc']='U';result['abc_eligible']=False;result['abc_exclusion_reason']='No released weeks'
        result['abc_revenue']=np.nan
        for col in ['eligible_revenue_share','cumulative_share','revenue_rank']:result[col]=np.nan
    else:
        abc=scenario_metrics(abcdata,[{'name':'baseline','window':'full'}])
        result=result.drop(columns=fields).merge(abc[KEYS+fields+['revenue']].rename(columns={'revenue':'abc_revenue'}),on=KEYS,how='left',validate='one_to_one')
        result['abc']=result.abc.fillna('U');result['abc_eligible']=result.abc_eligible.fillna(False)
    result['segment']=result.abc+result.xyz
    result['classification_eligible']=result.abc_eligible & result.xyz_eligible
    # Full-window revenue is deliberately not a forecasting control: use abc_revenue.
    return result.drop(columns=['revenue','portfolio_revenue_share'])

def rolling_at_origin(history,origin,lookback=28):
    d=asof_history(history,origin)
    if not isinstance(lookback,int) or lookback<1:raise ValueError('Invalid lookback')
    d=d[d.date>pd.Timestamp(origin)-pd.Timedelta(days=lookback)]
    return d.groupby(KEYS).units.agg(['mean','count'])

def validate_training_snapshot(feature_origin,target_end,outer_origin,horizon=28):
    if horizon<1 or target_end!=feature_origin+horizon or target_end>outer_origin:raise ValueError('Training label crosses outer origin')
    return True
