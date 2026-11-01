import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from supply_chain.robustness import lifecycle_metrics,scenario_metrics,transition_register,DEFAULT_THRESHOLDS
from supply_chain.diagnostic_config import load_config,fresh_target

def frame(units,item='a',prices=None):
    u=pd.Series(units,dtype=float);p=pd.Series(prices if prices is not None else [2.]*len(u),dtype=float)
    return pd.DataFrame(dict(item_id=item,store_id='s',cat_id='c',date=pd.date_range('2020-01-01',periods=len(u)),units=u,sell_price=p,revenue=(u*p).where(u.ne(0),0)))

def test_lifecycle_boundaries():
    row=lifecycle_metrics(frame([0,0,2,0,0,3,0])).iloc[0]
    assert row.first_positive_date==pd.Timestamp('2020-01-03') and row.last_positive_date==pd.Timestamp('2020-01-06')
    assert (row.leading_zero_days,row.trailing_zero_days,row.internal_zero_days,row.longest_zero_run)==(2,1,2,2)
    assert row.positive_span_days==4 and row.outside_span_observed_share==pytest.approx(3/7)
    assert row.full_adi==3.5 and row.full_cv==pytest.approx(np.std([0,0,2,0,0,3,0])/np.mean([0,0,2,0,0,3,0]))

def test_all_zero_and_missing_not_zero():
    z=lifecycle_metrics(frame([0]*30)).iloc[0]
    assert z.span_state=='all_zero' and z.longest_zero_run==30 and pd.isna(z.leading_zero_days) and pd.isna(z.positive_span_days)
    g=lifecycle_metrics(frame([0,np.nan,0,4,0])).iloc[0]
    assert g.longest_zero_run==1 and g.total_observed_days==4 and g.missing_sales_days==1 and g.full_adi==4
    assert (g.leading_zero_days,g.trailing_zero_days,g.internal_zero_days)==(2,1,0)
    assert g.outside_span_observed_share==.75

@pytest.mark.parametrize('units',[[0]*30,[np.nan]*30])
def test_undefined_span_preserves_series(units):
    d=frame(units);m=scenario_metrics(d,[dict(name='baseline',window='full'),dict(name='span',window='positive_span')])
    s=m[m.scenario=='span'].iloc[0]
    assert s.calendar_rows==0 and s.segment=='UU' and not s.classification_eligible
    assert s.abc_exclusion_reason=='No observations in scenario window'

def test_short_span_and_denominators():
    d=frame([0]*10+[3]*20+[0]*10)
    m=scenario_metrics(d,[dict(name='baseline',window='full'),dict(name='span',window='positive_span')]).set_index('scenario')
    assert m.loc['baseline','observed_days']==40 and m.loc['baseline','adi']==2
    assert m.loc['span','observed_days']==20 and m.loc['span','adi']==1 and m.loc['span','xyz']=='U'
    assert m.loc['span','positive_sales_frequency']==1 and m.loc['span','abc']=='A'

def test_priced_day_missing_sales_and_span_internal_null():
    d=frame([1]*30,prices=[np.nan]+[2]*29)
    d.loc[5,['units','revenue']]=np.nan
    m=scenario_metrics(d,[dict(name='baseline',window='full'),dict(name='priced',window='priced_days')]).set_index('scenario')
    assert m.loc['baseline','unpriced_positive_days']==1 and m.loc['baseline','segment']=='UU'
    assert m.loc['priced','calendar_rows']==29 and m.loc['priced','observed_days']==28
    assert m.loc['priced','segment']=='UU'  # Removing absent prices does not erase an internal null sale.

@pytest.mark.parametrize('bad',['duplicate','gap','negative','revenue'])
def test_invalid_daily_input(bad):
    d=frame([1]*30)
    if bad=='duplicate':d=pd.concat([d,d.iloc[[0]]])
    elif bad=='gap':d=d.drop(index=5)
    elif bad=='negative':d.loc[0,'units']=-1
    else:d.loc[0,'revenue']=500
    with pytest.raises(ValueError):lifecycle_metrics(d)

def test_threshold_boundary_and_no_false_cause():
    d=pd.concat([frame([8]*30,'a'),frame([1.]*30,'b'),frame([1.]*30,'c')])
    specs=[dict(name='baseline',window='full'),dict(name='abc',window='full',thresholds={'a':.85})]
    m=scenario_metrics(d,specs);base=m[m.scenario=='baseline'].set_index('item_id')
    assert base.loc['a','abc']=='A' and base.loc['b','abc']=='B' # Prior cumulative = .8.
    t=transition_register(m,lifecycle_metrics(d));b=t[(t.scenario=='abc')&(t.item_id=='b')].iloc[0]
    assert b.abc_changed and not b.xyz_changed and not b.observation_coverage_changed
    assert not b.cv_class_changed and 'Revenue-rank sensitivity' in b.diagnostic_categories

def test_adi_override_distinct_from_cv():
    d=frame(([0]+[1]*4)*20)
    specs=[dict(name='baseline',window='full'),dict(name='adi',window='full',thresholds={'intermittent':1.2})]
    t=transition_register(scenario_metrics(d,specs),lifecycle_metrics(d)).set_index('scenario')
    assert t.loc['baseline','cv']==pytest.approx(.5)
    assert t.loc['adi','adi_override_effective'] and not t.loc['adi','cv_class_changed']
    assert t.loc['adi','xyz_changed'] and not t.loc['adi','abc_changed']

def config_fixture(tmp_path):
    (tmp_path/'raw').mkdir();(tmp_path/'processed').mkdir();(tmp_path/'processed/analytics.duckdb').touch()
    c=dict(root='.',raw_directory='raw',processed_directory='processed',duckdb_path='processed/analytics.duckdb',analytical_output='new-analysis',report_output='new-report',scenarios=[dict(name='baseline',window='full')])
    p=tmp_path/'config.json';p.write_text(json.dumps(c));return p,c

def test_config_and_no_overwrite(tmp_path):
    p,c=config_fixture(tmp_path);v=load_config(p);target=fresh_target(v,'lifecycle');target.mkdir(parents=True);(target/'keep').write_text('protected')
    with pytest.raises(FileExistsError):fresh_target(v,'lifecycle')
    assert (target/'keep').read_text()=='protected'

@pytest.mark.parametrize('bad',['unknown','overlap','database','duplicate','baseline','infinity','scenario'])
def test_invalid_config(tmp_path,bad):
    p,c=config_fixture(tmp_path)
    if bad=='unknown':c['typo']=1
    elif bad=='overlap':c['analytical_output']='processed/subdir'
    elif bad=='database':c['duckdb_path']='raw/no-db'
    elif bad=='duplicate':c['scenarios']*=2
    elif bad=='baseline':c['scenarios'][0]['window']='last365'
    elif bad=='infinity':c['scenarios'][0]['thresholds']={'x':float('inf')}
    else:c['scenarios'][0]['window']='active_listing'
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):load_config(p)

def test_last365_calendar_inclusive_and_single_positive():
    d=frame([1]*400);m=scenario_metrics(d,[dict(name='baseline',window='full'),dict(name='recent',window='last365')]);r=m[m.scenario=='recent'].iloc[0]
    assert r.observed_days==365 and r.window_start==pd.Timestamp('2020-02-05')
    l=lifecycle_metrics(frame([0,2,0])).iloc[0];assert l.positive_span_days==1 and l.internal_zero_days==0

def test_eligibility_transitions_with_nullable_rank():
    d=frame([0]*10+[3]*20+[0]*10)
    specs=[dict(name='baseline',window='full'),dict(name='span',window='positive_span')]
    t=transition_register(scenario_metrics(d,specs),lifecycle_metrics(d)).set_index('scenario')
    assert t.loc['span','eligibility_changed'] and t.loc['span','xyz']=='U'
    z=frame([0]*30);t=transition_register(scenario_metrics(z,specs),lifecycle_metrics(z))
    assert not t.revenue_rank_changed.any() and t.segment.eq('UU').all()
    # Positive sales without baseline prices can become ABC-eligible under priced-only selection.
    d=frame([1]*30,prices=[np.nan]+[2]*29)
    specs=[dict(name='baseline',window='full'),dict(name='priced',window='priced_days')]
    t=transition_register(scenario_metrics(d,specs),lifecycle_metrics(d)).set_index('scenario')
    assert t.loc['priced','abc_eligible_changed'] and t.loc['priced','revenue_rank_changed']


def test_window_cv_change_not_mislabeled_threshold_change():
    d=frame([0]*60+[2]*40)
    specs=[dict(name='baseline',window='full'),dict(name='span',window='positive_span')]
    r=transition_register(scenario_metrics(d,specs),lifecycle_metrics(d)).set_index('scenario').loc['span']
    assert r.cv_class_changed and not r.cv_thresholds_changed
    assert 'CV-class change under observation-window change' in r.diagnostic_categories
    assert 'Multiple interacting changes' in r.diagnostic_categories

@pytest.mark.parametrize('units,prices',[
    ([0]*30,None),([np.nan]*30,None),([1]*30,None),
    ([0]*10+[3]*20+[0]*10,None),([1]*10+[np.nan]+[2]*19,None),
    ([1]*30,[np.nan]+[2]*29)
])
def test_sql_independently_matches_edge_scenarios(units,prices):
    import duckdb
    from supply_chain.robustness_sql import LIFECYCLE_SQL,SCENARIO_SQL,assert_matching
    d=frame(units,prices=prices)
    with duckdb.connect() as db:
        db.register('daily_sales',d)
        l=lifecycle_metrics(d);q=db.execute(LIFECYCLE_SQL).df()
        assert_matching(l,q,[c for c in q if c not in ['item_id','store_id']])
        for window in ['full','last365','positive_span','priced_days']:
            m=scenario_metrics(d,[dict(name='baseline',window=window)])
            q=db.execute(SCENARIO_SQL,DEFAULT_THRESHOLDS|{'window':window}).df()
            assert_matching(m,q,['calendar_rows','observed_days','positive_days','cv','adi','revenue','abc','xyz','segment','cv_class','adi_rule_triggered','adi_override_effective'])
