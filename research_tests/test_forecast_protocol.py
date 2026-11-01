import copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from supply_chain.forecast_protocol import splits,asof_history,cohort,origin_diagnostics,rolling_at_origin,check_lineage,validate_training_snapshot
R=Path(__file__).resolve().parents[1]
C=json.loads((R/'configs/week2/checkpoint-a-v1.json').read_text())

def frame(n=400):
 d=pd.DataFrame({'item_id':'a','store_id':'s','cat_id':'c','date':pd.date_range('2020-01-01',periods=n),'units':np.resize([0.,1.,2.,0.],n),'sell_price':2.})
 d['price_week_end']=d.date+pd.to_timedelta((4-d.date.dt.dayofweek)%7,unit='D')
 d['price_available_at']=d.price_week_end+pd.Timedelta(days=7)
 return d

def test_split_manifest_and_boundaries():
 expected=json.loads((R/'reports/week2/checkpoint-a-v1/temporal-splits.json').read_text())['splits']
 assert splits(C)==expected==splits(copy.deepcopy(C))
 assert expected[-1]['target_end_date']=='2016-04-24'
 for x in expected:assert x['train_end']<x['target_start'] and x['target_end']-x['target_start']+1==28
 assert expected[-2]['target_end']<expected[-1]['target_start']

@pytest.mark.parametrize('change',[{'origins':[1801,1801,1857,1885]},{'origins':[1801,1810,1857,1885]},{'origins':[1801,1829,1857,1910]},{'horizon':0},{'roles':['holdout']*4},{'min_history':2000}])
def test_bad_splits(change):
 with pytest.raises(ValueError):splits(C|change)

def test_future_rows_rejected_every_entry():
 d=frame();origin=d.date.iloc[-2]
 for fn in [asof_history,cohort,rolling_at_origin,origin_diagnostics]:
  with pytest.raises(ValueError):fn(d,origin)

def test_future_mutation_invariance_after_explicit_projection():
 d=frame();o=d.date.iloc[369];before=d[d.date<=o];changed=d.copy();changed.loc[changed.date>o,['units','sell_price']]=999999
 after=changed[changed.date<=o]
 pd.testing.assert_frame_equal(origin_diagnostics(before,o),origin_diagnostics(after,o))
 pd.testing.assert_frame_equal(rolling_at_origin(before,o),rolling_at_origin(after,o))
 assert cohort(before,o)==cohort(after,o)

def test_rolling_exact_endpoints():
 d=frame();o=d.date.max();x=rolling_at_origin(d,o,7).iloc[0]
 assert x['mean']==pytest.approx(d.units.iloc[-7:].mean()) and x['count']==7

def test_zero_missing_short_cohorts():
 d=frame();d.units=0;o=d.date.max();assert cohort(d,o)==[('a','s')]
 m=origin_diagnostics(d,o).iloc[0];assert m.xyz=='U' and m.abc=='U' and np.isnan(m.cv)
 d.loc[0,'units']=np.nan;assert cohort(d,o)==[]
 assert cohort(frame(30),frame(30).date.max())==[]

def test_cohort_order_and_hash_determinism():
 d=pd.concat([frame().assign(item_id=str(i)) for i in range(10)]);o=d.date.max()
 assert cohort(d,o,limit=3)==cohort(d.sample(frac=1,random_state=7),o,limit=3)
 assert len(cohort(d,o,limit=3))==3

@pytest.mark.parametrize('field',['available_at','source_max','fit_cutoff'])
def test_feature_lineage_rejects_future(field):
 r={'kind':'observed','name':'cv','available_at':'2020-01-01','source_max':'2020-01-01','fit_cutoff':'2020-01-01'};r[field]='2020-01-03'
 with pytest.raises(ValueError):check_lineage([r],'2020-01-02')

def test_calendar_whitelist():
 r={'kind':'calendar_arithmetic','name':'weekday','available_at':'2020-01-01','source_max':'2020-02-01','fit_cutoff':'2020-01-01'}
 assert check_lineage([r],'2020-01-02')
 with pytest.raises(ValueError):check_lineage([r|{'name':'sell_price'}],'2020-01-02')

def test_price_release_and_abc_past_only():
 d=frame();o=d.date.max();released=d.price_week_end+pd.Timedelta(days=7)<=o
 original=origin_diagnostics(d,o).iloc[0]
 d.loc[~released,'sell_price']=1e9
 d.loc[~released,'price_available_at']=o # cannot bypass week-end+7 release
 modified=origin_diagnostics(d,o).iloc[0]
 assert original.abc_revenue==modified.abc_revenue==pytest.approx((d.loc[released,'units']*2).sum())
 assert original.segment==modified.segment
 d.loc[released & d.units.gt(0),'price_available_at']=o+pd.Timedelta(days=1)
 assert origin_diagnostics(d,o).iloc[0].abc=='U'

def test_abc_rank_uses_only_available_revenue():
 a=frame().assign(units=1.);b=frame().assign(item_id='b',units=10.)
 d=pd.concat([a,b]);o=d.date.max();m=origin_diagnostics(d,o).set_index('item_id')
 assert m.loc['b','revenue_rank']==1 and m.loc['a','revenue_rank']==2
 d.loc[(d.item_id=='a') & (d.price_available_at>o),'sell_price']=1e10
 pd.testing.assert_series_equal(m.revenue_rank,origin_diagnostics(d,o).set_index('item_id').revenue_rank)

def test_training_snapshot_cutoff():
 assert validate_training_snapshot(365,393,400)
 with pytest.raises(ValueError):validate_training_snapshot(380,408,400)
 with pytest.raises(ValueError):validate_training_snapshot(365,392,400)

@pytest.mark.parametrize('kind',['duplicate','gap','negative','missing_date'])
def test_bad_history(kind):
 d=frame();o=d.date.max()
 if kind=='duplicate':d=pd.concat([d,d.iloc[[0]]])
 elif kind=='gap':d=d.drop(index=5)
 elif kind=='negative':d.loc[0,'units']=-1
 else:d.loc[0,'date']=pd.NaT
 with pytest.raises(ValueError):asof_history(d,o)
