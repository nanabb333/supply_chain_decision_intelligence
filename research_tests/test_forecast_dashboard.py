import io,json
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from supply_chain import forecast_dashboard_data as d

@pytest.fixture(scope='module')
def allrows():return d.query_cohort()

def test_headlines_reconcile(allrows):
 k=d.kpis();p=d.load('paired').set_index('comparison');assert k['series']==allrows[['item_id','store_id']].drop_duplicates().shape[0]
 s=d.summarize(allrows).set_index('model');assert np.isclose(k['improvement'],100*(s.loc['M1','MAE']-s.loc['M3','MAE'])/s.loc['M1','MAE'])
 series=allrows.groupby(['item_id','store_id'])[['M1_mae','M3_mae']].mean();assert np.isclose(k['improved_pct'],100*((series.M3_mae-series.M1_mae)<-1e-10).mean())
 assert np.isclose(s.loc['MA28','MAE'],d.load('baseline_ranking').query("model=='ma28'").macro_mae.iloc[0])

@pytest.mark.parametrize('name',list(d.FILTERS))
def test_each_filter_matches_independent_pandas(name,allrows):
 col=d.FILTERS[name];value=allrows[col].iloc[0];value=int(value) if name=='origin' else str(value)
 actual=d.query_cohort({name:[value]});expected=allrows[allrows[col]==value]
 assert len(actual)==len(expected)
 assert set(map(tuple,actual[['item_id','store_id','origin']].to_numpy()))==set(map(tuple,expected[['item_id','store_id','origin']].to_numpy()))
 np.testing.assert_allclose(d.summarize(actual).MAE,d.summarize(expected).MAE)

def test_single_empty_and_zero_reference(allrows):
 row=allrows.iloc[0];f={'origin':[int(row.origin)],'item':[row.item_id],'store':[row.store_id]};one=d.query_cohort(f);assert len(one)==1 and d.summarize(one).Series.eq(1).all()
 f['category']=[next(c for c in allrows.cat_id.unique() if c!=row.cat_id)];empty=d.query_cohort(f);assert empty.empty and d.summarize(empty).empty and not d.paired_scope(empty)['available']
 one=one.copy();one.M1_mae=0;assert d.paired_scope(one)['relative_pct'] is None

def test_exports_exact_scope_and_metadata(allrows):
 sub=allrows[allrows.origin==d.origins()[0]];s=d.summarize(sub);b=d.export_csv(s,['losses','memberships'],'test cohort','MAE in units/day',scope_origins=[d.origins()[0]])
 output=pd.read_csv(io.BytesIO(b));np.testing.assert_allclose(output.MAE,s.MAE);assert output.cohort_scope.eq('test cohort').all()
 for col in ['model','scope_origins','metric_definition','source_artifact','research_limitations','weighting','source_period']:assert output[col].notna().all()
 assert output.scope_origins.eq(json.dumps([d.origins()[0]])).all()

def test_guards():
 with pytest.raises(ValueError):d.query_cohort({'origin':[1885]})
 with pytest.raises(ValueError):d.load('../../data/raw/sales_train_validation.csv')
 with pytest.raises(ValueError):d.query_cohort({'anything':['value']})
 with pytest.raises(ValueError):d.export_csv(pd.DataFrame(),['losses'],'x','x',scope_origins=[1885])
 with pytest.raises(ValueError):d.load('losses')

def test_diagnostic_scope_and_reconciliation():
 c=d.load('classes');q=d.load('quality');w=d.load('windows')
 for row in q.itertuples():assert c[c.cohort==row.cohort].series.sum()==row.series
 assert set(w.cohort)=={'Selected sample'} and (w.changed<=w.series).all()
 assert (q.adi_effective_count<=q.adi_trigger_count).all()

@pytest.mark.parametrize('view',['Executive overview','Demand & classification','Forecast benchmarking','XYZ incremental value','Subgroup explorer','Research evidence'])
def test_all_views(view):
 at=AppTest.from_file(str(d.ROOT/'app/forecast_app.py'),default_timeout=30).run();at.selectbox(key='forecast_view').set_value(view).run()
 assert not at.exception and not at.error
 assert len(at.header)>0

def test_ui_filter_reset_empty_navigation():
 at=AppTest.from_file(str(d.ROOT/'app/forecast_app.py'),default_timeout=30).run();at.selectbox(key='forecast_view').set_value('Subgroup explorer').run()
 at.multiselect(key='filter_item').set_value(['FOODS_1_001']);at.multiselect(key='filter_category').set_value(['HOBBIES']).run()
 assert any('No matched observations' in w.value for w in at.warning)
 at.button[0].click().run();assert not at.exception and at.metric[0].value=='30,490'
 at.selectbox(key='forecast_view').set_value('Executive overview').run();assert at.metric[0].value=='30,490'
 at.selectbox(key='forecast_view').set_value('Forecast benchmarking').run();at.selectbox(key='benchmark_origin').set_value('1801').run();assert not at.exception and not at.error
