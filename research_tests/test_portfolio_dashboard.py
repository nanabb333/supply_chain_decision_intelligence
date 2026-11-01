import io,itertools
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from supply_chain.portfolio_data import load,inventory,export,forecast_differences,POLICIES
from supply_chain.portfolio_dashboard import VIEWS

@pytest.mark.parametrize('family,selection',[('operating',x) for x in ['primary','lead1','lead7','daily_review','empty_start','double_start','no_buffer','buffer7','penalty50']]+[('initialization',x) for x in ['primary','history1801_common','symmetric_forecast_common']])
def test_matched_inventory(family,selection):
 d=inventory(family,selection);assert len(d)==3 and d.series.eq(128).all();assert ((d.fulfilled_units+d.lost_units)==d.observed_sales_proxy_units).all();assert (abs(d.fill_percent-100*d.fulfilled_units/d.observed_sales_proxy_units)<1e-10).all()

@pytest.mark.parametrize('policies',[[],['M1'],['M3','MA28'],POLICIES])
def test_policy_filters_and_export(policies):
 d=inventory('initialization','history1801_common',policies);assert set(d.model)==set(policies)
 e=pd.read_csv(io.BytesIO(export(d,'initialization','128series;origin1801;history')));assert len(e)==len(d);assert 'source_artifact' in e and 'evidence_scope' in e

def test_untested_combinations_rejected():
 with pytest.raises(ValueError):inventory('initialization','penalty50')
 with pytest.raises(ValueError):inventory('operating','history1801_common')
 with pytest.raises(ValueError):inventory('invented','primary')

def test_forecast_reconciliation():assert len(forecast_differences())==3

def test_original_primary_agrees_between_families():
 a=inventory();b=inventory('initialization');cols=['model','total_cost','lost_units','pooled_fill','average_on_hand','terminal_inventory'];pd.testing.assert_frame_equal(a[cols].reset_index(drop=True),b[cols].reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-12)

@pytest.mark.parametrize('view',VIEWS)
def test_views_render(view):
 a=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/portfolio_app.py'),default_timeout=15).run();a.selectbox(key='pf_view').set_value(view).run();assert not a.exception

def test_family_switch_empty_and_reset():
 a=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/portfolio_app.py'),default_timeout=15).run();a.selectbox(key='pf_view').set_value(VIEWS[-1]).run();a.radio(key='pf_family').set_value('initialization').run();a.selectbox(key='pf_choice_initialization').set_value('symmetric_forecast_common').run();a.multiselect(key='pf_robust_policies').set_value([]).run();assert not a.exception and any('No policies' in x.value for x in a.info)
 a.button[0].click().run();assert a.selectbox(key='pf_view').value==VIEWS[0] and not a.exception
