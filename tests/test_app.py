from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from supply_chain.fixture import generate
from supply_chain.pipeline import run
from supply_chain.services import load_sales,get_sales_kpis


def test_dashboard_filters_and_metrics(tmp_path,monkeypatch):
    generate(tmp_path/'raw',n=30,days=70)
    run(tmp_path/'raw',tmp_path/'out',synthetic=True)
    db=tmp_path/'out/analytics.duckdb'
    monkeypatch.setenv('SUPPLY_CHAIN_DB',str(db))
    app=AppTest.from_file(Path('app/streamlit_app.py').resolve(),default_timeout=60).run()
    assert not app.exception
    expected=get_sales_kpis(load_sales(db))['units']
    assert app.metric[0].value==f'{expected:,.0f}'
    assert any('SYNTHETIC' in w.value for w in app.warning)
    app.sidebar.selectbox[0].select('CA_1').run()
    assert not app.exception
    expected=get_sales_kpis(load_sales(db,stores=['CA_1']))['units']
    assert app.metric[0].value==f'{expected:,.0f}'
    app.sidebar.selectbox[0].select('All').run()
    assert not app.exception
    app.sidebar.date_input[0].set_value((__import__('datetime').date(2015,2,1),__import__('datetime').date(2015,2,28))).run()
    assert not app.exception
    expected=get_sales_kpis(load_sales(db,start='2015-02-01',end='2015-02-28'))['units']
    assert app.metric[0].value==f'{expected:,.0f}'
    app.sidebar.selectbox[3].select('UU').run()
    assert not app.exception
    app.slider[0].set_value(.7).run()
    assert not app.exception


def test_missing_database(tmp_path,monkeypatch):
    monkeypatch.setenv('SUPPLY_CHAIN_DB',str(tmp_path/'absent.duckdb'))
    app=AppTest.from_file(Path('app/streamlit_app.py').resolve()).run()
    assert not app.exception and len(app.info)==1
