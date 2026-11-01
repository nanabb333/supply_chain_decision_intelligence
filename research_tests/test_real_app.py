from pathlib import Path
import duckdb
import pytest
from streamlit.testing.v1 import AppTest
DB=Path('data/processed/m5-real-v1/analytics.duckdb').resolve()
@pytest.mark.skipif(not DB.exists(),reason='Official M5 run not available')
def test_real_dashboard(monkeypatch):
    monkeypatch.setenv('SUPPLY_CHAIN_DB',str(DB))
    a=AppTest.from_file(Path('app/streamlit_app.py').resolve(),default_timeout=60).run()
    assert not a.exception
    with duckdb.connect(str(DB),read_only=True) as db:
        u,r,p=db.sql('select sum(units),sum(revenue),sum(revenue)/sum(units) filter(where sell_price is not null) from daily_sales').fetchone()
        assert [m.value for m in a.metric[:3]]==[f'{u:,.0f}',f'${r:,.2f}',f'${p:,.2f}']
        store,item,cat=db.sql('select store_id,item_id,cat_id from dim_series order by item_id,store_id limit 1').fetchone()
        a.sidebar.selectbox[0].select(store).run()
        a.sidebar.selectbox[1].select(cat).run()
        a.sidebar.selectbox[2].select(item).run()
        assert not a.exception
        assert len(a.get('plotly_chart'))==8  # No magic-expression duplicate single-series plots.
        u,r=db.execute('select sum(units),sum(revenue) from daily_sales where item_id=? and store_id=?',[item,store]).fetchone()
        assert a.metric[0].value==f'{u:,.0f}' and a.metric[1].value==f'${r:,.2f}'
        rank=db.execute('select item_id,store_id,sum(units) units from daily_sales group by all order by units desc,item_id,store_id limit 10').df()
        a.sidebar.selectbox[2].select('All').run();a.sidebar.selectbox[1].select('All').run();a.sidebar.selectbox[0].select('All').run()
        assert a.dataframe[0].value.item_id.tolist()==rank.item_id.tolist()
        # Reconcile a recent date window, prior-period growth, and segment filtering.
        from datetime import date, timedelta
        lo,hi=date(2016,3,1),date(2016,3,31)
        a.sidebar.date_input[0].set_value((lo,hi)).run()
        u,r=db.execute('select sum(units),sum(revenue) from daily_sales where date between ? and ?',[lo,hi]).fetchone()
        prior=db.execute('select sum(units) from daily_sales where date between ? and ?',[lo-timedelta(days=31),lo-timedelta(days=1)]).fetchone()[0]
        assert a.metric[0].value==f'{u:,.0f}' and a.metric[1].value==f'${r:,.2f}'
        assert a.metric[3].value==f'{u/prior-1:.1%}'
        a.sidebar.selectbox[3].select('AZ').run()
        u=db.execute("select sum(d.units) from daily_sales d join sku_segments s using(item_id,store_id) where s.segment='AZ' and date between ? and ?",[lo,hi]).fetchone()[0]
        assert a.metric[0].value==f'{u:,.0f}'
        a.sidebar.selectbox[3].select('All').run()
        a.sidebar.date_input[0].set_value((date(2011,1,29),date(2016,4,24))).run()
        a.slider[0].set_value(.75).run()
        a.slider[1].set_value(.91).run()
        assert not a.exception
        segments=db.sql('select distinct segment from sku_segments').df().segment.tolist()
        a.sidebar.selectbox[0].select(store).run();a.sidebar.selectbox[1].select(cat).run();a.sidebar.selectbox[2].select(item).run()
        own=db.execute('select segment from sku_segments where item_id=? and store_id=?',[item,store]).fetchone()[0]
        other=next(s for s in segments if s!=own)
        a.sidebar.selectbox[3].select(other).run()
        assert not a.exception and any('No records' in i.value for i in a.info)
