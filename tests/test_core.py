from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from supply_chain.fixture import generate
from supply_chain.pipeline import run
from supply_chain.segmentation import classify
from supply_chain.services import connect,load_sales,get_sales_kpis

@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    p=tmp_path_factory.mktemp('data'); generate(p/'raw')
    report=run(p/'raw',p/'out',synthetic=True)
    return p,report


def test_reconciliation(dataset):
    p,report=dataset
    wide=pd.read_csv(p/'raw/sales_train_validation.csv')
    cal=pd.read_csv(p/'raw/calendar.csv')
    price=pd.read_csv(p/'raw/sell_prices.csv')
    long=wide.melt(id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],var_name='d',value_name='units')
    joined=long.merge(cal,on='d',validate='many_to_one').merge(price,on=['item_id','store_id','wm_yr_wk'],how='left',validate='many_to_one')
    expected=(joined.units*joined.sell_price).where(joined.units.ne(0),0).sum()
    assert report['units']==long.units.sum()
    assert report['known_price_revenue']==pytest.approx(expected)
    assert report['rows']==60*365
    assert report['missing_sales']==1
    with connect(p/'out/analytics.duckdb') as db:
        segments=db.sql('SELECT * FROM sku_segments').df()
        assert len(segments)==60
        assert segments.revenue.sum()==pytest.approx(expected)
        assert segments.eligible_revenue_share.sum()==pytest.approx(1)
        assert db.sql('SELECT count(*) FROM daily_sales WHERE units=0 AND revenue!=0').fetchone()[0]==0
        assert db.sql('SELECT count(*) FROM daily_sales WHERE unpriced_positive_sale AND revenue IS NOT NULL').fetchone()[0]==0
        assert db.sql('SELECT count(*) FROM daily_sales WHERE missing_sales AND revenue IS NOT NULL').fetchone()[0]==0


def test_sql_examples(dataset):
    p,_=dataset
    with connect(p/'out/analytics.duckdb') as db:
        for script in Path('sql').glob('*.sql'):
            assert len(db.execute(script.read_text()).fetchall())>0


def test_reproducible(dataset,tmp_path):
    p,_=dataset
    one=run(p/'raw',tmp_path/'one',n=30,synthetic=True)
    two=run(p/'raw',tmp_path/'two',n=30,synthetic=True)
    assert one==two
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/'one/segments.parquet'),pd.read_parquet(tmp_path/'two/segments.parquet'))
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'one/selection.csv'),pd.read_csv(tmp_path/'two/selection.csv'))


@pytest.mark.parametrize('corruption',['duplicate_sales','duplicate_price','negative_sales','invalid_date','missing_calendar','invalid_item','negative_price','missing_column'])
def test_bad_inputs(tmp_path,corruption):
    generate(tmp_path/'raw',n=30,days=35)
    file='sell_prices.csv' if corruption in ['duplicate_price','negative_price'] else ('calendar.csv' if corruption in ['invalid_date','missing_calendar'] else 'sales_train_validation.csv')
    path=tmp_path/'raw'/file; f=pd.read_csv(path)
    if corruption.startswith('duplicate'): f=pd.concat([f,f.iloc[[0]]])
    elif corruption=='negative_sales': f.loc[0,'d_1']=-1
    elif corruption=='invalid_date': f.loc[0,'date']='2015-01-09'
    elif corruption=='missing_calendar': f=f.iloc[1:]
    elif corruption=='invalid_item': f.loc[0,'item_id']='oops'
    elif corruption=='negative_price': f.loc[0,'sell_price']=-1
    elif corruption=='missing_column': f=f.drop(columns='cat_id')
    f.to_csv(path,index=False)
    with pytest.raises((ValueError,TypeError)): run(tmp_path/'raw',tmp_path/'out',synthetic=True)
    assert not (tmp_path/'out/analytics.duckdb').exists()


def test_fixture_requires_disclosure(tmp_path):
    generate(tmp_path/'raw',n=30,days=35)
    with pytest.raises(ValueError,match='--synthetic'): run(tmp_path/'raw',tmp_path/'out')


def test_segmentation_edges():
    data=pd.DataFrame({'item_id':['a','b','c','d','e'],'store_id':['s']*5,'revenue':[80.,15.,5.,0.,100.],
        'units':[100,50,10,0,100],'missing_sales':[0,0,0,0,1],'unpriced_positive_days':[0]*5,
        'std_units':[.1,2.,1.,0.,1.],'mean_units':[1.,1.,1.,0.,1.],
        'observed_days':[100]*5,'positive_days':[100,100,10,0,100]})
    s=classify(data).set_index('item_id')
    assert list(s.loc[['a','b','c'],'abc'])==['A','B','C']
    assert s.loc['a','xyz']=='X'
    assert s.loc['c','xyz']=='Z'
    assert s.loc['d','segment']=='UU'
    assert s.loc['e','segment']=='UU'
    with pytest.raises(ValueError): classify(data,a=.99,b=.9)


def test_kpis_and_filter_scope(dataset):
    p,_=dataset; db=p/'out/analytics.duckdb'
    f=load_sales(db,stores=['CA_1'],start='2015-02-01',end='2015-02-28')
    k=get_sales_kpis(f)
    with connect(db) as con:
        units,rev=con.execute("SELECT sum(units),sum(revenue) FROM daily_sales WHERE store_id='CA_1' AND date BETWEEN '2015-02-01' AND '2015-02-28'").fetchone()
    assert k['units']==units; assert k['revenue']==pytest.approx(rev)
    assert k['growth'] is None
    assert k['asp']==pytest.approx(rev/f.loc[f.sell_price.notna(),'units'].sum())
    assert load_sales(db,items=['nonexistent']).empty
    assert get_sales_kpis(f,f)['growth']==pytest.approx(0)


def test_all_null_and_zero_metrics():
    f=pd.DataFrame({'units':[np.nan],'revenue':[np.nan],'sell_price':[np.nan],'date':[pd.Timestamp('2015-01-01')],
        'missing_sales':[True],'unpriced_positive_sale':[False]})
    k=get_sales_kpis(f)
    assert pd.isna(k['units']) and pd.isna(k['revenue']) and k['asp'] is None and k['growth'] is None
    f.loc[0,['units','revenue']]=0
    f.loc[0,'missing_sales']=False
    assert get_sales_kpis(f,f)['growth'] is None
