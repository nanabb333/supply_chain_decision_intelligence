import pandas as pd
import pytest
from supply_chain.compatibility import calendar_days
from supply_chain.pipeline import run
from supply_chain.fixture import generate

def test_date_anchor_not_row_order():
    f=pd.DataFrame({'date':['2011-01-31','2011-01-29','2011-01-30'],'weekday':['Monday','Saturday','Sunday'],'wday':[3,1,2]})
    assert calendar_days(f).d.tolist()==['d_3','d_1','d_2']

def test_bad_anchor():
    with pytest.raises(ValueError,match='anchor'):calendar_days(pd.DataFrame({'date':['2011-01-30']}))

def test_weekday_mismatch():
    with pytest.raises(ValueError,match='weekday'):calendar_days(pd.DataFrame({'date':['2011-01-29'],'weekday':['Monday']}))

def test_existing_output_is_preserved(tmp_path):
    generate(tmp_path/'raw',30,35)
    run(tmp_path/'raw',tmp_path/'out',synthetic=True)
    before=(tmp_path/'out/analytics.duckdb').read_bytes()
    with pytest.raises(FileExistsError):run(tmp_path/'raw',tmp_path/'out',synthetic=True)
    assert (tmp_path/'out/analytics.duckdb').read_bytes()==before
