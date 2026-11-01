import numpy as np
import pandas as pd
import pytest
import duckdb
from supply_chain.population import profile_row,independent_wide_sql,ecdf_distance,categorical_comparison,numerical_comparison

@pytest.mark.parametrize('values',[[0,2,0,0,3,0],[0]*30,[1]*30,[0,np.nan,0,2,0], [0]*5+[1]+[0]*28+[2]+[0]*4])
def test_profile_sql(values):
    p,runs=profile_row(values);days=[f'd_{i+1}' for i in range(len(values))]
    batch=pd.DataFrame([dict(item_id='a',store_id='s')|dict(zip(days,values))])
    with duckdb.connect() as db:
        db.register('batch',batch);q=db.execute(independent_wide_sql(days)).df().iloc[0]
    for k in ['observed_days','units','zero_days','positive_days','mean_units','std_units','cv','adi','leading_zero_days','trailing_zero_days','internal_zero_days']:
        if p[k] is None or pd.isna(p[k]):assert pd.isna(q[k]),k
        else:assert p[k]==pytest.approx(q[k]),k
    assert p['xyz']==q.xyz
    if p['span_defined']:assert sum(runs)==p['internal_zero_days']

def test_internal_run_partition_and_null_break():
    v=[0]*2+[1]+[0]+[1]+[0]*6+[1]+[0]*7+[1]+[0]*28+[1]+[0]*3
    p,r=profile_row(v);assert r.tolist()==[1,6,7,28]
    assert (p['leading_zero_days'],p['trailing_zero_days'],p['internal_zero_days'])==(2,3,42)
    assert [p[f'internal_runs_{b}'] for b in ['1','2-6','7-27','28+']]==[1]*4
    p,r=profile_row([1,0,np.nan,0,1]);assert r.tolist()==[1,1] and not p['xyz_eligible']

def test_zero_span_undefined():
    p,r=profile_row([0]*40);assert not p['span_defined'] and p['internal_zero_days'] is None and len(r)==0 and p['xyz']=='U'
    p,r=profile_row([0,1,0]);assert p['positive_span_days']==1 and p['internal_runs']==0 and p['xyz']=='U'

@pytest.mark.parametrize('values',[[],[-1,1],[1.5,1],[np.inf,1]])
def test_invalid_vectors(values):
    with pytest.raises(ValueError):profile_row(values)

def test_descriptive_distance_and_categorical_denominators():
    assert ecdf_distance([1,2],[1,2])==0
    assert ecdf_distance([1],[1,2])==.5
    p=pd.DataFrame(dict(selected=[True,False,False,True],cat=['a','a','b','b'],units=[1,2,3,4]))
    c=categorical_comparison(p,'cat');assert c.sample_share.sum()==1 and c.population_share.sum()==1
    assert c.share_difference_pp.eq(0).all() and c.selection_coverage.eq(.5).all()
    n=numerical_comparison(p,['units']);assert n.loc[n.statistic=='mean','absolute_difference'].iloc[0]==0
