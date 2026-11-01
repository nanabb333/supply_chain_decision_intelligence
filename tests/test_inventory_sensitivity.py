from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import json
import pytest
from supply_chain.inventory_engine import SyntheticFixture,simulate_synthetic
from supply_chain.inventory_protocol import Scenario
from supply_chain.inventory_pilot import empirical_metadata
from supply_chain.inventory_sensitivity import reconcile_scenario,validate_scope_d1

CONFIG=json.loads(Path('configs/week3/checkpoint-d1-v1.json').read_text())


def fixture():
    f=SyntheticFixture('SYNTHETIC_D1',((2.,)*28,tuple(.25*j for j in range(1,29)),(3.5,)*28),(3,0,7,1,8,0,0)*3)
    bound=dict(item_id=f.name,store_id='SYNTHETIC_STORE',origin=1801,forecasts=dict(zip(('MA28','M1','M3'),f.forecasts)),demand=list(f.demand))
    return f,bound


@pytest.mark.parametrize('record',CONFIG['scenarios'],ids=lambda r:r['name'])
def test_every_preregistered_scenario_reconciles(record):
    f,b=fixture();results=simulate_synthetic(f,Scenario(**record))
    assert len({r['initial_inventory'] for r in results})==1
    for result in results:reconcile_scenario(empirical_metadata(result),b,record)


def test_penalty_changes_only_cost_not_operating_decisions():
    f,b=fixture();primary=simulate_synthetic(f,Scenario());penalty=simulate_synthetic(f,Scenario(shortage_per_unit=50))
    for a,z in zip(primary,penalty):
        for x,y in zip(a['daily'],z['daily']):
            assert {k:v for k,v in x.items() if k!='shortage_cost'}=={k:v for k,v in y.items() if k!='shortage_cost'}
        assert z['summary']['shortage_cost']==5*a['summary']['shortage_cost']


@pytest.mark.parametrize('field',['receipts','pipeline_end','lost','holding_cost','shortage_cost','ordering_cost','target','due_day'])
def test_corruption_still_fails(field):
    f,b=fixture();sc=next(r for r in CONFIG['scenarios'] if r['name']=='empty_start');r=deepcopy(empirical_metadata(simulate_synthetic(f,Scenario(**sc))[0]));r['daily'][0][field]+=1
    with pytest.raises(ValueError):reconcile_scenario(r,b,sc)


def test_scope_and_primary_rerun_rejected():
    a=json.loads(Path('configs/week3/checkpoint-a-v1.json').read_text());b=json.loads(Path('configs/week3/checkpoint-b-v1.json').read_text());assert validate_scope_d1(CONFIG,a,b)
    for delta in ({'execute_scenarios':['primary']},{'series':129},{'origin':1885},{'runtime_seconds':120},{'max_attempts':2},{'scenarios':[]}):
        with pytest.raises(ValueError):validate_scope_d1(dict(CONFIG,**delta),a,b)


def test_future_demand_isolation_daily_review():
    f,_=fixture();z=replace(f,demand=f.demand[:9]+(999,)*12)
    for x,y in zip(simulate_synthetic(f,Scenario(review_days=1)),simulate_synthetic(z,Scenario(review_days=1))):
        for a,b in zip(x['daily'][:10],y['daily'][:10]):
            assert a['order_units']==b['order_units'] and a['target']==b['target']
