from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import json,sys
import pytest
from supply_chain.inventory_protocol import Scenario
from supply_chain.inventory_engine import SyntheticFixture,simulate_synthetic,DemandTape,run_empirical_pilot
from supply_chain.inventory_reconcile import reconcile
from supply_chain.inventory_resources import Limits,violation,supervise


def fixture(demand=None):
    return SyntheticFixture('SYNTHETIC_A',((2.,)*28,tuple(.25*j for j in range(1,29)),(3.5,)*28),demand or (3,0,7,1,8,0,0)*3)


def test_primary_independent_and_deterministic():
    f=fixture();s=Scenario()
    first=simulate_synthetic(f,s)
    assert first==simulate_synthetic(f,s)
    assert len({r['demand_sha256'] for r in first})==1
    assert {r['initial_inventory'] for r in first}=={26}
    assert {r['initial_pipeline'] for r in first}=={0}
    for result in first:assert reconcile(result,f,s)['status']=='PASS'


@pytest.mark.parametrize('name',['primary','lead1','lead7','daily_review','empty_start','double_start','no_buffer','buffer7','penalty50'])
def test_synthetic_config_boundaries(name):
    config=json.loads(Path('configs/week3/checkpoint-b-v1.json').read_text())
    s=Scenario(**next(x for x in config['scenarios'] if x['name']==name))
    f=fixture()
    results=simulate_synthetic(f,s)
    assert len({r['initial_inventory'] for r in results})==1
    for r in results:reconcile(r,f,s)


def test_zero_demand_zero_orders_and_fractional_forecast():
    f=SyntheticFixture('SYNTHETIC_ZERO',((0.,)*28,)*3,(0,)*21)
    for r in simulate_synthetic(f,Scenario()):
        reconcile(r,f,Scenario())
        assert r['summary']['fill_rate'] is None and r['orders']==[] and r['summary']['total_cost']==0
    f=SyntheticFixture('SYNTHETIC_FRACTION',((.25,)*28,)*3,(1,)*21)
    for r in simulate_synthetic(f,Scenario(initial_multiplier=0)):
        reconcile(r,f,Scenario(initial_multiplier=0))
        assert r['daily'][0]['order_units']==4 and r['daily'][0]['receipts']==0
        assert r['daily'][3]['receipts']==4


def test_day21_order_and_terminal_pipeline():
    f=SyntheticFixture('SYNTHETIC_TERMINAL',((1.,)*28,)*3,(1,)*21)
    sc=Scenario(review_days=1,initial_multiplier=0,ordering_per_order=2)
    for r in simulate_synthetic(f,sc):
        reconcile(r,f,sc)
        assert r['daily'][20]['order_units']>0
        assert r['daily'][20]['due_day']==24 and r['summary']['terminal_pipeline']>0
        assert r['summary']['ordering_cost']==2*r['summary']['order_count']
        assert all(e['due_day']>21 for e in r['terminal_orders'])


def test_hand_worked_multiday():
    # Hand calculation: L2+R1 gives target3; initial0; first order3 arrives3.
    # First4 demand(1,1,3,0), receipts(0,0,3,0), lost(1,1,0,0).
    # Day4 order3 arrives6; stock3 on days6..21: holding48, shortage20, order cost4.
    f=SyntheticFixture('SYNTHETIC_HAND',((1.,)*28,)*3,(1,1,3,0)+(0,)*17)
    sc=Scenario(lead_days=2,review_days=1,buffer_days=0,initial_multiplier=0,ordering_per_order=2)
    for r in simulate_synthetic(f,sc):
        assert [x['target'] for x in r['daily'][:4]]==[3,3,3,3]
        assert [x['receipts'] for x in r['daily'][:4]]==[0,0,3,0]
        assert [x['fulfilled'] for x in r['daily'][:4]]==[0,0,3,0]
        assert [x['lost'] for x in r['daily'][:4]]==[1,1,0,0]
        assert [x['order_units'] for x in r['daily'][:4]]==[3,0,0,3]
        assert r['summary']['holding_cost']==48 # arrival6, stock3 at end days6..21
        assert r['summary']['shortage_cost']==20 and r['summary']['ordering_cost']==4
        assert r['summary']['total_cost']==72 and r['summary']['terminal_inventory']==3
        reconcile(r,f,sc)


@pytest.mark.parametrize('field',['begin_inventory','receipts','fulfilled','lost','end_inventory','pipeline_begin','pipeline_end','order_units','holding_cost','history_max','target'])
def test_corruption_detected(field):
    f=fixture();r=deepcopy(simulate_synthetic(f,Scenario())[0]);r['daily'][0][field]+=1
    with pytest.raises(ValueError):reconcile(r,f,Scenario())


def test_order_corruption_detected():
    f=fixture();s=Scenario(initial_multiplier=0)
    base=simulate_synthetic(f,s)[0]
    for field in ('received_ids','order_id'):
        r=deepcopy(base);r['daily'][3][field]=['bad'] if field=='received_ids' else 'bad'
        with pytest.raises(ValueError):reconcile(r,f,s)
    r=deepcopy(base);r['orders'].append(r['orders'][0])
    with pytest.raises(ValueError):reconcile(r,f,s)


def test_reveal_barrier():
    tape=DemandTape((1,)*21)
    with pytest.raises(ValueError):tape.reveal(1)
    with pytest.raises(ValueError):tape.commit_decision(2)
    tape.commit_decision(1)
    with pytest.raises(ValueError):tape.reveal(2)
    assert tape.reveal(1)==1
    with pytest.raises(ValueError):tape.reveal(1)


@pytest.mark.parametrize('cutoff',[1,8,15,21])
def test_future_demand_cannot_change_current_or_prior_decisions(cutoff):
    f=fixture();changed=replace(f,demand=f.demand[:cutoff-1]+(999,)*(22-cutoff))
    for left,right in zip(simulate_synthetic(f,Scenario()),simulate_synthetic(changed,Scenario())):
        for a,b in zip(left['daily'][:cutoff],right['daily'][:cutoff]):
            for field in ('target','order_units','inventory_position','receipts','due_day'):
                assert a[field]==b[field]


def test_empirical_and_invalid_inputs_blocked():
    with pytest.raises(PermissionError):run_empirical_pilot(authorized=True)
    with pytest.raises(PermissionError):simulate_synthetic(object(),Scenario())
    for f in (replace(fixture(),name='M5_REAL'),replace(fixture(),demand=(-1,)*21),replace(fixture(),demand=(1,)*20)):
        with pytest.raises(ValueError):simulate_synthetic(f,Scenario())
    with pytest.raises(ValueError):simulate_synthetic(fixture(),Scenario(episode_days=28))


def test_resource_limits(tmp_path):
    limits=Limits()
    assert violation(0,2*1024**3,limits)=='HARD_MEMORY_STOP'
    assert violation(60,0,limits)=='RUNTIME_GATE'
    assert violation(0,1024**3+1,limits)=='MEMORY_GATE'
    assert violation(59,1024**3,limits) is None
    # Real bounded synthetic subprocess: timeout must terminate it.
    out,err=tmp_path/'out',tmp_path/'err'
    result=supervise([sys.executable,'-c','import time; time.sleep(2)'],out,err,Limits(seconds=.1))
    assert result['status']=='FAIL' and result['stop_reason']=='RUNTIME_GATE'
    with pytest.raises(FileExistsError):supervise([sys.executable,'-c','pass'],out,err)
    with pytest.raises(ValueError):Limits(memory_gate=3,hard_memory_stop=2).validate()
