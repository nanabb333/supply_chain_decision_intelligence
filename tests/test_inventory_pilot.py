from copy import deepcopy
import json,math
from pathlib import Path
import pytest
from supply_chain.inventory_engine import SyntheticFixture,simulate_synthetic
from supply_chain.inventory_protocol import Scenario
from supply_chain.inventory_pilot import (validate_scope,claim_attempt,empirical_metadata,reconcile_episode,
    independent_target,PRIMARY,KIND)


def synthetic_pair():
    f=SyntheticFixture('SYNTHETIC_C2_ADAPTER',((2.,)*28,tuple(.25*i for i in range(1,29)),(3.5,)*28),(3,0,7,1,8,0,0)*3)
    results=simulate_synthetic(f,Scenario())
    bound=dict(item_id=f.name,store_id='SYNTHETIC_STORE',origin=1801,forecasts=dict(zip(('MA28','M1','M3'),f.forecasts)),demand=list(f.demand))
    return results,bound


def test_adapter_changes_metadata_only_and_reconciles():
    results,bound=synthetic_pair()
    for result in results:
        out=empirical_metadata(result)
        assert out['data_kind']==KIND and result['data_kind']=='SYNTHETIC_ONLY'
        assert {k:v for k,v in out.items() if k!='data_kind'}=={k:v for k,v in result.items() if k!='data_kind'}
        verified=reconcile_episode(out,bound)
        assert verified['demand']==sum(bound['demand'])


@pytest.mark.parametrize('field',['receipts','fulfilled','lost','end_inventory','pipeline_end','order_units','target','inventory_position','history_max','holding_cost'])
def test_independent_checker_rejects_corruption(field):
    results,bound=synthetic_pair();out=deepcopy(empirical_metadata(results[0]));out['daily'][0][field]+=1
    with pytest.raises(ValueError):reconcile_episode(out,bound)


@pytest.mark.parametrize('leftover',['ATTEMPT_STARTED.json','WORKER_STARTED.json','WORKER_COMPLETE.json','episodes.jsonl','bound-inputs.jsonl','RESOURCE_USAGE.json'])
def test_once_only_leftovers_rejected(tmp_path,leftover):
    (tmp_path/leftover).write_text('{}')
    with pytest.raises(FileExistsError):claim_attempt(tmp_path,{'attempt':1})


def test_exclusive_claim(tmp_path):
    claim_attempt(tmp_path,{'attempt':1})
    with pytest.raises(FileExistsError):claim_attempt(tmp_path,{'attempt':2})
    assert json.loads((tmp_path/'ATTEMPT_STARTED.json').read_text())['attempt']==1


def test_scope_drift_rejected():
    config=json.loads(Path('configs/week3/checkpoint-c2-v1.json').read_text());b=json.loads(Path('configs/week3/checkpoint-b-v1.json').read_text())
    assert validate_scope(config,b)
    for update in ({'series':129},{'origin':1885},{'episode_days':28},{'max_attempts':2},{'runtime_seconds':120},{'models':['MA28']},{'scenario':dict(PRIMARY,lead_days=1)}):
        with pytest.raises(ValueError):validate_scope(dict(config,**update),b)


@pytest.mark.parametrize('mean',[1/28,2/28,15/28,1.1,0.1,0.25])
def test_independent_binary_sum_preserves_accepted_rounding(mean):
    target=independent_target([mean]*10,3.0,mean)
    assert target==math.fsum([mean]*10)+3.0*mean
    assert math.ceil(target)==math.ceil(math.fsum([mean]*10)+3.0*mean)
