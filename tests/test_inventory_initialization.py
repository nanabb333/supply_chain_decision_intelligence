from copy import deepcopy
from dataclasses import asdict,replace
import inspect,math
import pytest
from supply_chain.inventory_engine import SyntheticFixture,_fixture_inputs,_episode
from supply_chain.inventory_protocol import Scenario,initial_inventory
from supply_chain.inventory_pilot import empirical_metadata
from supply_chain.inventory_initialization import opening_stock,episode_explicit,METHODS
from supply_chain.inventory_initialization_reconcile import independent_opening,reconcile_initialization

def fixture():
 f=SyntheticFixture('SYNTHETIC_D3',((2.,)*28,(1.,)*28,(3.,)*28),(3,0,7,1,8,0,0)*3)
 inputs,manifest=_fixture_inputs(f)
 b=dict(item_id=f.name,store_id='SYNTHETIC_STORE',origin=1801,forecasts=dict(zip(('MA28','M1','M3'),f.forecasts)),demand=list(f.demand),history_mean=2.25)
 return f,inputs,manifest,b

def test_hand_calculations():
 _,_,_,b=fixture()
 assert opening_stock(METHODS[0],b['forecasts'],2.25)==independent_opening(METHODS[0],b['forecasts'],2.25)==30
 assert opening_stock(METHODS[1],b['forecasts'],2.25)==independent_opening(METHODS[1],b['forecasts'],2.25)==26

@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('model',['MA28','M1','M3'])
def test_independent_episode_reconciliation(method,model):
 f,i,m,b=fixture();initial=opening_stock(method,b['forecasts'],b['history_mean']);r=empirical_metadata(episode_explicit(i,m,Scenario(),model,f.demand,initial));r['initialization_method']=method
 reconcile_initialization(r,b,asdict(Scenario()))

@pytest.mark.parametrize('mean',[0.,.1,2.25,1/13])
def test_fractional_and_zero(mean):
 f={m:(mean,)*28 for m in ['MA28','M1','M3']}
 for method in METHODS:assert opening_stock(method,f,mean)==independent_opening(method,f,mean)

@pytest.mark.parametrize('bad',[None,-1,float('nan'),True])
def test_missing_invalid_history_rejected(bad):
 _,_,_,b=fixture()
 with pytest.raises((ValueError,TypeError)):opening_stock(METHODS[0],b['forecasts'],bad)

def test_accepted_transition_body_unchanged():
 assert inspect.getsource(episode_explicit).split('initial = state.on_hand')[1].strip()==inspect.getsource(_episode).split('initial = state.on_hand')[1].strip()

def test_original_initial_state_parity_synthetic_only():
 f,i,m,b=fixture()
 for model in ['MA28','M1','M3']:assert episode_explicit(i,m,Scenario(),model,f.demand,initial_inventory(2.,Scenario()))==_episode(i,m,Scenario(),model,f.demand)

@pytest.mark.parametrize('method',METHODS)
def test_future_demand_cannot_change_current_order(method):
 f,i,m,b=fixture();initial=opening_stock(method,b['forecasts'],b['history_mean'])
 for model in ['MA28','M1','M3']:
  a=episode_explicit(i,m,Scenario(),model,f.demand,initial);z=episode_explicit(i,m,Scenario(),model,f.demand[:7]+(999,)*14,initial)
  for x,y in zip(a['daily'][:8],z['daily'][:8]):assert (x['target'],x['order_units'])==(y['target'],y['order_units'])

@pytest.mark.parametrize('field',['receipts','pipeline_end','lost','holding_cost','order_units'])
def test_corrupt_event_rejected(field):
 f,i,m,b=fixture();r=empirical_metadata(episode_explicit(i,m,Scenario(),'M1',f.demand,30));r['initialization_method']=METHODS[0];r['daily'][0][field]+=1
 with pytest.raises(ValueError):reconcile_initialization(r,b,asdict(Scenario()))

def test_scenario_and_initial_guards():
 f,i,m,b=fixture()
 with pytest.raises(ValueError):episode_explicit(i,m,Scenario(lead_days=1),'M1',f.demand,30)
 with pytest.raises(ValueError):episode_explicit(i,m,Scenario(),'M1',f.demand,-1)
 with pytest.raises(ValueError):opening_stock('invented',b['forecasts'],2.25)
