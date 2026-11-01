from dataclasses import replace
import json
from pathlib import Path
import pytest
from supply_chain.inventory_protocol import Scenario
from supply_chain.inventory_policy import (ForecastInput, MatchedInputs, PolicyConfig,
    InventoryState, OrderEvent, initialize, calculate_order, validate_design)

@pytest.fixture
def bundle():
    manifest = {}
    rows = []
    for model in ('MA28', 'M1', 'M3'):
        source = ('reports/week2/checkpoint-b-v1/full-v1/forecasts/1801-000.parquet' if model == 'MA28'
                  else f'reports/week2/checkpoint-c-v2/population-1801/{model}-c0-forecasts.parquet')
        manifest[source] = 'a'*64
        rows.append(ForecastInput('item', 'store', 1801, model, (2.,)*28 if model == 'MA28' else tuple(float(i) for i in range(1,29)), source, 'a'*64, 1801))
    return MatchedInputs(tuple(rows)), manifest


def test_common_initialization_and_manual_targets(bundle):
    inputs, manifest = bundle
    states = [initialize(inputs, PolicyConfig(m), Scenario(), manifest) for m in ('MA28','M1','M3')]
    assert [s.on_hand for s in states] == [26,26,26] # (3+7+3)*2, independent hand calculation
    assert all(s.pipeline == () and s.backlog == 0 for s in states)
    ma = calculate_order(inputs, PolicyConfig('MA28'), states[0], manifest)
    m1 = calculate_order(inputs, PolicyConfig('M1'), states[1], manifest)
    assert (ma.target, ma.order_units, ma.due_day) == (26,0,None)
    assert (m1.target, m1.order_units, m1.due_day) == (61,35,4) # 1+...+10+6 = 61
    assert states[1].on_hand == 26 and states[1].pipeline == () # calculation cannot mutate


def test_pipeline_nonreview_and_ceil(bundle):
    inputs, manifest = bundle
    s = initialize(inputs, PolicyConfig('M1'), Scenario(lead_days=7), manifest)
    s = replace(s, day=8, observed_through=1808, on_hand=10,
                pipeline=()) # order from day1 already received by day8
    result = calculate_order(inputs, PolicyConfig('M1'), s, manifest)
    assert result.target == sum(range(8,22))+6 and result.due_day == 15
    # Daily review: day1 order is pending until day4 at the day2 snapshot.
    s = replace(s, scenario=Scenario(review_days=1), day=2, observed_through=1802,
                pipeline=(OrderEvent('o1',1,4,4),))
    result = calculate_order(inputs, PolicyConfig('M1'), s, manifest)
    assert result.inventory_position == 14 and result.target == 20 and result.order_units == 6
    nonreview = replace(s, scenario=Scenario(), pipeline=())
    assert calculate_order(inputs, PolicyConfig('M1'), nonreview, manifest).order_units == 0
    fs = tuple(replace(f, values=(0.21,)*28) if f.model=='M1' else f for f in inputs.forecasts)
    result = calculate_order(MatchedInputs(fs), PolicyConfig('M1'), replace(nonreview,day=1,observed_through=1801,on_hand=0), manifest)
    assert result.target == pytest.approx(8.1) and result.order_units == 9


@pytest.mark.parametrize('change',[
    {'origin':1885}, {'history_max_day':1802}, {'issue_convention':'actual'},
    {'values':(0.,)*27}, {'values':(float('nan'),)*28}, {'values':(-1.,)*28},
    {'values':(float('inf'),)*28}, {'values':(True,)*28},
    {'source_sha256':'b'*64}, {'source':'../wrong'}, {'model':'M2'}, {'item_id':''}
])
def test_forecast_rejects_invalid(bundle,change):
    inputs,m = bundle
    with pytest.raises(ValueError): replace(inputs.forecasts[1],**change).validate(m)


def test_ma28_and_match_guards(bundle):
    inputs,m = bundle
    with pytest.raises(ValueError): replace(inputs.forecasts[0],values=tuple(range(28))).validate(m)
    for forecasts in (inputs.forecasts[:2], (inputs.forecasts[0],)*3,
                      (replace(inputs.forecasts[0],store_id='other'),)+inputs.forecasts[1:]):
        with pytest.raises(ValueError): MatchedInputs(forecasts).validate(m)


@pytest.mark.parametrize('change',[
    {'on_hand':-1},{'on_hand':1.5},{'backlog':1},{'day':22},
    {'observed_through':1802},{'observed_through':1800},{'policy_model':'M2'},
    {'pipeline':(OrderEvent('x',1,4,1),)}, # not placed yet at day1
])
def test_invalid_state(bundle,change):
    inputs,m = bundle
    s=initialize(inputs,PolicyConfig('M1'),Scenario(),m)
    with pytest.raises(ValueError): calculate_order(inputs,PolicyConfig('M1'),replace(s,**change),m)


def test_orders_and_identity(bundle):
    inputs,m = bundle
    s=initialize(inputs,PolicyConfig('M1'),Scenario(review_days=1),m)
    for event in (OrderEvent('x',1,1,1),OrderEvent('x',1,4,0),OrderEvent('x',1,5,2)):
        with pytest.raises(ValueError): event.validate(s.scenario)
    for pipeline in ((OrderEvent('x',1,4,2),)*2,(OrderEvent('x',1,4,2),)):
        with pytest.raises(ValueError): replace(s,day=4,observed_through=1804,pipeline=pipeline).validate()
    with pytest.raises(ValueError): calculate_order(inputs,PolicyConfig('M3'),s,m)
    with pytest.raises(ValueError): calculate_order(inputs,PolicyConfig('M1'),replace(s,item_id='other'),m)


def test_zero_history_and_all_scenario_boundaries(bundle):
    inputs,m=bundle
    z=MatchedInputs(tuple(replace(f,values=(0.,)*28) for f in inputs.forecasts))
    a=json.loads(Path('configs/week3/checkpoint-a-v1.json').read_text())
    for record in a['scenarios']:
        sc=Scenario(**record)
        state=initialize(z,PolicyConfig('MA28'),sc,m)
        assert state.on_hand==0
        last=1+((20)//sc.review_days)*sc.review_days
        result=calculate_order(z,PolicyConfig('MA28'),replace(state,day=last,observed_through=1801+last-1),m)
        assert result.order_units==0 and max(result.protection_horizons)<=28


@pytest.mark.parametrize('policy',[PolicyConfig('M2'),PolicyConfig('M1',forecast_updates=True),PolicyConfig('M1',rounding='floor')])
def test_policy_restrictions(policy):
    with pytest.raises(ValueError): policy.validate()


def test_saved_design_and_drift():
    a=json.loads(Path('configs/week3/checkpoint-a-v1.json').read_text())
    b=json.loads(Path('configs/week3/checkpoint-b-v1.json').read_text())
    assert validate_design(b,a)
    for change in ({'simulation_enabled':True},{'pilot_execution_authorized':True},
                   {'development_origins':[1885]}, {'scenarios':[]}, {'episode_days':28},
                   {'policies':[]},{'pilot':{}},{'assumption_status':'implicit'}):
        with pytest.raises(ValueError): validate_design(dict(b,**change),a)
