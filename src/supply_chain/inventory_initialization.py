"""D3 isolated initialization interface. Accepted C1 transitions preserved verbatim."""
from dataclasses import replace,asdict
from hashlib import sha256
import json,math,statistics
from .inventory_engine import DemandTape
from .inventory_policy import PolicyConfig,OrderEvent,initialize,calculate_order,quantity
from .inventory_protocol import Scenario
METHODS=('history1801_common','symmetric_forecast_common')

def opening_stock(method,forecasts,history_mean):
    if method not in METHODS:raise ValueError('Unknown initialization method')
    quantity(history_mean,'historical mean')
    if set(forecasts)!=set(('MA28','M1','M3')):raise ValueError('Matched three forecasts required')
    for values in forecasts.values():
        if len(values)!=28:raise ValueError('28horizons required')
        for value in values:quantity(value,'forecast')
    if len(set(forecasts['MA28']))!=1:raise ValueError('MA28 must be constant')
    if method=='history1801_common':return math.ceil(13*history_mean)
    targets=[math.fsum(forecasts[m][:10])+3*forecasts['MA28'][0] for m in ('MA28','M1','M3')]
    return math.ceil(statistics.median(targets))

def episode_explicit(inputs, manifest, scenario, model, demand, initial_units):
    """D3 versioned core: accepted transitions, explicit common opening stock."""
    if scenario != Scenario():raise ValueError("D3 primary scenario only")
    quantity(initial_units,"explicit initial stock",integer=True)
    scenario.validate()
    if scenario.episode_days != 21 or len(demand) != 21:
        raise ValueError('Independent21-day episode required')
    policy = PolicyConfig(model)
    state = initialize(inputs, policy, scenario, manifest)
    state = replace(state, on_hand=initial_units).validate()
    initial = state.on_hand
    on_hand, pipeline, all_orders, daily = initial, (), [], []
    tape = DemandTape(demand)
    for day in range(1, 22):
        begin = on_hand
        pipeline_begin = sum(e.units for e in pipeline)
        received = tuple(e for e in pipeline if e.due_day == day)
        if any(e.due_day < day for e in pipeline):
            raise ValueError('Overdue order')
        receipts = sum(e.units for e in received)
        pipeline = tuple(e for e in pipeline if e.due_day > day)
        on_hand += receipts
        state = replace(state, day=day, on_hand=on_hand, pipeline=pipeline,
                        observed_through=state.origin+day-1)
        decision = calculate_order(inputs, policy, state, manifest)
        new_order = None
        if decision.order_units:
            new_order = OrderEvent(f'{model}:{day}', day, decision.due_day, decision.order_units).validate(scenario)
            pipeline += (new_order,)
            all_orders.append(new_order)
        tape.commit_decision(day)  # no current demand access before calculation/queueing
        proxy = tape.reveal(day)
        quantity(proxy, 'demand', integer=True)
        fulfilled = min(on_hand, proxy)
        lost = proxy - fulfilled
        on_hand -= fulfilled
        pipeline_end = sum(e.units for e in pipeline)
        if begin + receipts - fulfilled != on_hand or pipeline_begin + decision.order_units - receipts != pipeline_end or fulfilled + lost != proxy:
            raise ValueError('Event conservation failed')
        daily.append(dict(day=day, target_day=state.origin+day, history_max=state.origin+day-1,
                          forecast_origin=state.origin, begin_inventory=begin, receipts=receipts,
                          received_ids=[e.order_id for e in received], after_receipts=state.on_hand,
                          pipeline_begin=pipeline_begin, inventory_position=decision.inventory_position,
                          target=decision.target, protection_horizons=list(decision.protection_horizons),
                          order_units=decision.order_units, order_id=new_order.order_id if new_order else None,
                          due_day=decision.due_day, demand=proxy, fulfilled=fulfilled, lost=lost,
                          end_inventory=on_hand, pipeline_end=pipeline_end,
                          holding_cost=scenario.holding_per_unit_day*on_hand,
                          shortage_cost=scenario.shortage_per_unit*lost,
                          ordering_cost=scenario.ordering_per_order*int(new_order is not None)))
    total_demand = sum(r['demand'] for r in daily)
    totals = {k:sum(r[k] for r in daily) for k in ('receipts','order_units','fulfilled','lost','holding_cost','shortage_cost','ordering_cost')}
    totals.update(demand=total_demand, fill_rate=totals['fulfilled']/total_demand if total_demand else None,
                  average_inventory=sum(r['end_inventory'] for r in daily)/21,
                  stockout_days=sum(r['lost']>0 for r in daily), order_count=len(all_orders),
                  total_cost=sum(totals[k] for k in ('holding_cost','shortage_cost','ordering_cost')),
                  terminal_inventory=on_hand, terminal_pipeline=sum(e.units for e in pipeline))
    return dict(model=model, origin=state.origin, item_id=state.item_id, store_id=state.store_id,
                scenario=asdict(scenario), initial_inventory=initial, initial_pipeline=0,
                demand_sha256=sha256(json.dumps(list(demand),separators=(',',':')).encode()).hexdigest(),
                daily=daily, orders=[asdict(e) for e in all_orders], terminal_orders=[asdict(e) for e in pipeline],
                access_trace=tape.events, summary=totals, data_kind='SYNTHETIC_ONLY')

