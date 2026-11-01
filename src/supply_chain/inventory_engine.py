"""C1 deterministic episode core. Public execution is SYNTHETIC ONLY.

No empirical runner is enabled. No policy gets a demand path or updates forecasts.
"""
from dataclasses import dataclass, replace, asdict
from hashlib import sha256
import json
from .inventory_policy import (ForecastInput, MatchedInputs, PolicyConfig, OrderEvent,
                               initialize, calculate_order, quantity, MODELS)
from .inventory_protocol import Scenario


@dataclass(frozen=True)
class SyntheticFixture:
    name: str
    forecasts: tuple[tuple[float, ...], ...]  # MA28, M1, M3
    demand: tuple[int, ...]

    def validate(self):
        if not isinstance(self.name, str) or not self.name.startswith('SYNTHETIC_'):
            raise ValueError('Explicit synthetic identity required')
        if type(self.forecasts) is not tuple or len(self.forecasts) != 3:
            raise ValueError('Three immutable synthetic vectors required')
        if type(self.demand) is not tuple or len(self.demand) != 21:
            raise ValueError('Exactly21 synthetic daily units required')
        for d in self.demand: quantity(d, 'synthetic demand', integer=True)
        return self


class DemandTape:
    """Sequential reveal barrier. This object is never passed to policy code."""
    def __init__(self, values):
        self.__values = tuple(values)
        self.__next = 1
        self.__committed = False
        self.events = []

    def commit_decision(self, day):
        if day != self.__next or self.__committed:
            raise ValueError('Decision must commit once in sequence')
        self.__committed = True
        self.events.append((day, 'decision_committed'))

    def reveal(self, day):
        if day != self.__next or not self.__committed:
            raise ValueError('Demand access before decision, out of order or repeated')
        value = self.__values[day-1]
        self.events.append((day, 'demand_revealed'))
        self.__next += 1
        self.__committed = False
        return value


def _fixture_inputs(fixture):
    fixture.validate()
    rows, manifest = [], {}
    for model, values in zip(MODELS, fixture.forecasts):
        # Shape-compatible in-memory test provenance; NEVER a claim of real bytes.
        path = ('reports/week2/checkpoint-b-v1/full-v1/forecasts/1801-000.parquet' if model == 'MA28'
                else f'reports/week2/checkpoint-c-v2/population-1801/{model}-c0-forecasts.parquet')
        digest = sha256(('SYNTHETIC_ONLY:'+repr(values)).encode()).hexdigest()
        manifest[path] = digest
        rows.append(ForecastInput(fixture.name, 'SYNTHETIC_STORE', 1801, model, values, path, digest, 1801))
    return MatchedInputs(tuple(rows)).validate(manifest), manifest


def _episode(inputs, manifest, scenario, model, demand):
    """Private deterministic core; invoked only by synthetic wrapper in C1."""
    scenario.validate()
    if scenario.episode_days != 21 or len(demand) != 21:
        raise ValueError('Independent21-day episode required')
    policy = PolicyConfig(model)
    state = initialize(inputs, policy, scenario, manifest)
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


def simulate_synthetic(fixture: SyntheticFixture, scenario: Scenario):
    if type(fixture) is not SyntheticFixture:
        raise PermissionError('C1 allows only explicit synthetic fixtures')
    inputs, manifest = _fixture_inputs(fixture)
    results = tuple(_episode(inputs, manifest, scenario, model, fixture.demand) for model in MODELS)
    for key in ('initial_inventory','initial_pipeline','demand_sha256','scenario','origin','item_id','store_id'):
        if any(r[key] != results[0][key] for r in results):
            raise ValueError(f'Cross-policy mismatch: {key}')
    return results


def run_empirical_pilot(*args, **kwargs):
    raise PermissionError('C1: empirical pilot execution requires separate user authorization; disabled')
