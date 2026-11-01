"""Independent Decimal oracle; imports NO engine/policy calculation helpers."""
from decimal import Decimal, ROUND_CEILING
import math


def independent_episode(forecast, mean28, demand, scenario):
    """Reconstruct a complete synthetic episode using due-day buckets and Decimal."""
    d=lambda x:Decimal(str(x))
    L,R,b=scenario.lead_days,scenario.review_days,d(scenario.buffer_days)
    opening=int((d(scenario.initial_multiplier)*(d(L+R)+b)*d(mean28)).to_integral_value(rounding=ROUND_CEILING))
    stock=opening; due={}; records=[]; orders=[]
    for t in range(1,22):
        begin=stock; pipeline0=sum(due.values()); receipts=due.pop(t,0);stock+=receipts
        ip=stock+sum(due.values())
        target=None; order=0; horizons=[]
        if (t-1)%R==0:
            horizons=list(range(t,t+L+R))
            target=sum((d(forecast[j-1]) for j in horizons),Decimal(0))+b*d(mean28)
            order=max(0,int((target-d(ip)).to_integral_value(rounding=ROUND_CEILING)))
            if order: due[t+L]=due.get(t+L,0)+order;orders.append((t,t+L,order))
        used=min(stock,demand[t-1]); lost=demand[t-1]-used;stock-=used
        records.append(dict(day=t,begin_inventory=begin,receipts=receipts,after_receipts=begin+receipts,
            pipeline_begin=pipeline0,inventory_position=ip,target=float(target) if target is not None else None,
            protection_horizons=horizons,order_units=order,due_day=t+L if order else None,
            demand=demand[t-1],fulfilled=used,lost=lost,end_inventory=stock,pipeline_end=sum(due.values()),
            holding_cost=float(d(scenario.holding_per_unit_day)*stock),shortage_cost=float(d(scenario.shortage_per_unit)*lost),
            ordering_cost=float(d(scenario.ordering_per_order)*int(order>0))))
    totals={k:sum(r[k] for r in records) for k in ('receipts','order_units','fulfilled','lost','holding_cost','shortage_cost','ordering_cost')}
    total=sum(demand)
    totals.update(demand=total,fill_rate=totals['fulfilled']/total if total else None,
        average_inventory=sum(r['end_inventory'] for r in records)/21,stockout_days=sum(r['lost']>0 for r in records),
        order_count=len(orders),total_cost=sum(totals[k] for k in ('holding_cost','shortage_cost','ordering_cost')),
        terminal_inventory=stock,terminal_pipeline=sum(due.values()))
    return dict(initial_inventory=opening,daily=records,summary=totals,orders=orders)


def same(actual,expected,label):
    if isinstance(expected,float):
        ok=isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-9)
    else: ok=actual==expected
    if not ok: raise ValueError(f'Independent mismatch {label}: {actual!r} != {expected!r}')


def reconcile(result, fixture, scenario):
    idx=('MA28','M1','M3').index(result['model'])
    expected=independent_episode(fixture.forecasts[idx],fixture.forecasts[0][0],fixture.demand,scenario)
    same(result['data_kind'],'SYNTHETIC_ONLY','kind')
    same(result['item_id'],fixture.name,'item');same(result['store_id'],'SYNTHETIC_STORE','store')
    same(result['origin'],1801,'origin');same(result['initial_pipeline'],0,'initial pipeline')
    same(result['initial_inventory'],expected['initial_inventory'],'initial stock')
    for field in scenario.__dataclass_fields__: same(result['scenario'][field],getattr(scenario,field),'scenario.'+field)
    if len(result['daily'])!=21: raise ValueError('Incomplete daily ledger')
    previous_stock=result['initial_inventory'];previous_pipeline=0; seen_orders={};received=set()
    for actual,want in zip(result['daily'],expected['daily']):
        t=want['day']
        for k,v in want.items():same(actual[k],v,f'day{t}.{k}')
        same(actual['begin_inventory'],previous_stock,'stock continuity')
        same(actual['pipeline_begin'],previous_pipeline,'pipeline continuity')
        same(actual['begin_inventory']+actual['receipts']-actual['fulfilled'],actual['end_inventory'],'stock balance')
        same(actual['pipeline_begin']+actual['order_units']-actual['receipts'],actual['pipeline_end'],'pipeline balance')
        same(actual['demand'],actual['fulfilled']+actual['lost'],'demand balance')
        same(actual['target_day'],1801+t,'target day');same(actual['history_max'],1800+t,'past cutoff');same(actual['forecast_origin'],1801,'fixed origin')
        for k in ('begin_inventory','receipts','after_receipts','pipeline_begin','inventory_position','order_units','demand','fulfilled','lost','end_inventory','pipeline_end'):
            if type(actual[k]) is not int or actual[k]<0:raise ValueError('Noninteger/negative state')
        due_ids=[oid for oid,e in seen_orders.items() if e['due_day']==t]
        same(actual['received_ids'],due_ids,'received ID schedule')
        for oid in actual['received_ids']:
            if oid in received: raise ValueError('Order received twice')
            received.add(oid)
        expected_id=f"{result['model']}:{t}" if actual['order_units'] else None
        same(actual['order_id'],expected_id,'order ID')
        if expected_id:
            if expected_id in seen_orders:raise ValueError('Duplicate order')
            seen_orders[expected_id]=dict(order_id=expected_id,placed_day=t,due_day=actual['due_day'],units=actual['order_units'])
        previous_stock,previous_pipeline=actual['end_inventory'],actual['pipeline_end']
    same(result['orders'],list(seen_orders.values()),'order ledger')
    pending=[e for oid,e in seen_orders.items() if oid not in received]
    same(result['terminal_orders'],pending,'terminal pipeline ledger')
    if any(e['due_day']<=21 for e in pending):raise ValueError('Missed receipt')
    same(result['access_trace'],[(t,event) for t in range(1,22) for event in ('decision_committed','demand_revealed')],'information access sequence')
    for k,v in expected['summary'].items():same(result['summary'][k],v,'summary.'+k)
    return dict(status='PASS',days=21,model=result['model'],method='Independent Decimal/due-day buckets + separate ledger identities',data_kind='SYNTHETIC_ONLY')
