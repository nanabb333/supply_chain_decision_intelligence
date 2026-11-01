"""D3 independent initial-state and event oracle. No engine/policy calculations imported."""
import math,hashlib,json
from decimal import Decimal,localcontext
from .inventory_pilot import MODELS,KIND,equal,independent_target

def independent_opening(method,forecasts,history_mean):
    if not isinstance(history_mean,(int,float)) or isinstance(history_mean,bool) or not math.isfinite(history_mean) or history_mean<0:raise ValueError('Invalid historical mean')
    with localcontext() as ctx:
        ctx.prec=1100
        if method=='history1801_common':return math.ceil(float(Decimal(13)*Decimal.from_float(float(history_mean))))
        if method!='symmetric_forecast_common':raise ValueError('Unknown method')
        targets=sorted(independent_target(forecasts[m][:10],3,forecasts['MA28'][0]) for m in MODELS)
        return math.ceil(targets[1])

def reconcile_initialization(result,bound,scenario):
    from .inventory_protocol import Scenario
    Scenario(**scenario).validate()
    L,R,b = scenario["lead_days"],scenario["review_days"],scenario["buffer_days"]
    h,p,K = scenario["holding_per_unit_day"],scenario["shortage_per_unit"],scenario["ordering_per_order"]
    if result['model'] not in MODELS:raise ValueError('Unexpected model')
    equal(result['data_kind'],KIND,'scope label')
    for key in ('item_id','store_id','origin'):equal(result[key],bound[key],key)
    equal(result['origin'],1801,'origin')
    equal(result['scenario'],scenario,'scenario')
    equal(result['initial_pipeline'],0,'initial pipeline')
    mean=bound['forecasts']['MA28'][0];f=bound['forecasts'][result['model']];demand=bound['demand']
    if len(f)!=28 or len(demand)!=21:raise ValueError('Input horizon mismatch')
    expected_initial=independent_opening(result['initialization_method'],bound['forecasts'],bound['history_mean'])
    equal(result['initial_inventory'],expected_initial,'initial stock')
    equal(result['demand_sha256'],hashlib.sha256(json.dumps(demand,separators=(',',':')).encode()).hexdigest(),'shared demand hash')
    if len(result['daily'])!=21:raise ValueError('Incomplete episode')
    stock=expected_initial;outstanding={};placed=[];received=set();totals={k:0 for k in ('receipts','order_units','fulfilled','lost','holding_cost','shortage_cost','ordering_cost')}
    inventory_sum=0;stockout_days=0
    for t,row in enumerate(result['daily'],1):
        equal(row['day'],t,'local day');equal(row['target_day'],1801+t,'target day')
        equal(row['forecast_origin'],1801,'forecast origin');equal(row['history_max'],1800+t,'past cutoff')
        pipe_before=sum(o['units'] for o in outstanding.values())
        due=[o for o in outstanding.values() if o['due_day']==t]
        due_ids=[o['order_id'] for o in due];receipts=sum(o['units'] for o in due)
        equal(row['received_ids'],due_ids,'receipt identities')
        for order in due:
            if order['order_id'] in received:raise ValueError('Duplicate receipt')
            received.add(order['order_id']);del outstanding[order['order_id']]
        before=stock;stock+=receipts;ip=stock+sum(o['units'] for o in outstanding.values())
        target=None;horizons=[];q=0;due_day=None;oid=None
        if (t-1)%R==0:
            horizons=list(range(t,t+L+R));target=independent_target([f[j-1] for j in horizons],b,mean)
            q=max(0,math.ceil(target-ip))
            if q:
                due_day=t+L;oid=f"{result['model']}:{t}"
                order=dict(order_id=oid,placed_day=t,due_day=due_day,units=q)
                outstanding[oid]=order;placed.append(order)
        proxy=demand[t-1]
        if type(proxy)!=int or proxy<0:raise ValueError('Invalid sales units')
        fulfilled=min(stock,proxy);lost=proxy-fulfilled;stock-=fulfilled
        expected=dict(begin_inventory=before,receipts=receipts,after_receipts=before+receipts,
            pipeline_begin=pipe_before,inventory_position=ip,target=target,protection_horizons=horizons,
            order_units=q,order_id=oid,due_day=due_day,demand=proxy,fulfilled=fulfilled,lost=lost,end_inventory=stock,
            pipeline_end=sum(o['units'] for o in outstanding.values()),holding_cost=h*stock,shortage_cost=p*lost,ordering_cost=K*int(q>0))
        for key,value in expected.items():equal(row[key],value,f'day{t}.{key}')
        for key in ('begin_inventory','receipts','after_receipts','pipeline_begin','inventory_position','order_units','demand','fulfilled','lost','end_inventory','pipeline_end'):
            if type(row[key])!=int or row[key]<0:raise ValueError('Invalid integer state: '+key)
        equal(row['begin_inventory']+row['receipts']-row['fulfilled'],row['end_inventory'],'inventory conservation')
        equal(row['pipeline_begin']+row['order_units']-row['receipts'],row['pipeline_end'],'pipeline conservation')
        equal(row['fulfilled']+row['lost'],row['demand'],'demand conservation')
        for key in totals:totals[key]+=expected[key]
        inventory_sum+=stock;stockout_days+=int(lost>0)
    equal(result['orders'],placed,'complete order register')
    equal(result['terminal_orders'],list(outstanding.values()),'terminal orders')
    if any(o['due_day']<=21 for o in outstanding.values()):raise ValueError('Overdue terminal order')
    # JSON stores trace tuples as lists; accept representation only, not changed sequencing.
    equal([list(x) for x in result['access_trace']],[[t,e] for t in range(1,22) for e in ('decision_committed','demand_revealed')],'access sequence')
    total=sum(demand)
    totals.update(demand=total,fill_rate=totals['fulfilled']/total if total else None,average_inventory=inventory_sum/21,
        stockout_days=stockout_days,order_count=len(placed),total_cost=totals['holding_cost']+totals['shortage_cost']+totals['ordering_cost'],
        terminal_inventory=stock,terminal_pipeline=sum(o['units'] for o in outstanding.values()))
    for key,value in totals.items():equal(result['summary'][key],value,'summary.'+key)
    return dict(item_id=bound['item_id'],store_id=bound['store_id'],origin=1801,model=result['model'],
                initial_inventory=expected_initial,initial_pipeline=0,**totals)
