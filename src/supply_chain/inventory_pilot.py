"""C2 frozen-scope adapter and independent event reconciliation. No automatic run."""
from pathlib import Path
from decimal import Decimal, localcontext
import hashlib,json,math

MODELS=('MA28','M1','M3')
PRIMARY=dict(name='primary',lead_days=3,review_days=7,episode_days=21,initial_multiplier=1.0,
             initial_pipeline_units=0.0,buffer_days=3.0,holding_per_unit_day=1.0,
             shortage_per_unit=10.0,ordering_per_order=0.0,logic='lost_sales',cost_unit='hypothetical cost units')
KIND='EMPIRICAL_M5_SALES_PROXY_HYPOTHETICAL_INVENTORY'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def validate_scope(config,b_config):
    expected=dict(origin=1801,series=128,episode_days=21,forecast_horizon=28,models=list(MODELS),scenario=PRIMARY,
                  max_attempts=1,runtime_seconds=60,memory_gate_bytes=1024**3,hard_stop_bytes=2*1024**3,
                  demand_role='recorded_sales_proxy',rolling_forecasts=False,population_enabled=False)
    if config!=expected:raise ValueError('C2 scope/config drift')
    if b_config['pilot']!=dict(origin=1801,series=128,episode_days=21,scenarios=['primary'],selection='sha256(item_id|store_id), digest/item/store ascending'):
        raise ValueError('Frozen B pilot changed')
    if b_config['scenarios'][0]!=PRIMARY or b_config['assumption_status']!='ACCEPTED_FOR_DESIGN_ONLY':raise ValueError('Primary assumption drift')
    return True


def claim_attempt(out,record):
    out=Path(out)
    for name in ('ATTEMPT_STARTED.json','WORKER_STARTED.json','WORKER_COMPLETE.json','episodes.jsonl','bound-inputs.jsonl','RESOURCE_USAGE.json'):
        if (out/name).exists():raise FileExistsError('Existing attempt/output; no retry: '+name)
    write_new(out/'ATTEMPT_STARTED.json',record)  # exclusive create; simultaneous callers cannot both claim


def empirical_metadata(result):
    """Correct the legacy core's hardcoded synthetic label; no numeric modification."""
    return dict(result,data_kind=KIND)


def equal(actual,expected,label):
    if isinstance(expected,float):
        ok=isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-9)
    else:ok=actual==expected
    if not ok:raise ValueError(f'{label}: {actual!r} != {expected!r}')


def independent_target(values,buffer_days,mean28):
    # Independent exact-binary accumulation, then the accepted IEEE float operation boundaries.
    # Avoid Decimal(str(x)) changing a ceil decision near an integer. This convention is frozen
    # BEFORE empirical execution. No engine/policy calculation helper is imported here.
    with localcontext() as ctx:
        ctx.prec=1100
        summed=float(sum((Decimal.from_float(float(v)) for v in values),Decimal(0)))
    return summed+buffer_days*mean28


def reconcile_episode(result,bound):
    if result['model'] not in MODELS:raise ValueError('Unexpected model')
    equal(result['data_kind'],KIND,'scope label')
    for key in ('item_id','store_id','origin'):equal(result[key],bound[key],key)
    equal(result['origin'],1801,'origin')
    equal(result['scenario'],PRIMARY,'scenario')
    equal(result['initial_pipeline'],0,'initial pipeline')
    mean=bound['forecasts']['MA28'][0];f=bound['forecasts'][result['model']];demand=bound['demand']
    if len(f)!=28 or len(demand)!=21:raise ValueError('Input horizon mismatch')
    expected_initial=math.ceil(1.0*(3+7+3.0)*mean)
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
        if t in (1,8,15):
            horizons=list(range(t,t+10));target=independent_target([f[j-1] for j in horizons],3.0,mean)
            q=max(0,math.ceil(target-ip))
            if q:
                due_day=t+3;oid=f"{result['model']}:{t}"
                order=dict(order_id=oid,placed_day=t,due_day=due_day,units=q)
                outstanding[oid]=order;placed.append(order)
        proxy=demand[t-1]
        if type(proxy)!=int or proxy<0:raise ValueError('Invalid sales units')
        fulfilled=min(stock,proxy);lost=proxy-fulfilled;stock-=fulfilled
        expected=dict(begin_inventory=before,receipts=receipts,after_receipts=before+receipts,
            pipeline_begin=pipe_before,inventory_position=ip,target=target,protection_horizons=horizons,
            order_units=q,order_id=oid,due_day=due_day,demand=proxy,fulfilled=fulfilled,lost=lost,end_inventory=stock,
            pipeline_end=sum(o['units'] for o in outstanding.values()),holding_cost=1.0*stock,shortage_cost=10.0*lost,ordering_cost=0.0)
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
