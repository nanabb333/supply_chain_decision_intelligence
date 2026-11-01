"""Exclusive, preregistered C2 pilot coordinator. Never retries. No other scope."""
from pathlib import Path
import sys,json,os,secrets,time,datetime,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supply_chain.inventory_pilot import sha,write_new,validate_scope,claim_attempt,empirical_metadata
from supply_chain.inventory_resources import supervise,Limits
OUT=ROOT/'reports/week3/checkpoint-c2-v1'


def verify_frozen():
    manifest=json.loads((OUT/'PILOT_EXECUTION_MANIFEST.json').read_text())
    if sha(OUT/'PILOT_EXECUTION_MANIFEST.json')!=(OUT/'PILOT_EXECUTION_MANIFEST.sha256').read_text().strip():raise ValueError('Manifest changed')
    for p,h in manifest['frozen_files'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen code/config/input-manifest changed: '+p)
    if json.loads((OUT/'PREFLIGHT.json').read_text())['status']!='PASS':raise ValueError('Preflight not passed')
    config=json.loads((ROOT/'configs/week3/checkpoint-c2-v1.json').read_text())
    b=json.loads((ROOT/'configs/week3/checkpoint-b-v1.json').read_text());validate_scope(config,b)
    return manifest,config


def worker(token):
    started=time.perf_counter();manifest,config=verify_frozen()
    attempt=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())
    if token!=attempt['worker_token']:raise PermissionError('Wrong one-time worker token')
    write_new(OUT/'WORKER_STARTED.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(OUT/'PILOT_EXECUTION_MANIFEST.json')))
    from supply_chain.inventory_binding import load_frozen_pilot
    from supply_chain.inventory_engine import _episode
    from supply_chain.inventory_protocol import Scenario
    protected=json.loads((OUT/'protected-manifest.json').read_text())
    series,used=load_frozen_pilot(ROOT,protected)
    if len(series)!=128:raise ValueError('Pilot cohort mismatch')
    write_new(OUT/'input-source-receipt.json',dict(origin=1801,used_sources=used,series=128,source_binding='SHA256 verified buffers decoded; accepted C1 adapter unchanged'))
    scenario=Scenario(**config['scenario']);count=0
    with (OUT/'bound-inputs.jsonl').open('x') as inf,(OUT/'episodes.jsonl').open('x') as ef,(OUT/'execution-progress.jsonl').open('x') as pf:
        for n,bound in enumerate(series,1):
            first=bound.forecasts.forecasts[0]
            saved=dict(item_id=first.item_id,store_id=first.store_id,origin=first.origin,
                forecasts={f.model:list(f.values) for f in bound.forecasts.forecasts},demand=list(bound.demand),
                forecast_sources={f.model:dict(path=f.source,sha256=f.source_sha256) for f in bound.forecasts.forecasts},
                demand_source=dict(path=bound.demand_source,sha256=bound.demand_source_sha256))
            inf.write(json.dumps(saved,allow_nan=False)+'\n')
            for model in config['models']:
                result=empirical_metadata(_episode(bound.forecasts,protected,scenario,model,bound.demand))
                ef.write(json.dumps(result,allow_nan=False)+'\n');count+=1
            inf.flush();ef.flush();os.fsync(inf.fileno());os.fsync(ef.fileno())
            pf.write(json.dumps(dict(series_completed=n,policy_episodes_completed=count))+'\n');pf.flush()
    if count!=384:raise ValueError('Coverage mismatch')
    artifacts=['bound-inputs.jsonl','episodes.jsonl','execution-progress.jsonl','input-source-receipt.json']
    write_new(OUT/'WORKER_COMPLETE.json',dict(status='PASS',series=128,policy_episodes=count,event_rows=count*21,
        worker_elapsed_seconds=time.perf_counter()-started,output_sha256={p:sha(OUT/p) for p in artifacts},
        empirical_attempts=1,forecasts_regenerated=False))


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':return worker(sys.argv[2])
    if sys.argv[1:]!=['--execute-once']:raise ValueError('Use --execute-once only after frozen preflight; no retry')
    verify_frozen()
    token=secrets.token_hex(24)
    claim_attempt(OUT,dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker_token=token,manifest_sha256=sha(OUT/'PILOT_EXECUTION_MANIFEST.json'),attempt=1))
    try:
        resources=supervise([sys.executable,str(Path(__file__).resolve()),'--worker',token],OUT/'worker.stdout',OUT/'worker.stderr',Limits())
        write_new(OUT/'RESOURCE_USAGE.json',resources)
        passed=resources['status']=='PASS' and (OUT/'WORKER_COMPLETE.json').is_file()
        write_new(OUT/'EXECUTION_STATUS.json',dict(status='PASS' if passed else 'FAILED_STOP_NO_RETRY',attempts=1,resource_status=resources['status']))
        if not passed:raise RuntimeError('Pilot failed or resource gate breached; preserve evidence, no retry')
    except BaseException as exc:
        if not (OUT/'EXECUTION_ERROR.json').exists():write_new(OUT/'EXECUTION_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),retry_allowed=False))
        raise

if __name__=='__main__':main()
