"""One bounded D1 sensitivity batch; reuses C2 primary and never retries."""
from pathlib import Path
import sys,json,os,secrets,time,datetime,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supply_chain.inventory_pilot import sha,write_new,claim_attempt,empirical_metadata
from supply_chain.inventory_sensitivity import validate_scope_d1
from supply_chain.inventory_resources import supervise,Limits
OUT=ROOT/'reports/week3/checkpoint-d1-v1'


def verify_frozen():
    manifest=json.loads((OUT/'EXECUTION_MANIFEST.json').read_text())
    if sha(OUT/'EXECUTION_MANIFEST.json')!=(OUT/'EXECUTION_MANIFEST.sha256').read_text().strip():raise ValueError('Manifest drift')
    for p,h in manifest['frozen_files'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen file changed: '+p)
    if json.loads((OUT/'PREFLIGHT.json').read_text())['status']!='PASS':raise ValueError('Preflight not passed')
    config=json.loads((ROOT/'configs/week3/checkpoint-d1-v1.json').read_text())
    a=json.loads((ROOT/'configs/week3/checkpoint-a-v1.json').read_text());b=json.loads((ROOT/'configs/week3/checkpoint-b-v1.json').read_text())
    validate_scope_d1(config,a,b);return config


def worker(token):
    started=time.perf_counter();config=verify_frozen()
    if token!=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())['worker_token']:raise PermissionError('Wrong token')
    write_new(OUT/'WORKER_STARTED.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    from supply_chain.inventory_binding import load_frozen_pilot
    from supply_chain.inventory_engine import _episode
    from supply_chain.inventory_protocol import Scenario
    protected=json.loads((OUT/'protected-manifest.json').read_text());series,used=load_frozen_pilot(ROOT,protected)
    if len(series)!=128:raise ValueError('Cohort mismatch')
    # Verify new input binding equals C2's accepted vectors before any new scenario.
    c2_inputs=[json.loads(x) for x in (ROOT/'reports/week3/checkpoint-c2-v1/bound-inputs.jsonl').read_text().splitlines()]
    for source,saved in zip(series,c2_inputs):
        first=source.forecasts.forecasts[0]
        if (first.item_id,first.store_id,first.origin)!=(saved['item_id'],saved['store_id'],saved['origin']) or list(source.demand)!=saved['demand'] or {f.model:list(f.values) for f in source.forecasts.forecasts}!=saved['forecasts']:raise ValueError('Input changes relative to C2')
    write_new(OUT/'input-source-receipt.json',dict(used_sources=used,source_files=len(used),series=128,matched_C2_inputs=True))
    count=0;output_hashes={};records={x['name']:x for x in config['scenarios']}
    with (OUT/'execution-progress.jsonl').open('x') as progress:
        for name in config['execute_scenarios']:
            sc=Scenario(**records[name]);path=OUT/f'episodes-{name}.jsonl'
            with path.open('x') as f:
                for n,bound in enumerate(series,1):
                    for model in config['models']:
                        result=empirical_metadata(_episode(bound.forecasts,protected,sc,model,bound.demand))
                        f.write(json.dumps(result,allow_nan=False)+'\n');count+=1
                    f.flush();os.fsync(f.fileno())
                    progress.write(json.dumps(dict(scenario=name,series_completed=n,new_policy_episodes_completed=count))+'\n');progress.flush()
            output_hashes[path.name]=sha(path)
    if count!=3072:raise ValueError('New coverage mismatch')
    output_hashes['input-source-receipt.json']=sha(OUT/'input-source-receipt.json')
    output_hashes['execution-progress.jsonl']=sha(OUT/'execution-progress.jsonl')
    write_new(OUT/'WORKER_COMPLETE.json',dict(status='PASS',scenarios_executed=config['execute_scenarios'],new_episodes=count,new_daily_records=count*21,primary_reused_not_rerun=True,worker_elapsed_seconds=time.perf_counter()-started,output_sha256=output_hashes))


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':return worker(sys.argv[2])
    if sys.argv[1:]!=['--execute-once']:raise ValueError('Only --execute-once supported')
    verify_frozen()
    if list(OUT.glob('episodes-*.jsonl')):raise FileExistsError('Partial scenario output exists; no retry')
    token=secrets.token_hex(24);claim_attempt(OUT,dict(attempt=1,worker_token=token,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(OUT/'EXECUTION_MANIFEST.json')))
    try:
        usage=supervise([sys.executable,str(Path(__file__).resolve()),'--worker',token],OUT/'worker.stdout',OUT/'worker.stderr',Limits())
        write_new(OUT/'RESOURCE_USAGE.json',usage)
        passed=usage['status']=='PASS' and (OUT/'WORKER_COMPLETE.json').is_file()
        write_new(OUT/'EXECUTION_STATUS.json',dict(status='PASS' if passed else 'FAILED_STOP_NO_RETRY',attempts=1))
        if not passed:raise RuntimeError('D1 execution/resource failure; no retry')
    except BaseException as exc:
        write_new(OUT/'EXECUTION_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),retry_allowed=False));raise

if __name__=='__main__':main()
