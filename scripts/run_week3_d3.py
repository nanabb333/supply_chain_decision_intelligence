"""Single authorized D3 batch; two initialization bases; no primary replay."""
from pathlib import Path
import sys,json,os,secrets,time,datetime,traceback,io
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supply_chain.inventory_pilot import sha,write_new,claim_attempt,empirical_metadata
from supply_chain.inventory_resources import supervise,Limits
from supply_chain.inventory_binding import load_frozen_pilot,verified_bytes
from supply_chain.inventory_initialization import opening_stock,episode_explicit,METHODS
from supply_chain.inventory_initialization_reconcile import independent_opening
from supply_chain.inventory_protocol import Scenario
OUT=ROOT/'reports/week3/checkpoint-d3-v1'
HISTORY='reports/week2/checkpoint-c-v2/population-1801/evaluation-cohort.parquet'

def bind():
 protected=json.loads((OUT/'protected-manifest.json').read_text());series,used=load_frozen_pilot(ROOT,protected)
 h=pd.read_parquet(io.BytesIO(verified_bytes(ROOT,HISTORY,protected)))
 if len(h)!=30490 or not h.origin.eq(1801).all() or h.duplicated(['item_id','store_id']).any():raise ValueError('Historical coverage')
 lookup=h.set_index(['item_id','store_id']).train_mean.to_dict();records=[]
 prior=[json.loads(x) for x in (ROOT/'reports/week3/checkpoint-c2-v1/bound-inputs.jsonl').read_text().splitlines()]
 if len(prior)!=128 or len(series)!=128:raise ValueError('Cohort size')
 for b,old in zip(series,prior):
  f=b.forecasts.forecasts[0];key=(f.item_id,f.store_id)
  record=dict(item_id=f.item_id,store_id=f.store_id,origin=f.origin,forecasts={x.model:list(x.values) for x in b.forecasts.forecasts},demand=list(b.demand),history_mean=lookup[key])
  if any(record[k]!=old[k] for k in ['item_id','store_id','origin','forecasts','demand']):raise ValueError('C2 input mismatch')
  records.append(record)
 return protected,series,records,used

def verify():
 m=json.loads((OUT/'EXECUTION_MANIFEST.json').read_text())
 if sha(OUT/'EXECUTION_MANIFEST.json')!=(OUT/'EXECUTION_MANIFEST.sha256').read_text().strip():raise ValueError('Manifest changed')
 for p,h in m['frozen_files'].items():
  if sha(ROOT/p)!=h:raise ValueError('Frozen file changed:'+p)
 if json.loads((OUT/'PREFLIGHT.json').read_text())['status']!='PASS':raise ValueError('Failed preflight')
 if m['methods']!=list(METHODS) or m['new_episodes']!=768 or m['new_daily_records']!=16128:raise ValueError('Scope drift')
 return m

def worker(token):
 start=time.perf_counter();verify()
 if token!=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())['worker_token']:raise PermissionError('Token')
 write_new(OUT/'WORKER_STARTED.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 protected,series,records,used=bind()
 saved=[json.loads(x) for x in (OUT/'bound-inputs.jsonl').read_text().splitlines()]
 if records!=saved:raise ValueError('Preflight binding changed')
 count=0
 with (OUT/'episodes.jsonl').open('x') as output:
  for method in METHODS:
   for b,record in zip(series,records):
    initial=opening_stock(method,record['forecasts'],record['history_mean'])
    if initial!=independent_opening(method,record['forecasts'],record['history_mean']):raise ValueError('Opening oracle mismatch')
    for model in ['MA28','M1','M3']:
     result=empirical_metadata(episode_explicit(b.forecasts,protected,Scenario(),model,b.demand,initial));result['initialization_method']=method
     output.write(json.dumps(result,allow_nan=False)+'\n');count+=1
    output.flush();os.fsync(output.fileno())
 if count!=768:raise ValueError('Incomplete execution')
 write_new(OUT/'WORKER_COMPLETE.json',dict(status='PASS',episodes=count,daily_records=count*21,methods=list(METHODS),elapsed_seconds=time.perf_counter()-start,output_sha256={'episodes.jsonl':sha(OUT/'episodes.jsonl')},primary_reused_not_rerun=True))

def main():
 if len(sys.argv)==3 and sys.argv[1]=='--worker':return worker(sys.argv[2])
 if sys.argv[1:]!=['--execute-once']:raise ValueError('Only explicit execution supported')
 verify()
 # Bound inputs were frozen in preflight; use exclusive independent attempt marker.
 if any((OUT/x).exists() for x in ['ATTEMPT_STARTED.json','WORKER_STARTED.json','WORKER_COMPLETE.json','episodes.jsonl','RESOURCE_USAGE.json']):raise FileExistsError('Prior attempt;do not retry')
 token=secrets.token_hex(24);write_new(OUT/'ATTEMPT_STARTED.json',dict(attempt=1,worker_token=token,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(OUT/'EXECUTION_MANIFEST.json')))
 usage=supervise([sys.executable,str(Path(__file__).resolve()),'--worker',token],OUT/'worker.stdout',OUT/'worker.stderr',Limits());write_new(OUT/'RESOURCE_USAGE.json',usage)
 passed=usage['status']=='PASS' and (OUT/'WORKER_COMPLETE.json').exists()
 write_new(OUT/'EXECUTION_STATUS.json',dict(status='PASS' if passed else 'FAILED_STOP_NO_RETRY',attempts=1))
 if not passed:raise RuntimeError('Execution failed;preserve evidence,no retry')
if __name__=='__main__':main()
