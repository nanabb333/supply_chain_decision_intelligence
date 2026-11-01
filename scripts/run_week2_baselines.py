"""Chunked B development evaluation; fresh destinations only."""
from pathlib import Path
import sys,json,time,resource,hashlib,argparse,platform
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from supply_chain.baselines import *
RAW=R/'data/raw/m5-organizers-2026-10-13/sales_train_validation.csv'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def run(out,limit=None):
 if out.exists():raise FileExistsError('Output exists; use a new version')
 out.mkdir(parents=True)
 for d in ['forecasts','targets','metrics','cohorts','horizon','audit']: (out/d).mkdir()
 start=time.perf_counter();load_seconds=compute_seconds=write_seconds=0.;records=0;log=[]
 hist=iter(load_blocks(RAW,1,max(ORIGINS),limit=limit));targets=iter(load_blocks(RAW,min(ORIGINS)+1,max(ORIGINS)+H,limit=limit))
 for chunk in range(100000):
  t=time.perf_counter()
  try:block=next(hist)
  except StopIteration:break
  load_seconds+=time.perf_counter()-t;meta=block[META].reset_index(drop=True);u=block[[f'd_{j}' for j in range(1,max(ORIGINS)+1)]].to_numpy(float)
  cache={}
  # Freeze all cohorts and forecasts before exposing target arrays for this chunk.
  for origin in ORIGINS:
   t=time.perf_counter();history=u[:,:origin].copy();ok=eligible(history,origin);cohort=meta.copy();cohort['origin']=origin;cohort['eligible']=ok;cohort['reason']=np.where(ok,'','Missing/invalid past units or short history')
   cohort.to_parquet(out/'cohorts'/f'{origin}-{chunk:03}.parquet',index=False)
   selected=meta.loc[ok].reset_index(drop=True);history=history[ok];stats=training_stats(history);pred=predict(history,origin);compute_seconds+=time.perf_counter()-t
   cache[origin]=(ok,selected,stats,pred)
   t=time.perf_counter();forecast=[]
   for model,p in pred.items():
    f=selected.copy();f['origin']=origin;f['model']=model
    for h in range(H):f[f'f_{h+1}']=p[:,h]
    forecast.append(f)
   pd.concat(forecast,ignore_index=True).to_parquet(out/'forecasts'/f'{origin}-{chunk:03}.parquet',index=False)
   if chunk==0:
    # Deterministic real-data reference subset: first16 rows, all history through each origin only.
    audit=selected.head(16).copy();audit['history']=[x.tolist() for x in history[:16]];audit.to_parquet(out/'audit'/f'input-{origin}.parquet',index=False)
   write_seconds+=time.perf_counter()-t
  t=time.perf_counter();targetblock=next(targets);load_seconds+=time.perf_counter()-t
  if not meta.equals(targetblock[META].reset_index(drop=True)):raise ValueError('History/target identity mismatch')
  for origin,(ok,selected,stats,pred) in cache.items():
   y=targetblock[[f'd_{j}' for j in range(origin+1,origin+H+1)]].to_numpy(float)[ok]
   target=selected.copy();target['origin']=origin
   for h in range(H):target[f'y_{h+1}']=y[:,h]
   target.to_parquet(out/'targets'/f'{origin}-{chunk:03}.parquet',index=False)
   metric=[];hor=[]
   for model,p in pred.items():
    m=pd.concat([selected,score(y,p,stats)],axis=1);m['model']=model;m['origin']=origin
    for k,v in stats.items():m[k]=v
    metric.append(m)
    mask=np.isfinite(y);err=np.where(mask,p-y,0)
    for h in range(H):hor.append(dict(origin=origin,model=model,horizon=h+1,n=int(mask[:,h].sum()),sum_abs=float(np.abs(err[:,h]).sum()),sum_error=float(err[:,h].sum())))
   pd.concat(metric,ignore_index=True).to_parquet(out/'metrics'/f'{origin}-{chunk:03}.parquet',index=False)
   pd.DataFrame(hor).to_parquet(out/'horizon'/f'{origin}-{chunk:03}.parquet',index=False)
   log.append(dict(origin=origin,chunk=chunk,input_rows=len(meta),eligible=int(ok.sum()),forecast_rows=len(selected)*len(pred),target_start=origin+1,target_end=origin+H,source_max=origin,methods=len(pred)))
  records+=len(meta)
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4*1024**3:raise MemoryError('4GiB peak ceiling exceeded')
  if time.perf_counter()-start>7200:raise TimeoutError('2-hour run ceiling exceeded')
  if chunk%10==0:print(f'processed {records} series',flush=True)
 try:next(targets);raise ValueError('Extra target rows')
 except StopIteration:pass
 elapsed=time.perf_counter()-start;files=[p for p in out.rglob('*') if p.is_file()];size=sum(p.stat().st_size for p in files)
 info=dict(status='PASS',series=records,origins=list(ORIGINS),horizon=H,methods=10,elapsed_seconds=elapsed,history_target_loading_seconds=load_seconds,forecast_compute_seconds=compute_seconds,forecast_write_seconds=write_seconds,other_metric_io_seconds=elapsed-load_seconds-compute_seconds-write_seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,output_bytes=size,pilot=limit is not None,final_holdout_materialized=False,source_sha256=sha(RAW),module_sha256=sha(R/'src/supply_chain/baselines.py'),runner_sha256=sha(Path(__file__)),python=sys.version,numpy=np.__version__,pandas=pd.__version__)
 (out/'execution.json').write_text(json.dumps(info,indent=2));pd.DataFrame(log).to_csv(out/'origin-manifest.csv',index=False)
 (out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2))
 print(json.dumps(info),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();run(a.output,512 if a.pilot else None)
