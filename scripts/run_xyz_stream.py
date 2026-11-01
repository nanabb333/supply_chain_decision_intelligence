"""Atomic, memory-guarded, origin-by-origin jobs; no final-holdout interface."""
from pathlib import Path
import os,sys,time,json,hashlib,resource,threading,argparse
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
import numpy as np
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from supply_chain.xyz_stream import *
from supply_chain.xyz_features import select_pilot
from supply_chain.baselines import load_blocks,META,training_stats,score
C=R/'reports/week2/checkpoint-c-v2';B=R/'reports/week2/checkpoint-b-v1/full-v1';RAW=R/'data/raw/m5-organizers-2026-10-13/sales_train_validation.csv'
LIMIT=3*1024**3

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()

def atomic_json(p,v):
 if p.exists():raise FileExistsError(p)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)

def fingerprint():return {str(p.relative_to(R)):sha(p) for p in [R/'src/supply_chain/xyz_stream.py',R/'src/supply_chain/xyz_features.py',R/'src/supply_chain/baselines.py',Path(__file__),C/'COMPUTE_AMENDMENT.md']}

def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)

def monitor(out,stop):
 while not stop.wait(.05):
  if rss()>=LIMIT:
   (out/'MEMORY_STOP.json').write_text(json.dumps({'peak_rss_bytes':rss(),'threshold':LIMIT}));os._exit(90)

def selected_blocks(keys,origin,chunk):
 for raw in load_blocks(RAW,1,origin,chunk=chunk):
  mask=[(i,s) in keys for i,s in zip(raw.item_id,raw.store_id)]
  if any(mask):
   b=raw.loc[mask].reset_index(drop=True);yield b[META],b[[f'd_{j}' for j in range(1,origin+1)]].to_numpy(float)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--origin',type=int,required=True);ap.add_argument('--series',type=int,required=True);ap.add_argument('--name',required=True);ap.add_argument('--chunk',type=int,default=512);a=ap.parse_args();allowed_origin(a.origin)
 if a.series not in (256,2048,30490) or not 1<=a.chunk<=512 or Path(a.name).name!=a.name:raise ValueError('Unapproved bounds/job name')
 if a.series==30490:
  gate=json.loads((C/'resource-gate.json').read_text());assert gate['status']=='PASS' and gate['source_fingerprints']==fingerprint()
  for p,h in gate['pilot_receipts'].items():assert sha(R/p)==h
 out=C/a.name;out.mkdir(exist_ok=True)
 if (out/'execution.json').exists():raise FileExistsError('Completed job preserved; use its receipt, do not rerun')
 fp=fingerprint();config=dict(origin=a.origin,series=a.series,chunk=a.chunk,source_fingerprints=fp)
 if (out/'job.json').exists():assert json.loads((out/'job.json').read_text())==config
 else:atomic_json(out/'job.json',config)
 stop=threading.Event();threading.Thread(target=monitor,args=(out,stop),daemon=True).start();start=time.perf_counter()
 con=duckdb.connect();con.execute("SET threads=1");con.execute("SET memory_limit='256MB'")
 cohort=con.sql(f"SELECT * FROM read_parquet('{B}/cohorts/{a.origin}-*.parquet') ORDER BY item_id,store_id").df()
 assert len(cohort)==30490 and cohort.eligible.all() and not cohort.duplicated(['item_id','store_id']).any()
 selected=select_pilot(cohort,a.series);keys=set(zip(selected.item_id,selected.store_id));assert len(keys)==a.series
 if not (out/'selection.csv').exists():selected.to_csv(out/'selection.csv',index=False)
 else:assert set(zip(pd.read_csv(out/'selection.csv').item_id,pd.read_csv(out/'selection.csv').store_id))==keys
 with threadpool_limits(limits=1):
  if (out/'training.json').exists():
   train=json.loads((out/'training.json').read_text());assert sha(out/'statistics.npz')==train['statistics_sha256'];g=dict(np.load(out/'statistics.npz'));assert train['series']==a.series
  else:
   g=empty_stats();seen=set();coverage=[];yh=hashlib.sha256();train_start=time.perf_counter()
   for j,(meta,u) in enumerate(selected_blocks(keys,a.origin,a.chunk)):
    pairs=list(zip(meta.item_id,meta.store_id));assert not seen.intersection(pairs);seen.update(pairs)
    ts=validate_origin_history(u,a.origin);static=static_features(meta);before=int(g['n']);target_sum=0.
    for t in ts:
     s,c=factors(u[:,:t],meta,t,static);y=u[:,t:t+28];add_block(g,s,c,y);yh.update(y.astype('<f8').tobytes());target_sum+=float(y.sum())
    coverage.append(dict(block=j,series=len(u),snapshots=len(ts),rows=int(g['n'])-before,first_snapshot=min(ts),last_snapshot=max(ts),last_training_target=max(ts)+28,target_sum=target_sum))
    if j%5==0:print('train',a.name,'block',j,'series',len(seen),'peakMiB',round(rss()/1024**2,1),flush=True)
   assert seen==keys and int(g['n'])==a.series*28*len(snapshots(a.origin))
   with open(out/'statistics.npz.tmp','wb') as f:np.savez(f,**g)
   (out/'statistics.npz.tmp').replace(out/'statistics.npz');pd.DataFrame(coverage).to_csv(out/'training-coverage.csv',index=False)
   train=dict(series=len(seen),rows=int(g['n']),target_sum=float(g['sy']),target_sum_squares=float(g['yy']),target_stream_sha256=yh.hexdigest(),statistics_sha256=sha(out/'statistics.npz'),seconds=time.perf_counter()-train_start,peak_rss_bytes=rss(),feature_columns={arm:len(indices(arm)) for arm in ['M0','M1','M2','M3']})
   atomic_json(out/'training.json',train)
  models={};writers={};metricwriters={}
  for arm in ('M0','M1','M2','M3'):
   for k,lam in enumerate(LAMBDAS):
    tag=f'{arm}-c{k}';model=solve(g,arm,lam);models[tag]=model
    mp=out/f'{tag}-model.npz'
    if not mp.exists():
     with open(mp.with_suffix('.npz.tmp'),'wb') as f:np.savez(f,**model)
     mp.with_suffix('.npz.tmp').replace(mp)
    else:
     old=dict(np.load(mp));assert all(np.array_equal(old[key],value) for key,value in model.items())
  # Only now read development outcomes, with an explicit permitted origin.
  targets=con.sql(f"SELECT * FROM read_parquet('{B}/targets/{a.origin}-*.parquet')").df().set_index(['item_id','store_id']);assert targets.index.is_unique
  seen=set();evhash=hashlib.sha256();summaries={tag:[] for tag in models};cohortparts=[]
  def write_frame(frame,path,registry,tag):
   table=pa.Table.from_pandas(frame,preserve_index=False)
   if tag not in registry:registry[tag]=pq.ParquetWriter(path,table.schema,compression='zstd')
   registry[tag].write_table(table)
  for meta,u in selected_blocks(keys,a.origin,a.chunk):
   pairs=list(zip(meta.item_id,meta.store_id));assert not seen.intersection(pairs);seen.update(pairs)
   truth=targets.loc[pairs,[f'y_{h}' for h in range(1,29)]].to_numpy(float);assert truth.shape==(len(u),28) and np.isfinite(truth).all()
   evhash.update(truth.astype('<f8').tobytes());s,c=factors(u,meta,a.origin);st=training_stats(u)
   cp=meta.copy();cp['origin']=a.origin
   for key,val in st.items():cp[key]=val
   cp['xyz']=diagnostics(u,a.origin).xyz;cohortparts.append(cp)
   for tag,model in models.items():
    p=forecast(model,s,c);f=pd.concat([meta,pd.DataFrame(p,columns=[f'f_{h}' for h in range(1,29)])],axis=1);f['origin']=a.origin
    m=pd.concat([meta,score(truth,p,st)],axis=1);m['origin']=a.origin
    write_frame(f,out/f'{tag}-forecasts.parquet.tmp',writers,tag);write_frame(m,out/f'{tag}-metrics.parquet.tmp',metricwriters,tag)
    summaries[tag].append(dict(series=len(u),sum_mae=float(m.mae.sum()),sum_bias=float(m.bias.sum()),sum_abs=float(m.sum_abs.sum()),actual_sum=float(m.actual_sum.sum())))
  assert seen==keys
  for tag in models:
   writers[tag].close();metricwriters[tag].close()
   for kind in ['forecasts','metrics']:
    dest=out/f'{tag}-{kind}.parquet'
    if dest.exists():assert sha(dest)==sha(dest.with_suffix('.parquet.tmp'));dest.with_suffix('.parquet.tmp').unlink()
    else:dest.with_suffix('.parquet.tmp').replace(dest)
  pd.concat(cohortparts,ignore_index=True).to_parquet(out/'evaluation-cohort.parquet',index=False)
  records=[]
  for tag,parts in summaries.items():
   total=pd.DataFrame(parts).sum();assert int(total.series)==a.series
   records.append(dict(arm=tag[:2],candidate=int(tag[-1]),penalty=LAMBDAS[int(tag[-1])],origin=a.origin,series=a.series,predictions=a.series*28,macro_mae=total.sum_mae/a.series,bias=total.sum_bias/a.series,wape=total.sum_abs/total.actual_sum))
  pd.DataFrame(records).to_csv(out/'candidate-metrics.csv',index=False)
  assert rss()<LIMIT
  info=dict(status='PASS',origin=a.origin,series=a.series,training_rows=int(g['n']),forecasts_per_candidate=a.series*28,models=16,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=rss(),gate_bytes=LIMIT,evaluation_target_stream_sha256=evhash.hexdigest(),source_fingerprints=fp,final_holdout_used=False,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix!='.tmp'})
  atomic_json(out/'execution.json',info);stop.set();print(json.dumps({k:v for k,v in info.items() if k not in ('files','source_fingerprints')}),flush=True)
if __name__=='__main__':main()
