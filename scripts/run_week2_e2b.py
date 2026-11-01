"""Authorized staged E2-B execution. Final scoring is a separate one-time process."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
import sys,json,time,resource,threading,argparse
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,duckdb
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from supply_chain.final_evaluation import *
from supply_chain import xyz_stream as ridge
O=R/'reports/week2/checkpoint-e2b-v1';RAW=R/'data/raw/m5-organizers-2026-10-13/sales_train_validation.csv'

def con():
 c=duckdb.connect();c.execute('SET threads=1');c.execute("SET memory_limit='256MB'");c.execute(f"SET temp_directory='{O}/duckdb-spill'");return c

def frame_write(df,name):
 p=O/name;t=p.with_name(p.name+'.tmp');df.to_parquet(t,index=False);publish(t,p)
def csv_write(df,name):
 p=O/name;t=p.with_name(p.name+'.tmp');df.to_csv(t,index=False);publish(t,p)
def npz_write(arr,name):
 p=O/name;t=p.with_name(p.name+'.tmp')
 with t.open('xb') as f:np.savez(f,**arr)
 publish(t,p)
def receipt(name,paths,**kwargs):
 atomic_json(O/name,{'status':'PASS','protocol_sha256':DIGEST,'files':{p:sha(O/p) for p in paths},**kwargs})
def check_receipt(name):
 a=json.loads((O/name).read_text());assert a['status']=='PASS' and a['protocol_sha256']==DIGEST
 assert all(sha(O/k)==v for k,v in a['files'].items());return a

def histories():
 for f in pd.read_csv(RAW,usecols=history_columns(),chunksize=512):
  yield f[META].reset_index(drop=True),f[[f'd_{i}' for i in range(1,1886)]].to_numpy(float)

def cohort_stage():
 seen=set();parts=[]
 for meta,u in histories():
  keys=list(zip(meta.item_id,meta.store_id));assert len(set(keys))==len(keys) and not seen.intersection(keys);seen.update(keys)
  valid=history_valid(u);stats=training_stats(u);f=meta.copy();f['origin']=1885;f['eligible']=valid;f['exclusion_reason']=np.where(valid,'','incomplete_or_invalid_history');f['history_days']=1885
  for k,v in stats.items():f[k]=v
  if valid.all():f['xyz']=diagnostics(u,1885).xyz
  else:f['xyz']='UNAVAILABLE'
  parts.append(f)
 f=pd.concat(parts,ignore_index=True);f,q=final_bands(f);frame_write(f,'cohort.parquet');atomic_json(O/'historical-quartiles.json',{'q25':float(q[0]),'q50':float(q[1]),'q75':float(q[2])})
 assert len(f)==N and f.eligible.all(),'Unexpected eligibility; stop before fitting'
 receipt('COHORT_COMPLETE.json',['cohort.parquet','historical-quartiles.json'],series=N,history_end=1885,final_targets_read=False)

def fit_stage():
 check_receipt('COHORT_COMPLETE.json');g=ridge.empty_stats();coverage=[];seen=set();start=time.perf_counter()
 for j,(meta,u) in enumerate(histories()):
  assert history_valid(u).all();keys=set(zip(meta.item_id,meta.store_id));assert not seen.intersection(keys);seen.update(keys)
  static=ridge.static_features(meta);before=int(g['n'])
  for t in SNAPSHOTS:
   s,c=ridge.factors(u[:,:t],meta,t,static);ridge.add_block(g,s,c,u[:,t:t+28])
  coverage.append({'chunk':j,'series':len(meta),'snapshot_count':len(SNAPSHOTS),'first_snapshot':min(SNAPSHOTS),'last_snapshot':max(SNAPSHOTS),'last_training_target':max(SNAPSHOTS)+28,'training_rows':int(g['n'])-before})
  if j%5==0:print('fit chunk',j,'series',len(seen),'peakMiB',round(rss()/1024**2,1),flush=True)
 assert len(seen)==N and int(g['n'])==46100880
 npz_write(g,'statistics.npz');csv_write(pd.DataFrame(coverage),'training-coverage.csv')
 paths=['statistics.npz','training-coverage.csv']
 for arm in ['M1','M3']:
  m=ridge.solve(g,arm,1e-6);name=arm+'-model.npz';npz_write(m,name);paths.append(name)
 receipt('FIT_COMPLETE.json',paths,training_rows=int(g['n']),last_training_label=1877,seconds=time.perf_counter()-start,peak_rss_bytes=rss(),final_targets_read=False)

def predict_stage():
 check_receipt('FIT_COMPLETE.json');cohort= pd.read_parquet(O/'cohort.parquet');expected=set(zip(cohort.item_id,cohort.store_id));seen=set()
 models={a:dict(np.load(O/(a+'-model.npz'))) for a in ['M1','M3']};path=O/'forecasts.parquet';temp=path.with_name(path.name+'.tmp');writer=None;sample=[]
 sortedkeys=sorted(expected);samplekeys={sortedkeys[0],sortedkeys[len(sortedkeys)//2],sortedkeys[-1]}
 for meta,u in histories():
  s,c=final_factors(u,meta);arrays={a:ridge.forecast(models[a],s,c) for a in ['M1','M3']};arrays['MA28']=np.repeat(u[:,-28:].mean(axis=1)[:,None],28,axis=1)
  for i,key in enumerate(zip(meta.item_id,meta.store_id)):
   assert key not in seen;seen.add(key)
   if key in samplekeys:
    scalar=sum(float(v) for v in u[i,-28:])/28;assert np.isclose(scalar,arrays['MA28'][i,0],rtol=1e-12,atol=1e-12)
    sample.append({'item_id':key[0],'store_id':key[1],'independent_scalar_ma28':scalar,'saved_formula_ma28':float(arrays['MA28'][i,0])})
  table=pa.Table.from_pandas(long_forecasts(meta,arrays),preserve_index=False)
  if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
  writer.write_table(table)
 writer.close();assert seen==expected;publish(temp,path)
 csv_write(pd.DataFrame(sample),'independent-ma28-sample.csv')
 c=con();q=f"read_parquet('{path}')"
 coverage=c.sql(f"SELECT model,count(*) n,min(origin) omin,max(origin) omax,min(target_day) first_day,max(target_day) last_day,min(prediction) minimum,sum(CASE WHEN NOT isfinite(prediction) THEN 1 ELSE 0 END) invalid FROM {q} GROUP BY model ORDER BY model").df()
 assert len(coverage)==3 and coverage.n.eq(N*28).all() and coverage.omin.eq(1885).all() and coverage.omax.eq(1885).all() and coverage.first_day.eq(1886).all() and coverage.last_day.eq(1913).all() and coverage.minimum.ge(0).all() and coverage.invalid.eq(0).all()
 bad=c.sql(f"SELECT count(*) FROM (SELECT item_id,store_id,model,count(*) n,count(distinct horizon) h,min(horizon) lo,max(horizon) hi FROM {q} GROUP BY ALL HAVING n!=28 OR h!=28 OR lo!=1 OR hi!=28)").fetchone()[0];assert bad==0
 assert c.sql(f"SELECT count(*) FROM {q} WHERE target_day!=1885+horizon").fetchone()[0]==0
 assert c.sql(f"SELECT count(*) FROM (SELECT DISTINCT item_id,store_id FROM {q} EXCEPT SELECT item_id,store_id FROM read_parquet('{O}/cohort.parquet'))").fetchone()[0]==0;c.close()
 csv_write(coverage,'prediction-coverage.csv')
 paths=['forecasts.parquet','cohort.parquet','historical-quartiles.json','statistics.npz','training-coverage.csv','M1-model.npz','M3-model.npz','prediction-coverage.csv','independent-ma28-sample.csv']
 atomic_json(O/'FINAL_PREDICTION_MANIFEST.json',{'protocol_sha256':DIGEST,'coverage_validated':True,'files':{p:sha(O/p) for p in paths},'predictions_per_model':N*28,'predictions_total':N*28*3,'origin':1885,'target_days':[1886,1913],'final_targets_read':False})
 receipt('PREDICT_COMPLETE.json',['FINAL_PREDICTION_MANIFEST.json'],final_targets_read=False)

def score_stage():
 frozen=require_prediction_manifest(O);cohort=pd.read_parquet(O/'cohort.parquet').sort_values(['item_id','store_id']).reset_index(drop=True);keys=pd.MultiIndex.from_frame(cohort[['item_id','store_id']]);assert keys.is_unique
 # First and only actual final-target extraction, after prediction freeze and explicit user authorization.
 target=pd.read_csv(RAW,usecols=target_columns()).set_index(['item_id','store_id']);assert target.index.is_unique and set(target.index)==set(keys)
 y=target.loc[keys,[f'd_{i}' for i in range(1886,1914)]].to_numpy(float);assert y.shape==(N,28) and np.isfinite(y).all() and (y>=0).all() and (y==np.floor(y)).all()
 td=pd.DataFrame({'item_id':np.repeat(cohort.item_id.to_numpy(),28),'store_id':np.repeat(cohort.store_id.to_numpy(),28),'origin':1885,'horizon':np.tile(np.arange(1,29),N),'target_day':np.tile(np.arange(1886,1914),N),'actual':y.ravel(),'observed':True});frame_write(td,'targets.parquet');del target,td
 stats={k:cohort[k].to_numpy() for k in ['scale_abs','scale_sq']};parts=[];summary=[];horizon=[];zero=[];c=con()
 for model in MODELS:
  f=c.sql(f"SELECT item_id,store_id,horizon,prediction FROM read_parquet('{O}/forecasts.parquet') WHERE model='{model}' ORDER BY item_id,store_id,horizon").df()
  assert len(f)==N*28 and np.array_equal(f.item_id.to_numpy()[::28],cohort.item_id.to_numpy()) and np.array_equal(f.store_id.to_numpy()[::28],cohort.store_id.to_numpy())
  pred=f.prediction.to_numpy().reshape(N,28);metric=score(y,pred,stats);m=pd.concat([cohort[['item_id','store_id','origin']],metric],axis=1);m['model']=model;parts.append(m)
  summary.append({'model':model,'series':N,'target_observations':int(metric.n.sum()),'macro_mae':float(metric.mae.mean()),'bias':float(metric.bias.mean()),'wape':float(metric.sum_abs.sum()/metric.actual_sum.sum()) if metric.actual_sum.sum()>0 else None,'mase':float(metric.mase.mean()),'rmsse':float(metric.rmsse.mean()),'scaled_valid':int(metric.mase.notna().sum()),'mean_abs_total_error':float(metric.abs_total_error.mean())})
  for h in range(28):horizon.append({'model':model,'horizon':h+1,'mae':float(np.abs(pred[:,h]-y[:,h]).mean()),'bias':float((pred[:,h]-y[:,h]).mean())})
  for kind,mask in [('zero',y==0),('positive',y>0)]:zero.append({'model':model,'target_group':kind,'observations':int(mask.sum()),'mae':float(np.abs(pred-y)[mask].mean()),'bias':float((pred-y)[mask].mean()),'prediction_mass':float(pred[mask].sum())})
 c.close();m=pd.concat(parts,ignore_index=True);frame_write(m,'metrics.parquet');csv_write(pd.DataFrame(summary),'FINAL_MODEL_COMPARISON.csv');csv_write(pd.DataFrame(horizon),'HORIZON_METRICS.csv');csv_write(pd.DataFrame(zero),'ZERO_POSITIVE_DIAGNOSTICS.csv')
 require_prediction_manifest(O)
 receipt('SCORE_COMPLETE.json',['targets.parquet','metrics.parquet','FINAL_MODEL_COMPARISON.csv','HORIZON_METRICS.csv','ZERO_POSITIVE_DIAGNOSTICS.csv'],final_targets_read=True,one_time_scoring=True)

def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
def monitor(stop,stage):
 while not stop.wait(.05):
  if rss()>=3*1024**3:
   atomic_json(O/(stage+'-MEMORY_STOP.json'),{'rss_bytes':rss()});os._exit(90)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cohort','fit','predict','score']);a=ap.parse_args();verify_protocol(R)
 gate=json.loads((O/'pre-execution-gate.json').read_text());assert gate['status']=='PASS' and gate['protocol_sha256']==DIGEST
 auth=json.loads((O/'authorization.json').read_text());assert 'one-time scoring' in auth['authorized']
 code=json.loads((O/'RUNNER_MANIFEST.json').read_text());assert all(sha(R/p)==h for p,h in code.items())
 atomic_json(O/(a.stage+'-STARTED.json'),{'stage':a.stage,'protocol_sha256':DIGEST,'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),'code_hashes':code})
 stop=threading.Event();threading.Thread(target=monitor,args=(stop,a.stage),daemon=True).start()
 try:
  with threadpool_limits(1):globals()[a.stage+'_stage']()
  print(json.dumps({'stage':a.stage,'status':'PASS','peak_rss_bytes':rss()}),flush=True)
 except BaseException as e:
  atomic_json(O/(a.stage+'-FAILED.json'),{'error':repr(e),'action':'STOP for user review; preserve attempt, no corrective rerun'});raise
 finally:stop.set()
if __name__=='__main__':main()
