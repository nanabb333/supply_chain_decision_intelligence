"""Read-only E1 adapters. Whitelisted development summaries; no training/raw loaders."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb
ROOT=Path(__file__).resolve().parents[2]
D='reports/week2/checkpoint-d1-v1/'
C='reports/week2/checkpoint-c-v2/'
B='reports/week2/checkpoint-b-v1/'
P='reports/phase1_6/checkpoint-e-v1/diagnostics/'
SOURCES={
 'paired':D+'PAIRED_OVERALL.csv','origin':D+'ORIGIN_STABILITY.csv','audit':D+'MATCHED_DATA_AUDIT.csv','daily':D+'DAILY_SUMMARY.csv','distribution':D+'PAIRED_DISTRIBUTION.csv','uncertainty':D+'UNCERTAINTY_RESULTS.csv','subgroups':D+'SUBGROUP_EVALUATION.csv','practical':D+'PRACTICAL_SCALE.json','acceptance':D+'FINAL_ACCEPTANCE.json','evidence':D+'EVIDENCE_REGISTER.csv','baseline_ranking':B+'analysis-v1/development-ranking.csv','baseline_origins':B+'analysis-v1/per-origin.csv','models':C+'analysis-v1/matched-per-origin.csv','quality':P+'quality_summary.csv','classes':P+'class_distribution.csv','windows':P+'window_sensitivity.csv','zero_runs':P+'zero_run_bands.csv',
 'memberships':D+'subgroup-memberships.parquet','losses':D+'matched-series-origin.parquet'}
FILTERS={'origin':'origin','store':'store_id','category':'cat_id','frequency':'frequency_band','cv':'cv_band','adi':'adi_band','xyz':'xyz','intensity':'intensity','item':'item_id'}
MODEL_NAMES={'ma28':'MA28','M1':'M1','M3':'M3'}
LIMITATIONS='Development-only; model-selection optimism; three origins; XYZ-only ridge comparison; no demonstrated inventory/service/cost benefit; final holdout not evaluated.'

def load(name):
 if name not in SOURCES:raise ValueError('Source not allowlisted')
 p=ROOT/SOURCES[name]
 if p.suffix=='.json':return json.loads(p.read_text())
 if p.suffix!='.csv':raise ValueError('Use bounded cohort adapter for detailed data')
 return pd.read_csv(p)

def origins():
 values=tuple(int(v) for v in sorted(load('audit').origin.unique()))
 if values!=(1801,1829,1857):raise ValueError('Unapproved temporal scope')
 return values

def origin_labels():
 d=load('daily').groupby('origin').target_date.min()
 return {int(o):str((pd.Timestamp(v)-pd.Timedelta(days=1)).date()) for o,v in d.items()}

def dates():
 d=load('daily');return d.target_date.min(),d.target_date.max()

def kpis():
 p=load('paired').set_index('comparison');a=load('audit');primary=p.loc['M3_minus_M1'];base=p.loc['M3_minus_MA28'];h=(a.last_target-a.first_target+1).unique()
 if len(h)!=1 or not a.series.eq(primary.n).all():raise ValueError('Accepted scope mismatch')
 ranking=load('baseline_ranking').sort_values('macro_mae')
 return dict(series=int(primary.n),origins=a.origin.nunique(),horizon=int(h[0]),improvement=float(primary.reduction_pct),vs_baseline=float(base.relative_delta_pct),improved_pct=float(primary.improved_pct),best_baseline=MODEL_NAMES.get(ranking.iloc[0].model,ranking.iloc[0].model),primary=primary.to_dict(),baseline=base.to_dict())

def benchmark():
 d=load('models');d=d[d.model.isin(MODEL_NAMES)].copy();d['model']=d.model.map(MODEL_NAMES)
 d=d.rename(columns={'mae':'MAE','bias':'Bias','series_origin_observations':'Series'})
 return d[['origin','model','MAE','Bias','Series']]

def query_cohort(filters=None):
 filters=filters or {}
 if set(filters)-set(FILTERS):raise ValueError('Unknown filter')
 for o in filters.get('origin',[]):
  if o not in origins():raise ValueError('Final/unapproved origin inaccessible')
 conditions=[];params=[]
 for key,values in filters.items():
  if not isinstance(values,(list,tuple)):raise ValueError('Filters must be lists')
  if values:
   conditions.append(FILTERS[key]+' IN ('+','.join('?' for _ in values)+')');params.extend(values)
 where=' AND '.join(conditions) or 'TRUE'
 cols=['item_id','store_id','cat_id','origin','frequency_band','cv_band','adi_band','xyz','intensity']+[m+'_'+v for m in ['M1','M3','MA28'] for v in ['mae','bias']]
 sql=f"""SELECT {','.join(cols)} FROM read_parquet('{ROOT/SOURCES['losses']}') f
 JOIN read_parquet('{ROOT/SOURCES['memberships']}') s USING(item_id,store_id,origin,xyz)
 WHERE {where} ORDER BY item_id,store_id,origin"""
 con=duckdb.connect()
 try:
  con.execute('SET threads=1');con.execute("SET memory_limit='128MB'");d=con.execute(sql,params).df()
 finally:con.close()
 if len(d)>len(origins())*kpis()['series'] or not set(d.origin)<=set(origins()):raise ValueError('Unexpected scope')
 if d.duplicated(['item_id','store_id','origin']).any():raise ValueError('Duplicate analytical keys')
 if d[[m+'_mae' for m in ['M1','M3','MA28']]].isna().any().any():raise ValueError('Missing model coverage')
 return d

def summarize(d):
 if d.empty:return pd.DataFrame(columns=['model','MAE','Bias','Series','Series-origin observations','Origins','Origin-balanced MAE'])
 rows=[]
 for m in ['MA28','M1','M3']:
  rows.append(dict(model=m,MAE=d[m+'_mae'].mean(),Bias=d[m+'_bias'].mean(),Series=len(d[['item_id','store_id']].drop_duplicates()),**{'Series-origin observations':len(d),'Origins':d.origin.nunique(),'Origin-balanced MAE':d.groupby('origin')[m+'_mae'].mean().mean()}))
 return pd.DataFrame(rows)

def paired_scope(d):
 if d.empty:return dict(available=False)
 delta=d.M3_mae-d.M1_mae;reference=d.M1_mae.mean()
 return dict(available=True,delta=float(delta.mean()),relative_pct=float(100*delta.mean()/reference) if reference else None,improved=int((delta< -1e-10).sum()),worsened=int((delta>1e-10).sum()),tied=int((abs(delta)<=1e-10).sum()),observations=len(d))

def export_csv(frame,source_ids,scope,metric,weighting='Equal series-origin memberships; 28 equally weighted target days per membership',period=None,scope_origins=None):
 if any(s not in SOURCES for s in source_ids):raise ValueError('Unapproved export source')
 if scope_origins is not None and any(o not in origins() for o in scope_origins):raise ValueError('Unapproved export origin')
 out=frame.copy();out['cohort_scope']=scope;out['scope_origins']=json.dumps(scope_origins if scope_origins is not None else list(origins()));out['metric_definition']=metric;out['weighting']=weighting;out['source_artifact']='; '.join(SOURCES[s] for s in source_ids);out['research_limitations']=LIMITATIONS;out['source_period']=period or ' to '.join(dates())
 return out.to_csv(index=False).encode('utf-8')
