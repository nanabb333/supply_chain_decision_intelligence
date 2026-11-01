"""Week3 E1 read-only adapter. Accepted summaries only; no empirical computation."""
from pathlib import Path
import io,json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/week3/checkpoint-e1-v1'
SOURCES={
'forecast':'reports/week2/checkpoint-e2b-v1/FINAL_MODEL_COMPARISON.csv',
'paired':'reports/week2/checkpoint-e2b-v1/FINAL_PAIRED_COMPARISONS.csv',
'primary':'reports/week3/checkpoint-c2-v1/policy-comparison.csv',
'operating':'reports/week3/checkpoint-d1-v1/SCENARIO_COMPARISON.csv',
'initialization':'reports/week3/checkpoint-d3-v1/INITIALIZATION_COMPARISON.csv',
'structure':'reports/week3/checkpoint-d3-v1/INITIALIZATION_STRUCTURE.csv',
'categories':'reports/week3/checkpoint-d2-v1/category-comparison.csv',
'numeric':'reports/week3/checkpoint-d2-v1/numeric-comparison.csv',
'segments':'reports/phase1_6/run-v3/robustness/scenario_metrics.csv',
'windows':'reports/phase1_6/checkpoint-e-v1/diagnostics/window_sensitivity.csv'}
POLICIES=['MA28','M1','M3']
LABELS={'primary':'Primary','lead1':'Lead time · 1 day','lead7':'Lead time · 7 days','daily_review':'Daily review','empty_start':'Empty initial stock','double_start':'Double initial stock','no_buffer':'No safety buffer','buffer7':'Buffer · 7 days','penalty50':'Shortage penalty · 50','history1801_common':'Historical mean start','symmetric_forecast_common':'Median-target start'}
METRICS={'total_cost':'Hypothetical cost units','fill_percent':'Pooled simulated fill (%)','lost_units':'Lost proxy-demand units','stockout_series_days':'Stockout series-days','average_on_hand':'Mean end-day stock (units)','terminal_inventory':'Terminal stock (units)','terminal_pipeline':'Terminal pipeline (units)','day1_order_units':'Day-one orders (units)'}

def load(name):
 if name not in SOURCES:raise ValueError('Source not allowlisted')
 p=SOURCES[name];manifest=json.loads((OUT/'protected-manifest.json').read_text());raw=(ROOT/p).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=manifest[p]:raise ValueError('Accepted source hash mismatch: '+p)
 return pd.read_csv(io.BytesIO(raw),float_precision='round_trip')

def inventory(family='operating',selection='primary',policies=None):
 if family not in ['operating','initialization']:raise ValueError('Unsupported family')
 d=load(family)
 if family=='operating':
  d=d.rename(columns={'scenario':'configuration','total_hypothetical_cost':'total_cost','simulated_fill_rate':'pooled_fill','simulated_lost_units':'lost_units','simulated_fulfilled_units':'fulfilled_units','simulated_stockout_days':'stockout_series_days','average_on_hand_units':'average_on_hand','hypothetical_holding_cost':'holding_cost','hypothetical_shortage_cost':'shortage_cost','hypothetical_ordering_cost':'ordering_cost','terminal_on_hand_units':'terminal_inventory','terminal_pipeline_units':'terminal_pipeline','initial_on_hand_units':'initial_inventory','macro_fill_rate_positive_series':'macro_fill'})
 else:d=d.rename(columns={'method':'configuration'})
 if selection not in set(d.configuration):raise ValueError('Untested combination/configuration')
 policies=POLICIES if policies is None else policies
 if not set(policies)<=set(POLICIES):raise ValueError('Unknown policy')
 d=d[(d.configuration==selection)&d.model.isin(policies)].copy()
 # Rates recomputed only from their matched accepted numerators and denominators.
 d['pooled_fill']=d.fulfilled_units.div(d.observed_sales_proxy_units.where(d.observed_sales_proxy_units.ne(0)))
 d['fill_percent']=100*d.pooled_fill
 d['experiment_family']=family
 return d

def export(d,source,scope):
 out=d.copy();out['source_artifact']=SOURCES[source];out['evidence_scope']=scope
 out['limitations']='Frozen research; forecasts and inventory have different cohorts/windows; hypothetical inventory costs are not USD; no actual retailer impact.'
 return out.to_csv(index=False).encode()

def forecast_differences(policies=None):
 f=load('forecast').set_index('model');p=load('paired').copy()
 if policies is not None:
  p=p[p.comparison.map(lambda x:all(m in policies for m in x.split('_minus_')))]
 for r in p.itertuples():
  left,right=r.comparison.split('_minus_')
  if abs((f.loc[left,'macro_mae']-f.loc[right,'macro_mae'])-r.delta_mae)>1e-12:raise ValueError('Forecast reconciliation')
 return p
