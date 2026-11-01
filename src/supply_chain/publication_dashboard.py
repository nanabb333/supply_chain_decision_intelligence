"""Versioned F presentation; accepted E1 and E2-B artifacts are read-only."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from . import forecast_dashboard as dev
R=Path(__file__).resolve().parents[2];E=R/'reports/week2/checkpoint-e2b-v1';F=R/'reports/week2/checkpoint-f-v1'
PERIOD='2016-03-28 to 2016-04-24';FINAL='Fixed Final Holdout Evaluation';DEVELOPMENT='Development Evaluation'
@st.cache_data
def table(name):
 allowed=['FINAL_MODEL_COMPARISON','FINAL_PAIRED_COMPARISONS','FINAL_UNCERTAINTY','DEVELOPMENT_VS_FINAL','FINAL_SUBGROUP_COMPARISONS','FINAL_PRACTICAL_THRESHOLDS']
 if name not in allowed:raise ValueError('Source not allowlisted')
 return pd.read_csv(E/(name+'.csv'),float_precision='round_trip')
@st.cache_data
def rows():
 m=pd.read_parquet(E/'metrics.parquet');c=pd.read_parquet(E/'cohort.parquet');assert len(c)==30490 and not c.duplicated(['item_id','store_id']).any()
 return m.merge(c.drop(columns=['origin']),on=['item_id','store_id'],validate='many_to_one')
def select(filters):
 d=rows()
 for key,value in filters.items():
  if key not in ['store_id','cat_id','xyz','frequency_band','intensity','item_id']:raise ValueError('Unknown filter')
  if value and value!='All':d=d[d[key].astype(str)==str(value)]
 return d

def summarize(d):
 return d.groupby('model',as_index=False).agg(MAE=('mae','mean'),Bias=('bias','mean'),Series=('mae','size'),Target_observations=('n','sum'))
def export(df,source,scope='Full eligible population'):
 out=df.copy()
 for k,v in {'evaluation_scope':FINAL,'period':PERIOD,'origin':1885,'horizon':28,'filter_scope':scope,'weighting':'Equal series and 28 horizons; one final origin','metric_definition':'MAE=mean absolute unit error; Bias=prediction minus actual','source_artifact':source,'limitations':'Prior descriptive exposure; no inventory/cost benefit; no post-holdout tuning'}.items():out[k]=v
 if source.endswith('DEVELOPMENT_VS_FINAL.csv'):
  out['evaluation_scope']='Separate development and fixed final summaries'
  out['period']='Development: 2016-01-04 to 2016-03-27; Final: 2016-03-28 to 2016-04-24'
  out['origin']='Development: 1801,1829,1857; Final: 1885'
  out['weighting']='Each period computed separately; development equal origins; no pooled observations'
 return out.to_csv(index=False).encode()
def download(df,label,key,source,scope='Full eligible population'):
 st.download_button(label,export(df,source,scope),key+'.csv','text/csv',key=key)
def final_overview():
 f=table('FINAL_MODEL_COMPARISON');p=table('FINAL_PAIRED_COMPARISONS').set_index('comparison');st.header('MA28 retains the overall accuracy advantage')
 c=st.columns(3);c[0].metric('Series',f'{int(f.series.min()):,}');c[1].metric('M3 vs M1 reduction',f'{p.loc["M3_minus_M1","reduction_pct"]:.6f}%');c[2].metric('M3 vs MA28 higher MAE',f'{p.loc["M3_minus_MA28","relative_delta_pct"]:.6f}%')
 st.info('The XYZ increment fails the frozen practical screen. A small gain over M1 does not establish superiority over MA28 or operational value.')
 dev.model_chart(f.rename(columns={'macro_mae':'MAE','bias':'Bias'}),'final_overview_chart');st.dataframe(f[['model','macro_mae','series','target_observations']],hide_index=True,width='stretch',column_config={'macro_mae':st.column_config.NumberColumn(format='%.9f')})
 download(f,'Download final model results','final_overview_export','reports/week2/checkpoint-e2b-v1/FINAL_MODEL_COMPARISON.csv')
def final_benchmark():
 st.header('Keep final and development comparisons separate')
 f=table('FINAL_MODEL_COMPARISON');dev.model_chart(f.rename(columns={'macro_mae':'MAE','bias':'Bias'}),'final_benchmark_chart')
 st.caption('Absolute final macro MAE uses a zero-based axis. Lower is better; errors are units, not revenue.')
 st.dataframe(f[['model','macro_mae','bias','series','target_observations']],hide_index=True,width='stretch',column_config={'macro_mae':st.column_config.NumberColumn(format='%.9f')})
 st.subheader('Direction replicated across periods')
 c=table('DEVELOPMENT_VS_FINAL');st.dataframe(c[['comparison','relative_delta_pct_development','relative_delta_pct_final']],hide_index=True,width='stretch');st.caption('Relative delta = 100 × (left − reference) / reference. Negative favors the left model. Development selected models; final did not. Periods are never pooled.')
 download(c,'Download period comparison','final_comparison_export','reports/week2/checkpoint-e2b-v1/DEVELOPMENT_VS_FINAL.csv','Separate development and final summaries; no pooled observations')
def final_incremental():
 st.header('The XYZ increment is too small for the declared screen')
 p=table('FINAL_PAIRED_COMPARISONS');r=p.set_index('comparison').loc['M3_minus_M1'];c=st.columns(3);c[0].metric('M3 minus M1 · units/day',f'{r.delta_mae:+.9f}');c[1].metric('Relative reduction',f'{r.reduction_pct:.6f}%');c[2].metric('Series improved',f'{int(r.improved):,}')
 st.markdown('**M1:** continuous demand features plus base inputs. **M3:** identical design plus past-only XYZ labels. The primary scope is XYZ incremental predictive value, not revenue-based ABC–XYZ.')
 st.warning('Fails both illustrative thresholds: 0.1% relative reduction and 0.001 units/day. These thresholds are not measured economic break-even values.')
 u=table('FINAL_UNCERTAINTY');u=u[u.comparison=='M3_minus_M1'];st.dataframe(u[['method','estimate','p025','p975','status']],hide_index=True,width='stretch');st.caption('Conditional dependence sensitivities, not formal confidence guarantees. The 28-day block method is unavailable for a single final horizon.')
 download(p,'Download final paired comparisons','final_paired_export','reports/week2/checkpoint-e2b-v1/FINAL_PAIRED_COMPARISONS.csv')
def final_explorer():
 st.header('Explore final subgroups without selecting a new model')
 st.caption('Historical group definitions use only pre-origin data. These overlapping slices remain exploratory. No post-holdout optimization is authorized.')
 d=rows();filters={}
 for key,label in [('store_id','Store'),('cat_id','Category'),('xyz','Past-only XYZ'),('frequency_band','Positive-sales frequency'),('intensity','Historical intensity')]:
  filters[key]=st.selectbox(label,['All']+sorted(d[key].dropna().astype(str).unique().tolist()),key='final_'+key)
 filters['item_id']=st.text_input('Exact item ID (optional)',key='final_item')
 subset=select(filters);summary=summarize(subset)
 if subset.empty:st.info('No matching series. Clear filters to restore the population.');return
 n=int(summary.Series.min());st.write(f'**Selected scope:** {n:,} item–store series; {n*28:,} targets per model; origin 1885 only.')
 dev.model_chart(summary,'final_selected');st.dataframe(summary,hide_index=True,width='stretch',column_config={'MAE':st.column_config.NumberColumn(format='%.9f')})
 download(summary,'Export selected final comparison','final_selected_export','reports/week2/checkpoint-e2b-v1/metrics.parquet + cohort.parquet',json.dumps(filters,sort_keys=True))
def evidence():
 st.header('Final evaluation is complete and frozen')
 st.success('One-time final evaluation completed in E2-B. This dashboard only reads saved evidence.')
 st.markdown('**Development:** origins 1801, 1829 and 1857; selection and evaluation share those blocks.\n\n**Fixed final:** origin 1885, 28 targets from March 28 to April 24, 2016. The period had previous descriptive exposure; it is not a pristine blind test.\n\n**Primary scope:** XYZ-only, fixed streamed ridge, macro unit MAE. Prices, ABC and revenue are excluded.\n\n**Limits:** no observed inventory availability, lead times, cost savings or operational impact. Retrospective diagnostic labels are never forecasting inputs.')
 st.code('95466483b6e04f960a4f050dbed2bd0eb83de9641e227614aab372bb07cd910c',language=None)
 st.caption('Accepted canonical protocol SHA-256; source fingerprints and one-time stage receipts remain preserved.')
 reg=pd.read_csv(F/'FINAL_EVIDENCE_REGISTER.csv');st.dataframe(reg[['claim_id','value','scope','time_period']],hide_index=True,width='stretch');st.download_button('Download publication evidence register',reg.to_csv(index=False).encode(),'publication_evidence.csv','text/csv')
 for title,name in [('Executive research report','Executive_Research_Report.pdf'),('Technical appendix','Technical_Appendix.pdf')]:
  path=F/'publication-v2'/name
  if path.exists():st.download_button(title,path.read_bytes(),name,'application/pdf')
def render():
 st.set_page_config(page_title='M5 | Final research publication',layout='wide')
 st.markdown('<style>.stApp{background:#F7F9FC;color:#172B4D}.block-container{max-width:1240px;padding-top:1.5rem}h1{font-size:1.7rem!important}h2{font-size:1.35rem!important}[data-testid="stMetricValue"]{font-size:1.55rem}@media(max-width:640px){.block-container{padding:1rem}[data-testid="stMetricValue"]{font-size:1.3rem}}</style>',unsafe_allow_html=True)
 st.title('Does demand segmentation improve retail forecasting?')
 scope=st.selectbox('Evaluation scope',[FINAL,DEVELOPMENT],key='publication_scope')
 st.caption((PERIOD+' · origin 1885 · 28-day horizon') if scope==FINAL else '2016-01-04 to 2016-03-27 · origins 1801 / 1829 / 1857')
 view=st.selectbox('Analytical view',dev.VIEWS,key='publication_view')
 st.caption('Retrospective classification diagnostics have their own explicit historical scope; they are not pooled with either forecast period.')
 if view==dev.VIEWS[-1]:evidence()
 elif scope==DEVELOPMENT:dict(zip(dev.VIEWS[:-1],[dev.executive,dev.diagnostics,dev.benchmarking,dev.incremental,dev.explorer]))[view]()
 else:dict(zip(dev.VIEWS[:-1],[final_overview,dev.diagnostics,final_benchmark,final_incremental,final_explorer]))[view]()
 st.divider();st.caption('Publication layer · saved evidence only · no retraining or post-holdout tuning · no demonstrated inventory benefit')
