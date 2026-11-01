"""Evidence-grounded E1 forecasting interface; read-only development results."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from . import forecast_dashboard_data as data
VIEWS=['Executive overview','Demand & classification','Forecast benchmarking','XYZ incremental value','Subgroup explorer','Research evidence']
COLORS={'MA28':'#52657C','M1':'#284B80','M3':'#087F8C','Selected sample':'#284B80','M5 population':'#087F8C','Improved':'#087F8C','Worsened':'#AF5A28','Tied':'#728197'}
@st.cache_data
def table(name):return data.load(name)
@st.cache_data(max_entries=16)
def cohort(filters_json='{}'):return data.query_cohort(json.loads(filters_json))

def chart(fig,key,percent=False):
 fig.update_layout(template='plotly_white',font=dict(family='Arial, sans-serif',size=13,color='#172B4D'),paper_bgcolor='white',plot_bgcolor='white',margin=dict(l=15,r=15,t=20,b=15),height=300,legend=dict(orientation='h',y=1.14,x=0),hoverlabel=dict(bgcolor='white',font_color='#172B4D'))
 if percent:fig.update_yaxes(tickformat='.0%')
 st.plotly_chart(fig,width='stretch',config={'displayModeBar':False,'responsive':True},key=key)

def download(frame,ids,label,key,scope='Full accepted development cohort',metric='Daily recorded-unit error: MAE=mean(abs(prediction-actual)); Bias=mean(prediction-actual)',**kwargs):
 st.download_button(label,data.export_csv(frame,ids,scope,metric,**kwargs),key+'.csv','text/csv',key=key)

def model_chart(df,key):
 fig=px.bar(df,x='model',y='MAE',color='model',color_discrete_map=COLORS,labels={'model':'Model','MAE':'MAE · units / day'},hover_data={'MAE':':.9f','Bias':':.9f'})
 fig.update_yaxes(rangemode='tozero');fig.update_layout(showlegend=False);chart(fig,key)

def executive():
 k=data.kpis();st.header('A small XYZ gain does not beat the baseline')
 st.caption('Development-stage comparison · equal series, horizon and origin weights · lower MAE is better')
 cols=st.columns(3)
 for c,label,value in zip(cols,['Evaluated series','Development origins','Forecast horizon'],[f"{k['series']:,}",str(k['origins']),f"{k['horizon']} days"]):c.metric(label,value)
 cols=st.columns(3)
 for c,label,value in zip(cols,['M3 vs M1 · MAE reduction','M3 vs MA28 · higher MAE','Series improving · M3 vs M1'],[f"{k['improvement']:.4f}%",f"+{k['vs_baseline']:.4f}%",f"{k['improved_pct']:.2f}%"]):c.metric(label,value)
 st.info(f"{k['best_baseline']} is the strongest reported overall baseline. XYZ contributes very limited incremental value in the accepted ridge experiment; no inventory benefit has been demonstrated.")
 left,right=st.columns([1.2,1])
 with left:
  st.subheader('MA28 retains the overall advantage');b=data.benchmark().groupby('model',as_index=False)[['MAE','Bias']].mean();model_chart(b,'overview_chart')
 with right:
  st.subheader('Read the comparisons correctly')
  st.markdown('**M3 vs M1** isolates adding XYZ labels to the same base and continuous features.\n\n**M3 vs MA28** compares different model designs. Its difference cannot be credited solely to segmentation.\n\n**Subgroups** can favor learned models while overall results favor MA28. Explore all groups, including unfavorable ones.')
 st.caption('Series improvement counts average each item–store series over the development origins before comparing. Source: D1 paired overall results; baseline ranking: B.')
 download(table('paired'),['paired','baseline_ranking'],'Download executive evidence','overview_export',weighting='Equal series/horizon/origin; win counts on per-series means across origins')

def diagnostics():
 st.header('Classification is concentrated—and scope matters')
 q=table('quality');cl=table('classes');w=table('windows');z=table('zero_runs')
 st.warning('Retrospective descriptive evidence only. These Phase 1.6 summaries overlap the reserved historical period and are never forecasting inputs or final-holdout evaluation results.')
 st.caption('Descriptive period: '+str(q.horizon.iloc[0])+'. The purposive sample overlaps the population; it is not a random representative sample.')
 left,right=st.columns(2)
 with left:
  st.subheader('Z dominates both scopes')
  chart(px.bar(cl,x='xyz',y='share_all',color='cohort',barmode='group',color_discrete_map=COLORS,hover_data=['series','total_series'],labels={'xyz':'XYZ class','share_all':'Share of series','cohort':'Scope'}),'classes',True)
 with right:
  st.subheader('Sample demand is more frequently positive')
  chart(px.bar(q,x='cohort',y='mean_positive_frequency',color='cohort',color_discrete_map=COLORS,hover_data=['series'],labels={'cohort':'Scope','mean_positive_frequency':'Mean positive-sales frequency'}).update_layout(showlegend=False),'frequency',True)
 st.subheader('ADI triggers often; fewer labels actually change')
 st.caption('Triggered means ADI crosses the rule cutoff. Effective means the override changes a non-Z CV-only label to Z. Do not add overlapping rule contributions.')
 pretty=q[['cohort','series','adi_trigger_count','adi_effective_count','adi_effective_rate']].rename(columns={'cohort':'Scope','series':'Series','adi_trigger_count':'ADI triggers','adi_effective_count':'Labels changed by ADI','adi_effective_rate':'Share changed'})
 st.dataframe(pretty,hide_index=True,width='stretch',column_config={'Share changed':st.column_config.NumberColumn(format='percent')})
 left,right=st.columns(2)
 with left:
  st.subheader('Window changes: selected sample only')
  scenarios=w[w.scenario!='baseline'].copy();chart(px.bar(scenarios,x='scenario',y='changed',color_discrete_sequence=['#284B80'],hover_data=['series','xyz_changed','abc_changed'],labels={'scenario':'Observation scenario','changed':'ABC–XYZ segments changed'}),'windows')
  st.caption('Population alternate-window sensitivity is unavailable. ABC and XYZ change counts can overlap; sample revenue ranks retain the original portfolio context.')
 with right:
  st.subheader('Internal zero periods vary in length')
  chart(px.bar(z,x='band',y='zero_day_share',color='cohort',barmode='group',color_discrete_map=COLORS,hover_data=['runs','zero_days'],labels={'band':'Consecutive zero-sales days','zero_day_share':'Share of internal zero days'}),'zero_runs',True)
  st.caption('Internal runs exclude leading/trailing zeros. Zero recorded sales do not prove stockouts or absent demand.')
 with st.expander('Scope details and diagnostic exports'):
  st.dataframe(q[['cohort','series','eligible','excluded','hhi','effective_classes']],hide_index=True,width='stretch')
  for frame,ids,label,key in [(cl,['classes'],'Export class distribution','class_export'),(q,['quality'],'Export diagnostic summary','quality_export'),(w,['windows'],'Export sample-window sensitivity','window_export'),(z,['zero_runs'],'Export internal zero runs','zero_export')]:
   download(frame,ids,label,key,scope='Retrospective descriptive; scope per cohort column',metric='Classification/zero-sales diagnostics, not forecast metrics',period=q.horizon.iloc[0],scope_origins=[],weighting='Descriptive cohort counts/shares; sample and population separate')

def benchmarking():
 st.header('Benchmark before interpreting the ablation')
 b=data.benchmark();available=['All development origins']+[str(o) for o in data.origins()];choice=st.selectbox('Benchmark origin',available,key='benchmark_origin',format_func=lambda v: v if v==available[0] else data.origin_labels()[int(v)])
 filtered=b if choice==available[0] else b[b.origin==int(choice)]
 summary=filtered.groupby('model',as_index=False).agg(MAE=('MAE','mean'),Bias=('Bias','mean'),Series=('Series','min'))
 left,right=st.columns(2)
 with left:st.subheader('Overall errors favor the simple baseline');model_chart(summary,'benchmark_mae')
 with right:
  st.subheader('Bias is a separate question');fig=px.bar(summary,x='model',y='Bias',color='model',color_discrete_map=COLORS,labels={'Bias':'Bias · units / day'});fig.update_layout(showlegend=False);fig.add_hline(y=0,line_color='#172B4D');chart(fig,'benchmark_bias')
 st.dataframe(summary,hide_index=True,width='stretch',column_config={'MAE':st.column_config.NumberColumn(format='%.6f'),'Bias':st.column_config.NumberColumn(format='%.6f')})
 st.caption('Daily unit MAE, not revenue. All models share the same eligible series and target dates. Negative bias means underforecasting; mean errors can hide tail events.')
 st.subheader('The baseline advantage persists by origin')
 fig=px.line(filtered,x='origin',y='MAE',color='model',markers=True,color_discrete_map=COLORS,labels={'origin':'Forecast origin · day index','MAE':'MAE · units / day'});fig.update_xaxes(type='category',tickvals=list(data.origins()),ticktext=list(data.origin_labels().values()));fig.update_yaxes(rangemode='tozero');chart(fig,'benchmark_origins')
 with st.expander('Other accepted baseline candidates'):
  ranking=table('baseline_ranking')
  if choice!=available[0]:ranking=table('baseline_origins')[table('baseline_origins').origin==int(choice)][['model','macro_mae']]
  st.dataframe(ranking.sort_values('macro_mae'),hide_index=True,width='stretch');st.caption('Candidate ranking is development-only; these same blocks were used for model selection.')
 download(filtered,['models'],'Download benchmark table','benchmark_export',scope=choice,scope_origins=list(data.origins()) if choice==available[0] else [int(choice)],weighting='Equal series/horizon; equal selected origins')

def incremental():
 st.header('XYZ adds very limited incremental value')
 st.markdown('**M1 = base + continuous demand features**  ·  **M3 = M1 + XYZ labels**')
 p=table('paired').set_index('comparison').loc['M3_minus_M1'];scale=table('practical');o=table('origin');o=o[o.comparison=='M3_minus_M1']
 cols=st.columns(3);cols[0].metric('Paired MAE difference',f'{p.delta_mae:+.6f}');cols[1].metric('Relative MAE reduction',f'{p.reduction_pct:.4f}%');cols[2].metric('28-day absolute-error reduction',f"{scale['reduction_abs_error_sum_per_series_28days']:.5f}")
 st.caption('Negative paired delta favors M3. The 28-day value sums daily absolute-error differences; it is not inventory saved or the error of a 28-day total.')
 left,right=st.columns(2)
 with left:
  st.subheader('More series improve, but many worsen')
  counts=pd.DataFrame({'Outcome':['Improved','Worsened','Tied'],'Series':[p.improved,p.worsened,p.tied]})
  chart(px.bar(counts,x='Outcome',y='Series',color='Outcome',color_discrete_map=COLORS).update_layout(showlegend=False),'wins')
 with right:
  st.subheader('Small gains persist across origins')
  fig=px.bar(o,x=o.origin.astype(str),y='delta_mae',color_discrete_sequence=['#087F8C'],labels={'x':'Development origin','delta_mae':'M3 − M1 · units / day'});fig.update_xaxes(tickvals=[str(v) for v in data.origins()],ticktext=list(data.origin_labels().values()));fig.add_hline(y=0,line_color='#172B4D');chart(fig,'increment_origins')
 st.subheader('Most paired changes are close to zero')
 dist=table('distribution');chart(px.bar(dist,x='delta_bin',y='series',color_discrete_sequence=['#284B80'],labels={'delta_bin':'MAE difference bin · units / day','series':'Series'}).update_xaxes(tickangle=-35),'paired_distribution')
 st.info('The observed gain fails D1’s illustrative combined complexity screen. That screen is not an economically calibrated break-even; no inventory or service benefit is established.')
 with st.expander('Dependence-sensitive uncertainty and exact values'):
  u=table('uncertainty');u=u[u.comparison=='M3_minus_M1'];st.dataframe(u[['method','estimate','p025','p975']],hide_index=True,width='stretch')
  st.caption('Descriptive resampling ranges, not formal confidence coverage. Few origins, shared shocks and model-selection optimism remain. Daily predictions are not independent replications.')
  download(u,['uncertainty'],'Export uncertainty ranges','uncertainty_export',metric='Paired MAE difference; descriptive resampling percentiles')
 download(table('paired'),['paired','distribution'],'Download paired comparison','paired_export',weighting='Per-series equal-origin means; equal series')

def reset_filters():
 for key in data.FILTERS:st.session_state['filter_'+key]=[]

def explorer():
 st.header('Explore a matched historical cohort')
 st.caption('Exploratory subgroup comparisons. Filters intersect; no model or threshold is refitted. Empty selections mean all values.')
 full=cohort();filters={};labels={'origin':'Forecast origin','store':'Store','category':'Category','frequency':'Historical positive-sales frequency','cv':'Historical CV band','adi':'Historical ADI band','xyz':'Historical XYZ','intensity':'Historical sales intensity','item':'Item (optional detail)'}
 with st.expander('Filter cohort',expanded=True):
  cols=st.columns(2)
  for i,(name,column) in enumerate(data.FILTERS.items()):
   with cols[i%2]:filters[name]=st.multiselect(labels[name],sorted(full[column].unique().tolist()),key='filter_'+name,format_func=(lambda v: data.origin_labels()[int(v)]) if name=='origin' else str)
  st.button('Reset cohort filters',on_click=reset_filters)
 d=cohort(json.dumps(filters,sort_keys=True));summary=data.summarize(d);pair=data.paired_scope(d)
 if d.empty:
  st.warning('No matched observations for this filter combination. No estimates are available. Reset or broaden the filters.');return
 n=summary.iloc[0];cols=st.columns(3);cols[0].metric('Unique item–store series',f"{int(n['Series']):,}");cols[1].metric('Series-origin observations',f"{len(d):,}");cols[2].metric('Matched model coverage',f"{100*len(d.dropna(subset=['M1_mae','M3_mae','MA28_mae']))/len(d):.0f}%")
 st.caption('MAE and bias below equally weight selected series-origin memberships. With changing subgroup membership this differs from equal-origin weighting; the exact table also includes origin-balanced MAE.')
 if n['Series']<100:st.info('Small cohort: descriptive estimates only. Do not infer stable subgroup superiority.')
 model_chart(summary,'explorer_mae');st.dataframe(summary,hide_index=True,width='stretch',column_config={'MAE':st.column_config.NumberColumn(format='%.6f'),'Bias':st.column_config.NumberColumn(format='%.6f'),'Origin-balanced MAE':st.column_config.NumberColumn(format='%.6f')})
 st.write(f"M3 − M1: **{pair['delta']:+.6f} units/day**. Series-origin memberships: {pair['improved']:,} improved, {pair['worsened']:,} worsened, {pair['tied']:,} tied.")
 vals=summary.set_index('model').MAE
 st.info('M3 has lower observed MAE than MA28 in this selection. This exploratory pattern is not independent evidence of general superiority.' if vals.M3<vals.MA28 else 'MA28 has equal or lower observed MAE than M3 in this selection. A small M3–M1 gain does not imply baseline superiority.')
 byorigin=d.groupby('origin',as_index=False)[['M1_mae','M3_mae','MA28_mae']].mean();st.dataframe(byorigin,hide_index=True,width='stretch')
 download(summary,['losses','memberships'],'Export selected model comparison','explorer_export',scope=json.dumps(filters,sort_keys=True),scope_origins=sorted(d.origin.unique().tolist()))
 download(byorigin,['losses','memberships'],'Export selected origin metrics','explorer_origins_export',scope=json.dumps(filters,sort_keys=True),scope_origins=sorted(d.origin.unique().tolist()))

def evidence():
 st.header('Know what the evidence can—and cannot—support')
 st.success('Final holdout locked · not evaluated' if not table('acceptance')['final_holdout_used'] else 'Holdout status requires review')
 st.subheader('Research boundaries')
 st.markdown('''- Recorded sales are not necessarily unconstrained demand. Historical inventory and stockout status are unavailable.
- Future price-publication timing is uncertain. The primary experiment excludes ABC, revenue and price-dependent features.
- The matched comparison is **XYZ-only under a streamed-ridge design**, not the full revenue-based ABC–XYZ framework.
- Only the accepted development origins are evaluated. Candidate selection and evaluation use the same blocks; temporal generalization remains uncertain.
- No inventory cost savings, prevented stockouts or service-level gains are established.''')
 k=data.kpis();start,end=data.dates();st.write(f"**Development coverage:** {k['series']:,} series · {k['origins']} origins · {k['horizon']}-day horizon. Targets: {start} to {end}.")
 with st.expander('Models, metrics and feature availability',expanded=True):
  st.markdown('**MA28:** trailing 28-day mean. **M1:** base lag/calendar/static features plus historical continuous demand characteristics. **M3:** M1 plus past-only XYZ indicators. All learned-model outputs use the accepted nonnegative clipping rule.\n\n**MAE:** mean absolute unit error; lower is better. **Bias:** prediction minus actual; negative means underforecasting. **Paired delta:** M3 minus M1. Overview/benchmark weights are equal across series, horizons and origins; explorer weights selected memberships and discloses origin-balanced sensitivity.\n\nAll forecasting features and eligibility are computed using available history only. Retrospective Phase 1.6 diagnostics never enter forecast queries.')
 with st.expander('Provenance and independent checks'):
  st.write('Authoritative source: authentic M5 validation data retained in the accepted repository. The dashboard reads protected analytical artifacts only; it has no raw-data or training interface.')
  rec=json.loads((data.ROOT/data.D/'reconciliation.json').read_text());st.json(rec)
  st.code((data.ROOT/data.D/'full-regression.txt').read_text(),language='text')
  st.caption('This is accepted D1 test evidence; E1-specific QA is recorded separately in the E1 review directory.')
 mapping=data.ROOT/'reports/week2/checkpoint-e1-v1/EVIDENCE_MAPPING.csv'
 if mapping.exists():
  m=pd.read_csv(mapping);st.subheader('Dashboard claim-to-source map');st.dataframe(m,hide_index=True,width='stretch');st.download_button('Download evidence mapping',m.to_csv(index=False).encode(),'evidence_mapping.csv','text/csv')
 st.caption('Prior Phase 1.6 descriptive summaries include later historical dates. Their scope is explicitly separated from development forecasting; no final-holdout outcomes are exposed as forecast observations.')

def render():
 st.set_page_config(page_title='Supply Chain • Forecast Research',page_icon='📊',layout='wide')
 st.markdown('''<style>
 .stApp{background:#F7F9FC;color:#172B4D} .block-container{max-width:1240px;padding-top:1.5rem;padding-bottom:2rem}
 h1{font-size:1.65rem!important} h2{font-size:1.35rem!important} h3{font-size:1.05rem!important}
 [data-testid="stMetric"]{background:white;border:1px solid #DCE5EF;border-radius:9px;padding:12px}
 [data-testid="stMetricValue"]{font-size:1.65rem} [data-testid="stCaptionContainer"]{color:#465B73!important}
 @media(max-width:640px){.block-container{padding:1rem}.stPlotlyChart{min-width:0} [data-testid="stMetricValue"]{font-size:1.4rem}}
 </style>''',unsafe_allow_html=True)
 st.title('Supply chain forecasting research')
 st.caption('Accepted development evidence · descriptive diagnostics and predictive results kept separate')
 view=st.selectbox('Analytical view',VIEWS,key='forecast_view')
 try:dict(zip(VIEWS,[executive,diagnostics,benchmarking,incremental,explorer,evidence]))[view]()
 except (ValueError,FileNotFoundError,KeyError) as error:
  st.error('Evidence unavailable or inconsistent. No estimate is substituted.');st.caption(str(error))
 st.divider();st.caption('Research use only · final-holdout evaluation not performed · inventory simulation deferred')
