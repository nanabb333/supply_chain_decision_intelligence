"""Five-view portfolio presentation from accepted summaries only."""
import streamlit as st
import plotly.express as px
import pandas as pd
from .portfolio_data import load,inventory,export,forecast_differences,POLICIES,LABELS,METRICS,SOURCES
VIEWS=['Executive Overview','Demand & Data Diagnostics','Forecasting Performance','Inventory Policy Comparison','Robustness & Sensitivity']
COLORS={'MA28':'#176B87','M1':'#C45A16','M3':'#653FA0'}
INV_SCOPE='128 frozen item–store series · origin 1801 · Jan 4–24, 2016 · 21 days'
FINAL_SCOPE='30,490 item–store series · origin 1885 · Mar 28–Apr 24, 2016 · 28 days'

def chart(fig,key,percent=False):
 fig.update_layout(template='plotly_white',font=dict(color='#172B4D',size=13),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=10,r=20,t=20,b=20),legend=dict(orientation='h',y=1.12),height=350)
 fig.update_yaxes(rangemode='tozero');st.plotly_chart(fig,width='stretch',key=key,config={'displaylogo':False})
def bars(d,metric,title,key):
 st.subheader(title)
 chart(px.bar(d,x='model',y=metric,color='model',color_discrete_map=COLORS,category_orders={'model':POLICIES},labels={'model':'Policy',metric:title},hover_data={metric:':.6f'}).update_layout(showlegend=False),key)
def source(name):st.caption('Source: '+SOURCES[name])
def download(d,name,key,scope):st.download_button('Download displayed evidence',export(d,name,scope),key+'.csv','text/csv',key=key)
def policy_control(key):return st.multiselect('Policies',POLICIES,default=POLICIES,key=key)
def reset():
 for k in list(st.session_state):
  if k=='pf_view':st.session_state[k]=VIEWS[0]
  elif k=='pf_dimension':st.session_state[k]='xyz'
  elif k=='pf_family':st.session_state[k]='operating'
  elif k.startswith('pf_choice_'):st.session_state[k]='primary'
  elif k.endswith('_policies') and k.startswith('pf_'):st.session_state[k]=POLICIES.copy()
  elif k.startswith('pf_metric_'):st.session_state[k]='total_cost'
def explain_inventory():
 st.caption('Observed: recorded M5 sales. Predicted: frozen forecasts. Assumed: operating rules and costs. Simulated: inventory, service and lost sales.')
def inventory_panel(d,key):
 if d.empty:st.info('No policies selected. Choose a policy or reset the walkthrough.');return
 metric=st.selectbox('Metric', [k for k in METRICS if k in d.columns],format_func=lambda k:METRICS[k],key='pf_metric_'+key)
 bars(d,metric,METRICS[metric],key+'_bar')
 with st.expander('Exact values and denominators',expanded=False):
  cols=[k for k in ['model','observed_sales_proxy_units','fulfilled_units','lost_units','fill_percent','macro_fill','stockout_series_days','average_on_hand','holding_cost','shortage_cost','ordering_cost','total_cost','initial_inventory','terminal_inventory','terminal_pipeline','day1_order_units'] if k in d.columns]
  st.dataframe(d[cols],hide_index=True,width='stretch')
  st.caption('Pooled fill = fulfilled / recorded sales-proxy units. Mean stock uses all series-days. Macro fill excludes undefined zero-demand series only; all series remain in costs. Costs are hypothetical units, not USD.')

def executive():
 f=load('forecast');p=forecast_differences();base=inventory();init=load('initialization');seg=load('segments');seg=seg[seg.scenario=='baseline']
 st.markdown('### How do forecasting accuracy, replenishment policies and operating assumptions interact?')
 st.write('Forecast accuracy is one input to policy value. The operating rules and opening stock can change cost leadership—even with forecasts held fixed.')
 c=st.columns(3);c[0].metric('Final forecast series',f'{int(f.series.min()):,}');c[1].metric('Inventory pilot series',f'{int(base.series.min()):,}');c[2].metric('Retrospective segmentation series',f'{len(seg):,}')
 c=st.columns(2)
 with c[0]:
  bars(f,'macro_mae','Final-holdout macro MAE · units/day','exec_mae');st.caption(FINAL_SCOPE)
 with c[1]:
  bars(base,'total_cost','Primary hypothetical cost','exec_cost');st.caption(INV_SCOPE)
 change=p.set_index('comparison').loc['M3_minus_M1'];best=f.loc[f.macro_mae.idxmin(),'model'];leaders=init.loc[init.groupby('method').total_cost.idxmin()].set_index('method').model
 st.info(f"{best} has the lowest final macro MAE. Adding XYZ labels reduces M3's MAE versus M1 by {change.reduction_pct:.4f}%. Inventory cost leadership changes from {leaders['primary']} under the primary start to {leaders['symmetric_forecast_common']} under the median-target start.")
 st.warning('These are different evaluation populations and dates—not a matched causal chain from final forecast MAE to pilot inventory cost. No real retailer savings, actual stockout reductions or deployment winner are established.')
 with st.expander('Five-minute walkthrough and evidence boundaries'):
  st.write('1. Review the scope and result. 2. Inspect demand and sample limitations. 3. Compare final forecast errors. 4. Examine the primary service–cost tradeoff. 5. Test the accepted sensitivity views.')
  st.write('Historical sales are observed; forecasts are predictions; operating parameters are hypothetical; inventory outcomes are simulated. Empirical research is frozen. Final holdout had prior descriptive exposure; it is not a pristine blind test.')
 source('forecast');source('initialization')

def diagnostics():
 st.header('Demand structure and scope limitations')
 st.caption('Past-only pilot comparison: d1–d1801. The 128-series engineering cohort is distinct from the earlier 90-series descriptive sample.')
 c=load('categories');n=load('numeric');dimension=st.selectbox('Population comparison',['xyz','store_id','cat_id','dept_id','intermittent','recent28_all_zero'],key='pf_dimension',format_func=lambda x:{'xyz':'XYZ composition','store_id':'Stores','cat_id':'Categories','dept_id':'Departments','intermittent':'ADI intermittency','recent28_all_zero':'Zero sales in preceding 28 days'}[x])
 q=c[c.dimension==dimension].copy();q=q.melt(id_vars=['level'],value_vars=['sample_share','population_share'],var_name='Scope',value_name='Share');q['Share (%)']=q.Share*100;q.Scope=q.Scope.map({'sample_share':'128-series pilot','population_share':'30,490-series population'})
 chart(px.bar(q,x='level',y='Share (%)',color='Scope',barmode='group',color_discrete_sequence=['#176B87','#75869A'],labels={'level':'Historical group'}),'diag_composition')
 with st.expander('Historical distributions and coverage'):
  st.dataframe(n[['metric','sample_mean','population_mean','sample_median','population_median','ks_distance']],hide_index=True,width='stretch');st.caption('ECDF distances are descriptive, not random-sample significance or equivalence tests. Undefined ADI/CV remain excluded only from their numeric distributions.')
 st.warning('Store imbalance and missing rare classes remain. Hash selection avoids outcome-based selection but does not establish representativeness. Zero sales are not proof of stockouts or absent demand; lifecycle boundaries can inflate CV and ADI.')
 st.subheader('Retrospective ABC–XYZ segmentation')
 seg=load('segments');seg=seg[seg.scenario=='baseline'];counts=seg.groupby('segment').size().reset_index(name='Series')
 st.caption(f"{len(seg)} selected series · {seg.window_start.min()} to {seg.window_end.max()} · retrospective, not forecast inputs")
 chart(px.bar(counts,x='segment',y='Series',color_discrete_sequence=['#176B87'],labels={'segment':'ABC–XYZ segment'}),'segments')
 w=load('windows').set_index('scenario').loc['last365']
 st.write(f"The last-365-day window changed {int(w.changed)} ABC–XYZ labels, including {int(w.xyz_changed)} XYZ-only changes. These are different counts.")
 st.caption('ABC ranks estimated units × selling price, not units alone. XYZ uses daily CV with ADI intermittency override. Both are heuristic descriptions, not optimal inventory policies.')
 source('segments');source('categories');download(c[c.dimension==dimension],'categories','diag_export','Historical d1–d1801;sample128/population30490;'+dimension)

def forecasting():
 st.header('Forecast accuracy is not inventory value');st.caption(FINAL_SCOPE)
 policies=policy_control('pf_forecast_policies');f=load('forecast');f=f[f.model.isin(policies)]
 if f.empty:st.info('No policies selected. Choose a policy or reset.');return
 bars(f,'macro_mae','Macro MAE · units per series-day (lower is better)','forecast_mae')
 st.dataframe(f[['model','macro_mae','series','target_observations']],hide_index=True,width='stretch',column_config={'macro_mae':st.column_config.NumberColumn(format='%.9f')})
 p=forecast_differences(policies);st.subheader('Matched absolute and relative differences');st.dataframe(p[['comparison','delta_mae','relative_delta_pct']],hide_index=True,width='stretch')
 st.caption('Delta = left model − reference. Relative delta = 100 × delta / reference MAE; negative favors the left. Macro MAE weights each series equally over its 28 daily errors.')
 st.info('M1 uses continuous demand features; M3 adds past-only XYZ labels. This tests XYZ incremental predictive value, not revenue-based ABC–XYZ value. Forecast error does not encode holding costs, shortage penalties or initial stock.')
 st.warning('Final evaluation is frozen and had prior descriptive exposure. Inventory replay uses origin 1801, 128 series and a 21-day window; do not attribute its cost differences to these final-holdout MAE differences.')
 source('forecast');source('paired');download(f,'forecast','forecast_export',FINAL_SCOPE)

def inventory_view():
 st.header('Primary scenario: service and cost trade off');st.caption(INV_SCOPE);explain_inventory()
 st.caption('Primary assumptions: lead 3 days · review every 7 days · buffer 3 days · holding 1 · shortage 10 · ordering 0 · lost sales · unlimited supply · empty pipeline')
 policies=policy_control('pf_inventory_policies');d=inventory(policies=policies);inventory_panel(d,'primary')
 if d.empty:return
 chart(px.scatter(d,x='total_cost',y='fill_percent',color='model',symbol='model',color_discrete_map=COLORS,hover_data=['lost_units','average_on_hand'],labels={'total_cost':'Hypothetical cost units','fill_percent':'Pooled simulated fill (%)'}).update_traces(marker_size=14),'tradeoff')
 st.caption('Tradeoff chart uses a zero-based fill axis; exact differences are in the table. Initial stock is MA28-based, with no purchase cost; terminal stock has no salvage value.')
 source('primary');download(d,'operating','inventory_export',INV_SCOPE+';primary;matched policies')

def robustness():
 st.header('Cost leadership depends on the assumptions');st.caption(INV_SCOPE);explain_inventory()
 family=st.radio('Evidence family',['operating','initialization'],format_func=lambda x:'D1 · nine operating scenarios' if x=='operating' else 'D3 · three initialization methods',key='pf_family',horizontal=True)
 raw=load(family);col='scenario' if family=='operating' else'method';options=raw[col].drop_duplicates().tolist()
 choice=st.selectbox('Scenario' if family=='operating' else'Initialization method',options,format_func=lambda x:LABELS[x],key='pf_choice_'+family)
 policies=policy_control('pf_robust_policies');d=inventory(family,choice,policies);inventory_panel(d,'robust_'+family)
 st.caption('These controls select existing evidence only. D1 varies operating scenarios with its original initialization rule. D3 varies opening-stock basis under the primary scenario only. Untested cross-combinations are unavailable.')
 st.subheader('Cost leadership across the selected evidence family')
 if family=='operating':leaders=raw.loc[raw.groupby('scenario').total_hypothetical_cost.idxmin(),['scenario','model','total_hypothetical_cost']].rename(columns={'scenario':'Configuration','model':'Lowest-cost policy','total_hypothetical_cost':'Hypothetical cost'})
 else:leaders=raw.loc[raw.groupby('method').total_cost.idxmin(),['method','model','total_cost']].rename(columns={'method':'Configuration','model':'Lowest-cost policy','total_cost':'Hypothetical cost'})
 leaders.Configuration=leaders.Configuration.map(LABELS);st.dataframe(leaders,hide_index=True,width='stretch');st.caption('Leadership table always compares all three policies; the chart above follows your policy selection.')
 if family=='initialization':
  st.subheader('Day-one ordering asymmetry');bars(d,'day1_order_units','Day-one orders · units','day1') if not d.empty else None
  st.info('Identical opening stocks do not prove neutrality. The median of related model targets can align with M1/M3; long-history initialization may be stale. Subsequent policy rules remain unchanged.')
 else:st.info('Shortage penalty changes valuation, not physical decisions. Lead/review/buffer changes also change opening stock mechanically, so these are not isolated effects at fixed initial inventory.')
 st.warning('D1 and D3 share the accepted primary baseline. Their view counts must not be added as independent experiments. One cohort and one episode cannot establish a universal winner or population inventory outcomes.')
 source(family);download(d,family,'robust_export',INV_SCOPE+';'+family+';'+choice)

def render():
 st.set_page_config(page_title='Supply chain decision intelligence',layout='wide')
 st.markdown('<style>.block-container{max-width:1240px;padding-top:1.2rem}h1{font-size:1.9rem!important}h2{font-size:1.45rem!important}h3{font-size:1.15rem!important}[data-testid="stMetricValue"]{font-size:1.8rem}@media(max-width:640px){.block-container{padding:1rem}.stMain h1{font-size:1.5rem!important}}</style>',unsafe_allow_html=True)
 st.title('Supply chain decision intelligence');st.caption('Frozen research portfolio · observations → forecasts → assumptions → simulated outcomes')
 view=st.selectbox('Portfolio view',VIEWS,key='pf_view');st.button('Reset walkthrough',on_click=reset)
 dict(zip(VIEWS,[executive,diagnostics,forecasting,inventory_view,robustness]))[view]()
 st.divider();st.caption('Research demonstration only · no observed retailer savings or deployment recommendation · no new simulations or model fitting')
