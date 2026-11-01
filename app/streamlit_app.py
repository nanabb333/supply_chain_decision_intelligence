from pathlib import Path
import os
import sys
from numbers import Real

# Resolve project source consistently for Streamlit script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supply_chain.services import connect,load_sales,get_sales_kpis,daily_series
from supply_chain.segmentation import classify

px.defaults.color_discrete_sequence=['#087F8C','#2563A6','#B87518']
ROOT=Path(__file__).resolve().parents[1]
real_db = ROOT/'data/processed/m5-real-v1/analytics.duckdb'
default_db = real_db if real_db.exists() and (ROOT/'reports/phase1_5/validation.json').exists() else ROOT/'data/processed/analytics.duckdb'
DB=Path(os.environ.get('SUPPLY_CHAIN_DB',default_db))
st.set_page_config(page_title='Supply Chain | Sales intelligence',page_icon='📦',layout='wide')
st.title('Sales performance')

_chart_index=0
def render_chart(figure, **kwargs):
    global _chart_index
    _chart_index += 1
    figure.update_layout(template='plotly_white', font=dict(color='#172B4D', size=12),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', height=380,
        margin=dict(l=55,r=65,t=35,b=75), colorway=['#087F8C','#2563A6','#B87518'],
        legend=dict(orientation='h',y=1.16,x=0))
    for trace in figure.data:
        if trace.type=='bar':
            trace.width=0.4 if len(trace.x)==1 else 0.55
            if len(trace.x)==1:
                center=float(trace.x[0]) if isinstance(trace.x[0],Real) else 0
                # Assign the conditional result: Streamlit magic displays bare expressions.
                figure.update_xaxes(range=[center-2,center+2])
        if trace.type=='scatter' and len(trace.x)==1: trace.mode='lines+markers'
    st.plotly_chart(figure, width='stretch', theme=None, key=f'analysis_chart_{_chart_index}_{store}_{category}_{item}_{segment}_{start}_{end}')

if not DB.exists():
    st.info('No analytical database yet. Follow README to ingest M5 or explicitly generate the synthetic test fixture.')
    st.stop()
with connect(DB) as con:
    metadata=con.execute('SELECT * FROM run_metadata').df().iloc[0]
    dimensions=con.execute('SELECT * FROM dim_series ORDER BY store_id,item_id').df()
    saved_segments=con.execute('SELECT * FROM sku_segments').df()
    low,high=con.execute('SELECT min(date),max(date) FROM daily_sales').fetchone()
if metadata.source_kind=='synthetic_fixture':
    st.warning('SYNTHETIC TEST FIXTURE — these are generated observations, not M5/Walmart sales. Business findings on real data remain pending.')
elif metadata.source_kind=='M5_official_archive': st.info(f'Official M5 organizer archive | {len(dimensions)}-series research subset. Observed sales are not unconstrained demand.')
else: st.caption('Source: user-supplied M5 validation data • Historical observed sales, not unconstrained customer demand')
st.caption(f'{len(dimensions):,} item–store series • {low:%b %d, %Y} – {high:%b %d, %Y} • USD price-based revenue estimate')
with st.sidebar:
    st.header('Explore sales')
    store=st.selectbox('Store',['All']+sorted(dimensions.store_id.unique()))
    category=st.selectbox('Category',['All']+sorted(dimensions.cat_id.unique()))
    available=dimensions.copy()
    if store!='All': available=available[available.store_id==store]
    if category!='All': available=available[available.cat_id==category]
    item=st.selectbox('Product',['All']+sorted(available.item_id.unique()))
    segment=st.selectbox('ABC–XYZ segment',['All']+sorted(saved_segments.segment.unique()))
    dates=st.date_input('Date range',value=(low,high),min_value=low,max_value=high)
    st.caption('Segments use the complete ingested history and stay fixed when date filters change. U means unclassifiable.')
if len(dates)!=2:
    st.info('Choose both a start and end date.'); st.stop()
start,end=dates
filters=dict(stores=None if store=='All' else [store],categories=None if category=='All' else [category],items=None if item=='All' else [item],segments=None if segment=='All' else [segment])
frame=load_sales(DB,**filters,start=start,end=end)
if frame.empty:
    st.info('No records match these filters. Choose All or widen the date range.'); st.stop()
prior_end=pd.Timestamp(start)-pd.Timedelta(days=1)
prior_start=prior_end-pd.Timedelta(days=(end-start).days)
previous=load_sales(DB,**filters,start=prior_start.date(),end=prior_end.date()) if prior_start.date()>=low else None
kpi=get_sales_kpis(frame,previous)

def fmt(value,kind='number'):
    if value is None or pd.isna(value): return 'N/A'
    return f'${value:,.2f}' if kind=='money' else (f'{value:.1%}' if kind=='percent' else f'{value:,.0f}')

cards=st.columns([1,1.6,1,1])
cards[0].metric('Observed units sold',fmt(kpi['units']),border=True,help='Sum of recorded units. Missing observations excluded, not filled with zero.')
cards[1].metric('Known-price revenue',fmt(kpi['revenue'],'money'),border=True,help='Units × observed weekly price. Positive sales with missing prices are excluded; zero sales contributes zero.')
cards[2].metric('Average selling price',fmt(kpi['asp'],'money'),border=True,help='Known-price revenue / units with observed prices. Unit-weighted; not a simple price average.')
cards[3].metric('Unit sales growth',fmt(kpi['growth'],'percent'),border=True,help='Versus immediately preceding equal-length period, same filters. N/A for insufficient history, missing sales, or zero baseline.')
if kpi['missing_sales'] or kpi['unpriced_positive_days']:
    st.warning(f"Partial coverage: {kpi['missing_sales']:,} missing sales observations; {kpi['unpriced_positive_days']:,} positive-sales days without prices. Revenue is partial; trend gaps mark incomplete daily totals.")
trend=daily_series(frame)
fig=go.Figure()
fig.add_scatter(x=trend.date,y=trend.units,name='Units',line_color='#177e89',connectgaps=False)
fig.update_layout(height=310,margin=dict(l=0,r=0,t=20,b=0),yaxis_title='Units sold',template='plotly_white')
st.subheader('Historical sales')
render_chart(fig,width='stretch')
with st.expander('Revenue and observed selling prices'):
    render_chart(px.line(trend,x='date',y='revenue',labels={'revenue':'Known-price revenue (USD)'},template='plotly_white'),width='stretch')
    priced=frame[frame.sell_price.notna() & frame.units.notna()].groupby('date')[['units','revenue']].sum(min_count=1)
    prices=(priced.revenue / priced.units.where(priced.units>0)).rename('Selling price (USD/unit)').reindex(pd.DatetimeIndex(frame.date.unique(),name='date').sort_values()).reset_index()
    render_chart(px.line(prices,x='date',y='Selling price (USD/unit)',template='plotly_white'),width='stretch')
left,right=st.columns(2)
with left:
    st.subheader('Category contribution')
    cats=frame.groupby('cat_id').revenue.sum(min_count=1).reset_index()
    total=cats.revenue.sum(min_count=1)
    if pd.notna(total) and total>0:
        cats['Share of known-price revenue']=cats.revenue/total
        render_chart(px.bar(cats,x='cat_id',y='Share of known-price revenue',hover_data=['revenue'],labels={'cat_id':'Category'},template='plotly_white').update_yaxes(tickformat='.0%'),width='stretch')
    else: st.info('No positive known-price revenue in this selection.')
with right:
    st.subheader('Weekly sales pattern')
    trend['Weekday']=pd.to_datetime(trend.date).dt.day_name()
    weekly=trend.groupby('Weekday').units.mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()
    render_chart(px.bar(weekly,x='Weekday',y='units',labels={'units':'Mean daily portfolio units'},template='plotly_white'),width='stretch')
st.subheader('Monthly sales pattern')
trend['Month']=pd.to_datetime(trend.date).dt.month
monthly=trend.groupby('Month').units.mean().reset_index()
render_chart(px.line(monthly,x='Month',y='units',markers=True,labels={'units':'Mean daily portfolio units'},template='plotly_white'),width='stretch')
st.caption(f"Daily portfolio sales volatility (population standard deviation): {fmt(kpi['volatility'])} units. Weekly/monthly patterns are descriptive and may reflect trend, assortment, or availability; incomplete days are excluded.")
rank=frame.groupby(['item_id','store_id']).agg(units=('units',lambda x:x.sum(min_count=1)),known_revenue=('revenue',lambda x:x.sum(min_count=1)),missing_days=('missing_sales','sum')).reset_index().sort_values(['units','item_id','store_id'],ascending=[False,True,True],na_position='last')
l,r=st.columns(2)
with l: st.subheader('Top selling SKUs'); st.dataframe(rank.head(10),hide_index=True,width='stretch')
with r: st.subheader('Bottom selling SKUs'); st.dataframe(rank.sort_values(['units','item_id','store_id'],na_position='last').head(10),hide_index=True,width='stretch')
st.divider()
st.subheader('ABC–XYZ segmentation')
st.caption('Full-history classification, scoped to selected store/category/product/segment. A/B/C: cumulative eligible revenue; X/Y/Z: daily sales variability. The date filter affects sales views above, not saved segments.')
keys=frame[['item_id','store_id']].drop_duplicates()
scope=saved_segments.merge(keys,on=['item_id','store_id'])
matrix=pd.crosstab(scope.abc,scope.xyz).reindex(index=['A','B','C','U'],columns=['X','Y','Z','U'],fill_value=0).fillna(0)
render_chart(px.imshow(matrix,text_auto=True,labels={'x':'Variability class','y':'Revenue class','color':'Series'},color_continuous_scale='Teal',aspect='auto'),width='stretch')
pareto=scope[scope.abc!='U'].sort_values(['revenue','item_id','store_id'],ascending=[False,True,True]).copy()
if not pareto.empty:
    pareto['SKU/store']=pareto.item_id+' / '+pareto.store_id
    pareto['Cumulative scoped share']=pareto.revenue.cumsum()/pareto.revenue.sum()
    pareto['Revenue rank']=range(1,len(pareto)+1)
    chart=go.Figure(go.Bar(x=pareto['Revenue rank'],y=pareto.revenue,customdata=pareto['SKU/store'],hovertemplate='%{customdata}<br>Revenue: $%{y:,.2f}<extra></extra>',name='Eligible revenue'))
    chart.add_scatter(x=pareto['Revenue rank'],y=pareto['Cumulative scoped share'],name='Cumulative share',yaxis='y2',line=dict(color='#B87518'))
    chart.update_layout(xaxis_title='SKU rank by eligible revenue (hover for SKU/store)',yaxis_title='Known-price revenue (USD)',yaxis2=dict(overlaying='y',side='right',tickformat='.0%',range=[0,1.05],tickvals=[0,.25,.5,.75,1],title='Cumulative share'),template='plotly_white',height=350)
    render_chart(chart,width='stretch')
st.dataframe(scope[['item_id','store_id','segment','revenue','cv','adi','abc_reason','xyz_reason']],hide_index=True,width='stretch')
st.download_button('Download scoped segments',scope.to_csv(index=False),'sku_segments.csv','text/csv')
with st.expander('Threshold sensitivity and operational interpretation'):
    st.markdown('A/B revenue cutoffs default to 80%/95%; X/Y daily CV cutoffs to 0.5/1.0. Average demand interval (ADI) ≥1.32 forces Z. Boundary-crossing SKU stays in the earlier ABC band. Missing sales or missing positive-sales prices makes ABC unclassifiable. All-zero sales is unclassifiable. These are illustrative policies, not universal inventory rules.')
    a=st.slider('A revenue cutoff',.50,.90,.80,.05)
    b=st.slider('B revenue cutoff',.91,.99,.95,.01)
    x=st.slider('X CV cutoff',.1,.8,.5,.1)
    y=st.slider('Y CV cutoff',.9,2.,1.,.1)
    changed=classify(saved_segments,a=a,b=b,x=x,y=y)
    comparison=saved_segments[['item_id','store_id','segment']].merge(changed[['item_id','store_id','segment']],on=['item_id','store_id'],suffixes=('_saved','_scenario'))
    st.write(f'{(comparison.segment_saved != comparison.segment_scenario).sum()} of {len(comparison)} portfolio series change segment. This preview does not overwrite saved segments.')
    st.dataframe(comparison,hide_index=True)
    st.markdown('**AX/AY/AZ:** prioritize high-revenue monitoring; Z requires careful intermittent-sales evaluation. **BX/BY/BZ:** moderate revenue priority, increasing variability across X→Z. **CX/CY/CZ:** lower revenue priority; consider simpler review policies, with extra caution for Z. **U:** investigate coverage before using classifications. No segment establishes a reorder quantity or observed stockout risk.')
with st.expander('Definitions and limitations'):
    st.markdown((ROOT/'docs/KPIS.md').read_text())
    st.markdown('M5 provides sales and weekly selling prices, not inventory balances or true unmet demand. Inventory turnover, days of supply, fill rate, and stockout rate cannot be reported as observed metrics. Revenue is an estimate using weekly prices, not audited transaction receipts. Selection is descriptive, not a representative population sample. Segments must be recomputed at historical cutoffs before future forecasting evaluation.')
