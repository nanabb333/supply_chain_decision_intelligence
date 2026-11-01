"""Read-only Streamlit diagnostics, scoped consistently to saved C/D evidence."""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from .diagnostic_quality import scope,quality,sensitivity,within_z,zero_bands,quantiles,window_summary,transition_matrix,export_csv,HORIZON,LIMITS
COLORS={'Selected sample':'#087F8C','M5 population':'#B15A14'}

@st.cache_data
def load_evidence(directory):
    return pd.read_parquet(directory/'profile.parquet'),pd.read_parquet(directory/'transition_explanations.parquet')

def chart(fig,key,percent=False):
    fig.update_layout(template='plotly_white',paper_bgcolor='#FFFFFF',plot_bgcolor='#FFFFFF',font=dict(color='#172B4D',size=12),height=430,margin=dict(l=45,r=20,t=20,b=105),legend=dict(orientation='h',x=0,y=-.25,font=dict(size=11)))
    if key=='e_thresholds':fig.update_layout(height=490,margin=dict(l=45,r=20,t=20,b=155),legend=dict(y=-.42),xaxis_title=None)
    if percent:fig.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig,width='stretch',theme=None,key=key,config={'displayModeBar':False})

def reset():
    for key in ['e_store','e_category','e_department','e_item']:
        st.session_state[key]='All'

def render(root):
    st.set_page_config(page_title='Classification diagnostics | Supply Chain',layout='wide')
    st.title('Classification Robustness & Diagnostic Quality')
    st.caption('Authentic M5 validation data · '+HORIZON+' · Accepted XYZ thresholds unchanged')
    directory=root/'reports/phase1_6/checkpoint-e-v1/diagnostics'
    if not (directory/'manifest.json').exists():st.error('Verified Checkpoint E diagnostic evidence is unavailable.');st.stop()
    population,transitions=load_evidence(directory)
    with st.sidebar:
        st.header('Diagnostic scope')
        store=st.selectbox('Diagnostic store',['All']+sorted(population.store_id.unique()),key='e_store')
        category=st.selectbox('Diagnostic category',['All']+sorted(population.cat_id.unique()),key='e_category')
        department=st.selectbox('Diagnostic department',['All']+sorted(population.dept_id.unique()),key='e_department')
        item=st.selectbox('Diagnostic product',['All']+sorted(population.item_id.unique()),key='e_item')
        st.button('Reset diagnostic filters',on_click=reset)
        st.caption('Filters apply to both cohorts and every diagnostic tab/export. Windows are fixed evidence scenarios; sales-page date/segment filters do not apply here.')
    p=scope(population,store,category,department,item)
    t=transitions.merge(p[['item_id','store_id']],on=['item_id','store_id'],how='inner',validate='many_to_one')
    desc=f'Store={store}; category={category}; department={department}; product={item}; baseline={HORIZON}'
    st.caption(f'Current scope: {len(p[p.selected]):,} selected-sample series / {len(p):,} population series. Population includes sample; this is not a random or representative sample.')
    if p.empty:st.info('No series match these diagnostic filters. Reset filters or select a compatible category and department.');st.stop()
    st.info('These diagnostics describe classification stability and discrimination. Balanced classes are not inherently better; operational usefulness has not been demonstrated.')
    overview,rules,changes=st.tabs(['Sample vs population','Rule diagnostics','Explain a change'])
    q,d=quality(p)
    with overview:
        st.subheader('Class distribution')
        st.caption('Full-history labels. Bars use all scoped series, including U if ineligible. The concentration measures below use eligible X/Y/Z only.')
        chart(px.bar(d,x='xyz',y='share_all',color='cohort',barmode='group',category_orders={'xyz':['X','Y','Z','U']},color_discrete_map=COLORS,hover_data=['series','total_series'],labels={'xyz':'Class','share_all':'Share of all series','cohort':'Cohort'}),'e_distribution',True)
        st.subheader('Concentration and eligibility')
        view=q[['cohort','series','eligible','excluded','largest_class_share','hhi','normalized_entropy','effective_classes']].rename(columns={'cohort':'Cohort','series':'Series','eligible':'Eligible','excluded':'Excluded','largest_class_share':'Largest class share','hhi':'HHI','normalized_entropy':'Normalized entropy','effective_classes':'Effective classes'})
        st.dataframe(view,hide_index=True,width='stretch',column_config={'Largest class share':st.column_config.NumberColumn(format='%.4f'),'HHI':st.column_config.NumberColumn(format='%.4f'),'Normalized entropy':st.column_config.NumberColumn(format='%.4f'),'Effective classes':st.column_config.NumberColumn(format='%.2f')})
        st.caption('HHI = sum of squared class shares (1/3–1). Normalized entropy = −Σp ln(p) / ln(3) (0–1). Effective classes = 1 / HHI. Higher concentration means less label differentiation, not necessarily a worse policy. Fractions use 0–1.')
        st.subheader('Observed sales comparison')
        st.dataframe(q[['cohort','median_units','mean_positive_frequency','mean_zero_frequency']].rename(columns={'cohort':'Cohort','median_units':'Median total units / series','mean_positive_frequency':'Mean positive-sales frequency','mean_zero_frequency':'Mean zero-sales frequency'}),hide_index=True,width='stretch')
        st.caption('Frequencies are fractions of observed days, averaged equally across series. Sample revenue concentration and weekend effects must not be generalized to the population.')
        st.subheader('Internal zero-sales run lengths')
        bands=zero_bands(p)
        measure=st.radio('Run comparison',['Share of internal zero days','Share of internal runs'],horizontal=True,key='e_runs')
        field='zero_day_share' if measure=='Share of internal zero days' else 'run_share'
        chart(px.bar(bands,x='band',y=field,color='cohort',barmode='group',color_discrete_map=COLORS,hover_data=['runs','zero_days','series_with_run','series'],labels={'band':'Consecutive zero days',field:measure,'cohort':'Cohort'}),'e_runs_chart',True)
        st.caption('Internal runs lie strictly between first and last positive sales; leading/trailing zeros are excluded. Bands are descriptive. Zero sales do not prove stockouts or absent demand.')
        st.download_button('Download scoped diagnostic profiles',export_csv(p,desc+'; population rows with selected flag'),'diagnostic_profiles.csv','text/csv',key='e_profile_export')
        st.download_button('Download concentration summary',export_csv(q,desc),'diagnostic_quality.csv','text/csv',key='e_quality_export')
        st.download_button('Download scoped class distribution',export_csv(d,desc),'class_distribution.csv','text/csv',key='e_class_export')
        st.download_button('Download scoped zero-run table',export_csv(bands,desc),'zero_run_bands.csv','text/csv',key='e_runs_export')
    with rules:
        st.subheader('How much does the ADI rule add?')
        st.caption('Triggered: ADI ≥ 1.32. Effective: the trigger changes a non-Z CV-only label to Z. Rates use eligible series. The overlapping rule triggers must not be added together.')
        st.dataframe(q[['cohort','eligible','adi_trigger_count','adi_trigger_rate','adi_effective_count','adi_effective_rate']].rename(columns={'cohort':'Cohort','eligible':'Eligible','adi_trigger_count':'ADI triggered','adi_trigger_rate':'Trigger rate','adi_effective_count':'ADI effective','adi_effective_rate':'Effective rate'}),hide_index=True,width='stretch')
        st.subheader('CV and ADI distributions')
        curves=quantiles(p)
        st.caption('Quantile profiles use 101 quantiles across all eligible scoped series; log value axes reveal skew without dropping the tails. CV = daily population SD / mean. ADI = observed days / positive-sales days. All observed zeros are included.')
        for metric in ['cv','adi']:
            chart(px.line(curves[curves.metric==metric],x='value',y='quantile',color='cohort',log_x=True,color_discrete_map=COLORS,labels={'value':metric.upper()+' (log scale)','quantile':'Quantile','cohort':'Cohort'}).update_xaxes(tickmode='array',tickvals=[.1,1,10,100],ticktext=['0.1','1','10','100']),'e_'+metric,True)
        st.subheader('Threshold sensitivity')
        sens=sensitivity(p)
        chart(px.bar(sens,x='variant',y='Z_share',color='cohort',barmode='group',color_discrete_map=COLORS,hover_data=['changed','eligible'],labels={'variant':'Rule scenario','Z_share':'Z share of eligible series','cohort':'Cohort'}).update_xaxes(tickangle=-30),'e_thresholds',True)
        st.caption('Fixed sensitivity variants reuse accepted CV/ADI. “CV only” removes the ADI override as a diagnostic ablation, not a recommended replacement. No thresholds were optimized for balance, forecast accuracy, or cost.')
        st.dataframe(sens[['cohort','variant','eligible','X','Y','Z','U','changed']],hide_index=True,width='stretch')
        st.subheader('Within-Z heterogeneity')
        z=within_z(p)
        st.dataframe(z[['cohort','rule_group','series','share_Z','cv_median','adi_median','positive_frequency_median','longest_internal_zero_run_median']].rename(columns={'cohort':'Cohort','rule_group':'Rule group','series':'Series','share_Z':'Share of Z','cv_median':'Median CV','adi_median':'Median ADI','positive_frequency_median':'Median positive frequency','longest_internal_zero_run_median':'Median longest internal run (days)'}),hide_index=True,width='stretch')
        st.caption('Groups are disjoint: ADI only, CV only, or both rules. Medians summarize within-group variation; full p10/median/p90 diagnostics are downloadable. Z is not a homogeneous operational-risk category.')
        st.download_button('Download threshold diagnostics',export_csv(sens,desc),'threshold_diagnostics.csv','text/csv',key='e_threshold_export')
        st.download_button('Download within-Z diagnostics',export_csv(z,desc),'within_z.csv','text/csv',key='e_z_export')
    with changes:
        st.subheader('Observation-window stability — selected sample only')
        st.caption('Population observation-window transitions are not available in accepted evidence. Filters subset saved comparisons; revenue ranks and ABC shares retain the original 90-series portfolio context. Labels are not recomputed within filters.')
        if t.empty:
            st.info('No selected-sample series in this population scope. Population diagnostics remain available; sample transitions cannot be inferred.')
        else:
            windows=window_summary(t);st.dataframe(windows.drop(columns='limitations'),hide_index=True,width='stretch')
            scenario=st.selectbox('Alternative scenario',[x for x in t.scenario.unique() if x!='baseline'],index=list(x for x in t.scenario.unique() if x!='baseline').index('last365'),key='e_scenario')
            alt=t[t.scenario==scenario]
            st.write(f'{int(alt.segment_changed.sum())} of {len(alt)} scoped sample series change segment in {scenario}.')
            st.subheader('Baseline versus alternative segments')
            chart(px.imshow(transition_matrix(alt),text_auto=True,color_continuous_scale='Teal',aspect='auto',labels={'x':'Alternative segment','y':'Baseline segment','color':'Series'}),'e_matrix')
            changed_only=st.checkbox('Show changed series only',value=True,key='e_changed')
            choices=alt[alt.segment_changed] if changed_only else alt
            if choices.empty:st.info('No changed series for this scenario and scope. Turn off the changed-only filter to inspect stable labels.')
            else:
                options=(choices.item_id+' / '+choices.store_id).tolist()
                selected=st.selectbox('Item–store explanation',options,key='e_explain_item')
                row=choices.iloc[options.index(selected)]
                st.subheader(f'{row.segment_baseline} → {row.segment}')
                st.write(row.explanation_categories);st.write(row.explanation)
                pairs={'Segment':(row.segment_baseline,row.segment),'CV':(row.cv_baseline,row.cv),'ADI':(row.adi_baseline,row.adi),'CV-only class':(row.cv_class_baseline,row.cv_class),'Revenue rank (original portfolio)':(row.revenue_rank_baseline,row.revenue_rank),'Observed days':(row.observed_days_baseline,row.observed_days),'Window start':(str(row.window_start_baseline.date()),str(row.window_start.date())),'Window end':(str(row.window_end_baseline.date()),str(row.window_end.date())),'ABC eligible':(row.abc_eligible_baseline,row.abc_eligible),'XYZ eligible':(row.xyz_eligible_baseline,row.xyz_eligible),'ADI triggered':(row.adi_rule_triggered_baseline,row.adi_rule_triggered),'ADI effective':(row.adi_override_effective_baseline,row.adi_override_effective)}
                details=pd.DataFrame([{'Measure':k,'Baseline':f'{v[0]:.6f}' if k in ['CV','ADI'] else str(v[0]),'Alternative':f'{v[1]:.6f}' if k in ['CV','ADI'] else str(v[1])} for k,v in pairs.items()]);st.dataframe(details,hide_index=True,width='stretch')
                st.write(f'Alternative minus baseline: CV {row.cv_delta:+.4f}; ADI {row.adi_delta:+.4f}; revenue rank {row.revenue_rank_delta:+g}; observed days {row.observed_days_delta:+g}.')
                st.write(f'Baseline full-history zero days: leading {int(row.leading_zero_days):,}; trailing {int(row.trailing_zero_days):,}; internal {int(row.internal_zero_days):,}. These are baseline counts, not alternative-window counts.')
                st.caption(row.limitations)
            st.download_button('Download scoped transition explanations',export_csv(alt,desc+f'; sample only; scenario={scenario}; all changed and stable rows'),'transition_explanations.csv','text/csv',key='e_transition_export')
            st.download_button('Download sample window sensitivity',export_csv(windows,desc+'; sample only'),'window_sensitivity.csv','text/csv',key='e_window_export')
        st.subheader('What evidence is still needed?')
        st.write('Observed facts: units, prices, zeros and rule-based label changes. Diagnostic proxies: CV, ADI, retrospective lifecycle boundaries and scenario sensitivity. Unverified explanations: stockouts, availability, promotions or supplier problems. Inventory/availability and operational outcome data would be needed to evaluate those explanations and decision usefulness.')
    with st.expander('Definitions and evidence'):
        st.write('X: CV ≤ 0.5; Y: 0.5 < CV ≤ 1; Z: CV > 1 or ADI ≥ 1.32. Rule comparisons round to 12 decimals. XYZ requires complete sales, at least 28 observations and positive mean; U denotes ineligibility. ABC uses the accepted cumulative revenue rules. Revenue uses weekly average price, not audited receipts.')
        st.write('Source: accepted Phase 1.6 Checkpoint D population profiles and Checkpoint C transition register. No raw population recalculation. Internal run-island verification in D covered 120 audit series; full-population count/moment and revenue reconciliations covered all 30,490.')
        st.write(LIMITS)
        st.download_button('Download evidence provenance',(directory/'provenance.json').read_bytes(),'diagnostic_provenance.json','application/json',key='e_provenance_export')
