from pathlib import Path
from io import BytesIO
import hashlib,json,subprocess,sys
import numpy as np
import pandas as pd
import pytest
import duckdb
from streamlit.testing.v1 import AppTest
from supply_chain.diagnostic_quality import quality,labels,sensitivity,within_z,scope,zero_bands,transition_matrix,explanations,export_csv,window_summary
R=Path(__file__).resolve().parents[1];E=R/'reports/phase1_6/checkpoint-e-v1';D=E/'diagnostics'
@pytest.fixture(scope='module')
def p():return pd.read_parquet(D/'profile.parquet')
@pytest.fixture(scope='module')
def t():return pd.read_parquet(D/'transition_explanations.parquet')

def test_quality_independent_sql(p):
    q,d=quality(p)
    with duckdb.connect() as db:
        db.register('p',p)
        for cohort,where in [('Selected sample','WHERE selected'),('M5 population','')]:
            raw=db.sql(f'SELECT xyz,count(*) n FROM p {where} GROUP BY xyz').df().set_index('xyz').n
            n=raw.sum();prob=raw/n;r=q[q.cohort==cohort].iloc[0]
            assert r.hhi==pytest.approx(sum(prob**2));assert r.normalized_entropy==pytest.approx(-sum(prob*np.log(prob))/np.log(3))
            assert r.series==n;assert d[d.cohort==cohort].series.sum()==n
            trig,eff=db.sql(f"SELECT count(*) FILTER(WHERE round(adi,12)>=1.32),count(*) FILTER(WHERE round(adi,12)>=1.32 AND round(cv,12)<=1) FROM p {where}").fetchone()
            assert r.adi_trigger_count==trig and r.adi_effective_count==eff
    assert q.adi_effective_count.tolist()==[4,448]

def test_threshold_sql_and_saved_C(p,t):
    from supply_chain.diagnostic_quality import VARIANTS
    s=sensitivity(p)
    with duckdb.connect() as db:
        db.register('p',p)
        for name,(x,y,adi) in VARIANTS.items():
            clause='false' if adi is None else f'round(adi,12)>={adi}'
            alt=db.sql(f"SELECT CASE WHEN NOT xyz_eligible THEN 'U' WHEN {clause} OR round(cv,12)>{y} THEN 'Z' WHEN round(cv,12)<={x} THEN 'X' ELSE 'Y' END cls FROM p").df().cls
            assert labels(p,x,y,adi).tolist()==alt.tolist()
    for scenario,name in [('cv04_08','CV 0.4 / 0.8'),('cv06_12','CV 0.6 / 1.2'),('adi1_20','ADI 1.20'),('adi1_50','ADI 1.50')]:
        c=t[t.scenario==scenario];r=s[(s.cohort=='Selected sample')&(s.variant==name)].iloc[0]
        assert r.Z==c.xyz.eq('Z').sum();assert r.changed==c.xyz_changed.sum()

def test_within_z_partition_and_runs(p):
    z=within_z(p);b=zero_bands(p)
    for cohort,g in [('Selected sample',p[p.selected]),('M5 population',p)]:
        assert z[z.cohort==cohort].series.sum()==g.xyz.eq('Z').sum()
        assert b[b.cohort==cohort].runs.sum()==g.internal_runs.sum()
        assert b[b.cohort==cohort].zero_days.sum()==g.internal_zero_days.sum()

def test_explanations_and_transitions(t):
    raw=pd.read_parquet(R/'reports/phase1_6/run-v3/robustness/transition_register.parquet');out=explanations(raw)
    assert len(out)==900 and not out.explanation.isna().any()
    assert out.cv_delta.equals(out.cv-out.cv_baseline)
    assert out.adi_delta.equals(out.adi-out.adi_baseline)
    for scenario,g in out.groupby('scenario'):
        assert transition_matrix(g).to_numpy().sum()==90
    changed=out[out.segment_changed];assert changed.explanation.str.len().min()>0
    cutoff=out[out.abc_thresholds_changed & out.abc_changed & ~out.revenue_rank_changed]
    assert len(cutoff)>0 and cutoff.explanation_categories.str.contains('ABC threshold').all()
    assert not cutoff.explanation_categories.str.contains('Revenue-ranking').any()
    mixed=out[out.explanation_categories.str.contains('Multiple interacting')];assert len(mixed)>0
    assert mixed.explanation.str.contains('unique causal attribution is not identified').all()
    assert window_summary(t).set_index('scenario').loc['last365','changed']==27
    assert out.lifecycle_counts_scope.str.contains('Baseline full-history').all()

@pytest.mark.parametrize('store,category,department,item',[('CA_1','All','All','All'),('All','FOODS','HOBBIES_1','All'),('TX_2','HOUSEHOLD','All','HOUSEHOLD_1_061')])
def test_filter_and_export_scopes(p,t,store,category,department,item):
    g=scope(p,store,category,department,item);q,d=quality(g)
    assert q.series.tolist()==[int(g.selected.sum()),len(g)]
    filtered=t.merge(g[['item_id','store_id']],on=['item_id','store_id'],validate='many_to_one')
    assert set(map(tuple,filtered[['item_id','store_id']].drop_duplicates().to_numpy()))==set(map(tuple,g[g.selected][['item_id','store_id']].to_numpy()))
    exported=pd.read_csv(BytesIO(export_csv(g,'test scope')))
    assert len(exported)==len(g);assert exported.columns.tolist()==g.columns.tolist()+['export_scope','evidence_limitations']
    if len(g):assert exported.export_scope.eq('test scope').all() and exported.evidence_limitations.str.contains('not validated operational').all()

def test_empty_single_and_ineligible(p):
    for g in [p.iloc[:0],p.iloc[:1]]:
        q,d=quality(g);assert q.series.iloc[1]==len(g)
        sensitivity(g);within_z(g);zero_bands(g)
        if g.empty:assert q.hhi.isna().all() and d.share_all.isna().all()
        else:assert q.hhi.iloc[1]==1 and q.effective_classes.iloc[1]==1
    bad=p.iloc[:1].copy();bad['xyz_eligible']=False;bad['xyz']='U';bad['cv']=np.nan;bad['adi']=np.nan
    q,d=quality(bad);assert q.eligible.iloc[1]==0 and q.hhi.isna().all();assert labels(bad).iloc[0]=='U'

def test_population_not_sample(t,p):
    outsider=p[~p.selected].iloc[0];g=scope(p,outsider.store_id,item=outsider.item_id);q,d=quality(g)
    assert q.series.tolist()==[0,1]
    assert t.merge(g[['item_id','store_id']],on=['item_id','store_id']).empty

def test_no_overwrite_and_protected():
    for manifest,base in [(E/'protected-baseline.json',R),(D/'manifest.json',D)]:
        for f,h in json.loads(manifest.read_text()).items():assert hashlib.sha256((base/f).read_bytes()).hexdigest()==h,f
    proc=subprocess.run([sys.executable,str(R/'scripts/phase16_quality.py')],cwd=R,capture_output=True,text=True)
    assert proc.returncode!=0 and 'Completed output exists' in proc.stderr

def test_diagnostic_dashboard(p):
    a=AppTest.from_file(R/'app/phase16_app.py',default_timeout=60).run();assert not a.exception
    assert len(a.get('plotly_chart'))==6
    a.selectbox(key='e_store').select('TX_2').run();a.selectbox(key='e_item').select('HOUSEHOLD_1_061').run();assert not a.exception
    assert any('1 selected-sample series / 1 population series' in x.value for x in a.caption)
    a.selectbox(key='e_scenario').select('positive_span').run();assert not a.exception
    a.checkbox(key='e_changed').uncheck().run();assert not a.exception
    a.selectbox(key='e_category').select('FOODS').run();assert any('No series match' in x.value for x in a.info)
    a.button[0].click().run();assert not a.exception
    assert any('90 selected-sample series / 30,490 population series' in x.value for x in a.caption)
    outsider=p[~p.selected].iloc[0]
    a.selectbox(key='e_store').select(outsider.store_id).run();a.selectbox(key='e_item').select(outsider.item_id).run()
    assert not a.exception and any('No selected-sample series' in x.value for x in a.info)
    a.sidebar.radio[0].set_value('Sales performance').run();assert not a.exception and a.title[0].value=='Sales performance'
