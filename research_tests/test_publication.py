from pathlib import Path
import sys,io
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from supply_chain import publication_dashboard as p
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
import week2_publication_data as evidence

def test_independent_publication_numbers():
 assert evidence.verify().passed.all()

def test_final_scope_and_export():
 d=p.select({});s=p.summarize(d);assert len(d)==91470;assert set(d.origin)=={1885};assert set(s.Series)=={30490};assert set(s.Target_observations)=={853720}
 f=p.table('FINAL_MODEL_COMPARISON').set_index('model')
 for _,r in s.iterrows():assert np.isclose(r.MAE,f.loc[r.model,'macro_mae'],atol=1e-12)
 x=pd.read_csv(io.BytesIO(p.export(s,'metrics.parquet')));assert set(x.evaluation_scope)=={p.FINAL};assert set(x.origin)=={1885};assert set(x.period)=={p.PERIOD}

def test_filters_empty_single():
 assert p.select({'item_id':'DOES_NOT_EXIST'}).empty
 d=p.select({'store_id':'CA_1','item_id':'FOODS_1_001'});assert len(d)==3;assert set(p.summarize(d).Target_observations)=={28}
 with pytest.raises(ValueError):p.select({'origin':1801})
 with pytest.raises(ValueError):p.table('forecasts')

@pytest.mark.parametrize('scope',[p.FINAL,p.DEVELOPMENT])
@pytest.mark.parametrize('view',p.dev.VIEWS)
def test_all_publication_views(scope,view):
 at=AppTest.from_file(str(R/'app/final_research_app.py'),default_timeout=30).run()
 at.selectbox(key='publication_scope').set_value(scope);at.selectbox(key='publication_view').set_value(view);at.run();assert not at.exception

def test_no_training_entry_points():
 s=(R/'src/supply_chain/publication_dashboard.py').read_text()
 assert 'run_week2_e2b' not in s and 'subprocess' not in s and 'final_evaluation import' not in s

def test_publication_refuses_overwrite():
 import publish_week2_f
 with pytest.raises(FileExistsError):publish_week2_f.publish(evidence.F/'publication-v2')


def test_period_comparison_metadata():
 x=pd.read_csv(io.BytesIO(p.export(p.table('DEVELOPMENT_VS_FINAL'),'reports/week2/checkpoint-e2b-v1/DEVELOPMENT_VS_FINAL.csv')))
 assert x.period.str.contains('Development:').all() and x.period.str.contains('Final:').all()
 assert x.weighting.str.contains('no pooled').all()
