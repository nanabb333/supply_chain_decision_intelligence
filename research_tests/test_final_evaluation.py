import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd,pytest
from supply_chain import final_evaluation as f
R=Path(__file__).resolve().parents[1]

def fixture():
 meta=pd.DataFrame({'item_id':['a','b','c','d'],'store_id':['s']*4,'cat_id':['cat']*4,'dept_id':['dept']*4,'state_id':['st']*4})
 u=np.random.default_rng(123).poisson(.8,(4,1913)).astype(float);u[0,:1885]=0
 return meta,u

@pytest.fixture
def runner(monkeypatch,tmp_path):
 if importlib.util.find_spec('threadpoolctl') is None:monkeypatch.syspath_prepend(str(R/'reports/week2/checkpoint-c-v1/model-env/lib/python3.12/site-packages'))
 spec=importlib.util.spec_from_file_location('final_runner_test',R/'scripts/run_week2_e2b.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 monkeypatch.setattr(m,'O',tmp_path);monkeypatch.setattr(m,'N',4);return m

def test_fingerprint_exact_and_corrupt_copy(tmp_path):
 assert f.verify_protocol(R)['origin']==1885
 a=tmp_path/'reports/week2/checkpoint-e2a-v1';a.mkdir(parents=True);(a/'FROZEN_PROTOCOL_MANIFEST.json').write_text('{}')
 with pytest.raises(ValueError):f.verify_protocol(tmp_path)

def test_feature_target_isolation_and_boundary():
 meta,u=fixture();before=f.final_factors(u[:,:1885],meta);u[:,1885:]=999999
 after=f.final_factors(u[:,:1885],meta)
 for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
 with pytest.raises(ValueError):f.final_factors(u,meta)
 assert max(f.SNAPSHOTS)+28==1877 and len(f.SNAPSHOTS)==54
 assert not set(f.history_columns())&set(f.target_columns()[2:])
 assert before[0][0,107]==1  # all-zero U, retained

def test_output_immutability_and_release_guard(tmp_path):
 p=tmp_path/'x.json';f.atomic_json(p,{'x':1})
 with pytest.raises(FileExistsError):f.atomic_json(p,{'x':2})
 assert json.loads(p.read_text())=={'x':1}
 with pytest.raises(ValueError):f.require_prediction_manifest(tmp_path)
 f.atomic_json(tmp_path/'FINAL_PREDICTION_MANIFEST.json',{'protocol_sha256':f.DIGEST,'coverage_validated':True,'files':{'x.json':f.sha(p)}})
 f.require_prediction_manifest(tmp_path);p.write_text('tampered')
 with pytest.raises(ValueError):f.require_prediction_manifest(tmp_path)

def test_score_rejects_before_any_target_io(runner,monkeypatch):
 def forbidden(*a,**k):raise AssertionError('Target read before prediction freeze')
 monkeypatch.setattr(runner.pd,'read_csv',forbidden)
 with pytest.raises(ValueError):runner.score_stage()

def test_synthetic_end_to_end_publication_scoring(runner,monkeypatch,tmp_path):
 meta,u=fixture();monkeypatch.setattr(runner,'histories',lambda:iter([(meta,u[:,:1885])]))
 runner.cohort_stage()
 paths=[]
 for arm in ['M1','M3']:
  # Synthetic fixed models, not final-real fitting. Exercise complete output schema.
  ix=f.ridge.indices(arm);model={'indices':ix,'coef':np.zeros(len(ix)),'intercept':np.array(.75),'mean':np.zeros(len(ix)),'scale':np.ones(len(ix)),'penalty':np.array(1e-6),'residual':np.array(0.)}
  name=arm+'-model.npz';runner.npz_write(model,name);paths.append(name)
 runner.npz_write(f.ridge.empty_stats(),'statistics.npz');runner.csv_write(pd.DataFrame({'training_rows':[0]}),'training-coverage.csv');paths+=['statistics.npz','training-coverage.csv']
 runner.receipt('FIT_COMPLETE.json',paths)
 runner.predict_stage();f.require_prediction_manifest(tmp_path)
 truth=meta[['item_id','store_id']].copy()
 truth=pd.concat([truth,pd.DataFrame(u[:,1885:],columns=[f'd_{i}' for i in range(1886,1914)])],axis=1)
 original=runner.pd.read_csv
 def read(path,*a,**k):
  if Path(path)==runner.RAW:
   assert k['usecols']==f.target_columns();return truth.copy()
  return original(path,*a,**k)
 monkeypatch.setattr(runner.pd,'read_csv',read);runner.score_stage()
 result=original(tmp_path/'FINAL_MODEL_COMPARISON.csv').set_index('model')
 assert result.loc['M1','macro_mae']==np.abs(.75-u[:,1885:]).mean()
 assert result.target_observations.eq(112).all()
 assert f.sha(tmp_path/'forecasts.parquet')==f.require_prediction_manifest(tmp_path)['files']['forecasts.parquet']
 with pytest.raises(FileExistsError):runner.predict_stage()

def test_invalid_forecasts_rejected():
 meta,u=fixture();a={m:np.zeros((4,28)) for m in f.MODELS};a['M1'][0,0]=-1
 with pytest.raises(ValueError):f.long_forecasts(meta,a)

def test_final_bands_match_accepted_development_method():
 from supply_chain.forecast_robustness import bands
 frame=pd.DataFrame({'origin':[1801]*4,'positive_frequency':[0,.1,.5,1],'cv':[np.nan,.5,1,2],'adi':[np.nan,1.32,1,2],'train_mean':[0,1,2,3]})
 new,q=f.final_bands(frame);quart=pd.DataFrame({'origin':[1801,1829,1857],'q25':[q[0]]*3,'q50':[q[1]]*3,'q75':[q[2]]*3}).set_index('origin');old=bands(frame,quart)
 for col in ['frequency_band','cv_band','adi_band','intensity']:assert new[col].tolist()==old[col].tolist()
