from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import pytest
from supply_chain.baselines import predict,eligible,score,training_stats,load_blocks,ORIGINS,H

def history(pattern=None):
 u=np.zeros((1,1801))
 if pattern is not None:u[0,-len(pattern):]=pattern
 return u

def test_naive_seasonal_and_ma_boundaries():
 u=history(list(range(1,29)));p=predict(u,1801)
 assert np.array_equal(p['naive'],np.full((1,28),28.))
 assert np.array_equal(p['seasonal7'][0],np.tile(np.arange(22,29),4))
 assert np.all(p['ma7']==25) and np.all(p['ma28']==14.5)
 assert all(v.shape==(1,28) for v in p.values())

@pytest.mark.parametrize('origin',[1885,1913,1802])
def test_final_or_unapproved_origin_locked(origin):
 with pytest.raises(ValueError):predict(np.zeros((1,origin)),origin)

@pytest.mark.parametrize('end',[1886,1913,1941])
def test_holdout_loader_never_calls_reader(monkeypatch,end):
 def fail(*a,**k):pytest.fail('reader must not be called')
 monkeypatch.setattr(pd,'read_csv',fail)
 with pytest.raises(ValueError):next(load_blocks('notused',1,end))

def test_loader_column_projection(monkeypatch):
 seen={}
 def reader(*args,**kwargs):seen.update(kwargs);return iter([])
 monkeypatch.setattr(pd,'read_csv',reader);assert list(load_blocks('notused',1802,1885))==[]
 days=[int(x[2:]) for x in seen['usecols'] if x.startswith('d_')]
 assert min(days)==1802 and max(days)==1885 and len(days)==84

def test_zero_and_single_event():
 assert all(np.all(v==0) for v in predict(history(),1801).values())
 u=history();u[0,9]=20;p=predict(u,1801)
 assert np.all(p['croston_0.1']==2) and np.all(p['sba_0.1']==1.9)
 assert training_stats(u)['weak_croston'][0]

def test_croston_hand_calculation():
 u=history();u[0,2]=6;u[0,5]=12;p=predict(u,1801)
 # initial size6 interval3, next interval3 -> size6.6 / interval3.
 assert p['croston_0.1'][0,0]==pytest.approx(2.2)
 assert p['sba_0.1'][0,0]==pytest.approx(2.09)
 assert not training_stats(u)['weak_croston'][0]

def test_zeros_do_not_update_croston_states():
 u=history();u[0,10]=8;u[0,30]=4
 p=predict(u,1801);v=np.pad(u,((0,0),(0,28)));q=predict(v,1829)
 for k in p:
  if k.startswith(('croston','sba')):assert np.array_equal(p[k],q[k])

def test_deterministic_and_nonnegative():
 u=np.random.default_rng(42).poisson(.2,(3,1801));a=predict(u,1801);b=predict(u,1801)
 for k in a:assert np.array_equal(a[k],b[k]) and (a[k]>=0).all()

def test_origin_isolation_and_eligibility():
 full=np.zeros((2,1885));full[0,:1801]=1;full[1,0]=np.nan
 a=eligible(full[:,:1801],1801);assert a.tolist()==[True,False]
 p=predict(full[a,:1801].copy(),1801);full[:,1801:]=1e9
 q=predict(full[a,:1801].copy(),1801)
 for k in p:assert np.array_equal(p[k],q[k])
 with pytest.raises(ValueError):predict(full[a],1801)

def test_metrics_manual_and_missing_mask():
 u=np.tile(np.arange(1801),(2,1));stats=training_stats(u);y=np.ones((2,28));p=np.full((2,28),3.);y[1,0]=np.nan
 m=score(y,p,stats)
 assert m.n.tolist()==[28,27] and (m.mae==2).all() and (m.bias==2).all()
 assert m.abs_total_error.tolist()==[56,54] and (m.mase==2).all() and (m.rmsse==2).all()
 assert m.actual_sum.tolist()==[28,27]

def test_zero_denominators_remain_undefined():
 u=history();stats=training_stats(u);m=score(np.zeros((1,28)),np.ones((1,28)),stats).iloc[0]
 assert m.mae==1 and m.bias==1 and m.actual_sum==0 and np.isnan(m.mase) and np.isnan(m.rmsse)
 assert m.zero_n==28 and m.zero_prediction_mass==28

@pytest.mark.parametrize('bad',['negative','nan','shape'])
def test_method_failure_not_silent_drop(bad):
 y=np.ones((1,28));p=y.copy()
 if bad=='negative':p[0,0]=-1
 elif bad=='nan':p[0,0]=np.nan
 else:p=p[:,:27]
 with pytest.raises(ValueError):score(y,p,training_stats(history()))

def test_protected_artifacts():
 r=Path(__file__).resolve().parents[1];m=json.loads((r/'reports/week2/checkpoint-b-v1/protected-manifest.json').read_text())
 for k,v in m.items():
  h=hashlib.sha256()
  with (r/k).open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert h.hexdigest()==v,k

def test_accepted_split_lock_and_saved_matching():
 r=Path(__file__).resolve().parents[1]
 c=json.loads((r/'configs/week2/checkpoint-a-v1.json').read_text())
 assert tuple(c['origins'][:-1])==ORIGINS and c['horizon']==H and c['origins'][-1]==1885
 b=r/'reports/week2/checkpoint-b-v1'
 cov=pd.read_csv(b/'analysis-v1/model-coverage.csv')
 assert len(cov)==30 and cov.series.eq(30490).all() and cov.incomplete.eq(0).all()
 validation=json.loads((b/'analysis-v1/validation.json').read_text())
 assert validation['status']=='PASS' and not validation['final_holdout_targets_used']
 for origin in ORIGINS:assert cov[cov.origin==origin].targets.nunique()==1

def test_no_overwrite_execution_target(tmp_path):
 import importlib.util
 r=Path(__file__).resolve().parents[1]
 spec=importlib.util.spec_from_file_location('baseline_runner',r/'scripts/run_week2_baselines.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 with pytest.raises(FileExistsError):module.run(tmp_path)
