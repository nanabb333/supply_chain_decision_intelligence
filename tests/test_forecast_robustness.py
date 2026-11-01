import hashlib
import numpy as np
import pandas as pd
import pytest
from supply_chain.forecast_robustness import *

def frame():
 return pd.DataFrame(dict(item_id=['a','b'],store_id=['s','s'],M1_mae=[2.,2.],M3_mae=[1.,3.],MA28_mae=[1.,1.],M1_bias=[0.,0.],M3_bias=[-1.,1.],MA28_bias=[0.,0.]))

def test_paired_uniqueness_and_target_matching():
 f=frame();assert len(validate_keys(f))==2
 with pytest.raises(ValueError):validate_keys(pd.concat([f,f]))
 with pytest.raises(ValueError):validate_keys(f,{('a','s')})
 y=np.zeros((2,28));validate_values([y,y],[y,y.copy()])
 z=y.copy();z[0,1]=1
 with pytest.raises(ValueError,match='targets'):validate_values([y,y],[y,z])
 with pytest.raises(ValueError):validate_values([-np.ones_like(y)],[y])

def test_sign_macro_and_zero_denominator():
 f=frame();r=describe(f);assert r['improved']==r['worsened']==1 and r['delta_mae']==0 and r['median_delta']==0
 assert describe(f.iloc[:1])['delta_mae']==-1 and describe(f.iloc[:1])['reduction_pct']==50
 assert np.isnan(relative(1,0))
 f.M1_mae=0;assert np.isnan(describe(f)['relative_delta_pct'])

def test_origin_collapse_and_decomposition():
 f=pd.concat([frame().assign(origin=o,M3_mae=[1.+j,3.+j]) for j,o in enumerate(ORIGINS)])
 overall=describe(collapse_origins(f))['delta_mae'];per=[describe(g)['delta_mae'] for _,g in f.groupby('origin')]
 assert overall==np.mean(per)==1
 with pytest.raises(ValueError):collapse_origins(f.iloc[:-1])

def test_historical_band_boundaries_and_undefined():
 f=pd.DataFrame(dict(origin=[1801]*6,positive_frequency=[0,.099,.1,.499,.5,1],adi=[np.nan,1,1.32,2,3,1],cv=[np.nan,.5,1,1.01,0,2],train_mean=[0,1,2,3,4,5]))
 q=pd.DataFrame(dict(origin=ORIGINS,q25=[1]*3,q50=[2]*3,q75=[3]*3)).set_index('origin');b=bands(f,q)
 assert b.frequency_band.tolist()==['[0,0.1)','[0,0.1)','[0.1,0.5)','[0.1,0.5)','[0.5,1]','[0.5,1]']
 assert b.cv_band.tolist()==['undefined','<=0.5','(0.5,1]','>1','<=0.5','>1']
 assert b.intensity.tolist()==['Q1','Q1','Q2','Q3','Q4','Q4']
 for col in ['frequency_band','adi_band','cv_band','intensity']:assert b.groupby(col).size().sum()==len(f)

@pytest.mark.parametrize('mode',['row','column','crossed'])
def test_cluster_aware_deterministic_resampling(mode):
 x=np.array([[1.,2.],[10.,20.],[100.,200.]])
 a=cluster_draws(x,mode,100,42);b=cluster_draws(x,mode,100,42);np.testing.assert_array_equal(a,b)
 np.testing.assert_array_equal(cluster_draws(np.ones((4,3)),mode,10),np.ones(10))
 if mode=='row':
  rng=np.random.default_rng(42);ix=rng.integers(0,3,(100,3));expected=x.mean(1)[ix].mean(1);np.testing.assert_allclose(a,expected)

def test_global_calendar_shocks_remain_shared():
 x=np.tile(np.array([-5.,5.]),(100,1))
 assert np.all(cluster_draws(x,'row',100)==0)
 assert np.std(cluster_draws(x,'column',100))>0

def test_holdout_and_protected_files(tmp_path):
 with pytest.raises(ValueError):check_origin(1885)
 p=tmp_path/'accepted';p.write_text('original');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m={'accepted':sha(p)}
 assert verify_manifest(tmp_path,m,sha)==1
 p.write_text('changed')
 with pytest.raises(ValueError):verify_manifest(tmp_path,m,sha)

def test_tolerance_and_no_missing_success_only_scoring():
 f=frame();f.M3_mae=f.M1_mae+1e-12;assert describe(f)['tied']==2
 f.loc[0,'M3_mae']=np.nan
 with pytest.raises(ValueError):describe(f)
