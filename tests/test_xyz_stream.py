import numpy as np
import pandas as pd
import pytest
from supply_chain.xyz_stream import *
from supply_chain.xyz_features import matrix

def fixture(n=4,t=1801):
 u=np.random.default_rng(42).poisson(.8,(n,t)).astype(float);u[0]=0
 m=pd.DataFrame(dict(item_id=[f'i{k}' for k in range(n)],store_id=['s']*n,cat_id=['c']*n,dept_id=['d']*n))
 return u,m

def explicit(s,c):return np.column_stack([np.repeat(s,28,axis=0),np.tile(c,(len(s),1))])

@pytest.mark.parametrize('arm',['M0','M1','M2','M3'])
def test_streamed_gram_and_predictions_match_explicit(arm):
 u,m=fixture();g=empty_stats();xx=[];yy=[]
 for t in [365,393]:
  s,c=factors(u[:,:t],m,t);y=u[:,t:t+28];add_block(g,s,c,y);xx.append(explicit(s,c));yy.append(y.ravel())
 x=np.vstack(xx);y=np.concatenate(yy);np.testing.assert_allclose(g['xx'],x.T@x,rtol=1e-10,atol=1e-7)
 np.testing.assert_allclose(g['xy'],x.T@y,rtol=1e-10,atol=1e-8);assert g['n']==len(y)
 ix=indices(arm);z=x[:,ix];mean=z.mean(0);scale=z.std(0);scale[scale<1e-10]=1;z=(z-mean)/scale
 for lam in LAMBDAS:
  model=solve(g,arm,lam);beta=np.linalg.solve(z.T@z+lam*len(y)*np.eye(len(ix)),z.T@(y-y.mean()))
  expected=np.maximum(z@beta+y.mean(),0)
  actual=np.maximum(x[:,ix]@model['coef']+model['intercept'],0)
  np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-7)

def test_features_preserve_old_information():
 u,m=fixture();s,c=factors(u,m,1801);old=matrix(u,m,1801,'M3').reshape(4,28,24)
 np.testing.assert_allclose(s[:,:5],old[:,0,:5]);np.testing.assert_allclose(c,old[0,:,9:13])
 np.testing.assert_allclose(s[:,97:104],np.nan_to_num(old[:,0,13:20],nan=0));np.testing.assert_array_equal(s[:,104:],old[:,0,20:])
 off=5
 for j,(_,b) in enumerate(HASHES):
  np.testing.assert_array_equal(np.argmax(s[:,off:off+b],axis=1),old[:,0,5+j]);off+=b

@pytest.mark.parametrize('origin',[1801,1829,1857])
def test_training_coverage_and_future_isolation(origin):
 u,m=fixture(t=1885);ts=validate_origin_history(u[:,:origin],origin)
 assert len(ts)=={1801:51,1829:52,1857:53}[origin] and max(ts)+28<=origin
 a=factors(u[:,:origin],m,origin);u[:,origin:]=9999;b=factors(u[:,:origin],m,origin)
 for v,w in zip(a,b):np.testing.assert_array_equal(v,w)

def test_final_origin_empty_and_bad_dimensions_rejected():
 with pytest.raises(ValueError):validate_origin_history(np.zeros((1,1885)),1885)
 with pytest.raises(ValueError):solve(empty_stats(),'M0',1e-4)
 with pytest.raises(ValueError):factors(np.zeros((1,365)),pd.DataFrame(index=range(2)),365)
 with pytest.raises(ValueError):add_block(empty_stats(),np.zeros((1,108)),np.zeros((28,4)),np.zeros((1,27)))

def test_chunk_order_consistency():
 u,m=fixture();s,c=factors(u[:,:365],m,365);a=empty_stats();b=empty_stats();y=u[:,365:393]
 add_block(a,s,c,y)
 for i in range(len(u)):add_block(b,s[i:i+1],c,y[i:i+1])
 for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-9)
 for arm in ('M0','M1','M2','M3'):
  np.testing.assert_allclose(forecast(solve(a,arm,1e-4),s,c),forecast(solve(b,arm,1e-4),s,c),atol=1e-7)
