import numpy as np
import pandas as pd
import pytest
from supply_chain.xyz_features import diagnostics,matrix,training_arrays,snapshots,select_pilot,feature_names

def meta(n=2):return pd.DataFrame({'item_id':[str(i) for i in range(n)],'store_id':'s','cat_id':'c','dept_id':'d'})

def test_origin_specific_cv_adi_xyz_and_zero():
 u=np.zeros((2,365));u[0]=2;d=diagnostics(u,365)
 assert d.loc[0,'cv']==0 and d.loc[0,'adi']==1 and d.loc[0,'xyz']=='X'
 assert d.loc[1,'xyz']=='U' and np.isnan(d.loc[1,'adi'])
 u[0,::2]=0;d=diagnostics(u,365);assert d.loc[0,'xyz']=='Z' and d.loc[0,'adi']==pytest.approx(365/182)

def test_features_exact_lags_and_window():
 u=np.tile(np.arange(365),(2,1));x=matrix(u,meta(),365,'M3')
 assert x.shape==(56,24) and x[0,0]==364 and x[0,1]==358 and x[0,2]==337
 assert x[0,3]==361 and x[0,4]==350.5
 assert np.array_equal(x[:28,12],np.arange(1,29))
 assert x[0,9]==(pd.Timestamp('2011-01-29')+pd.Timedelta(days=365)).dayofweek

def test_future_mutation_and_no_outer_labels():
 u=np.ones((2,1801));u[:,365:]=99
 a=matrix(u[:,:365],meta(),365,'M3');u[:,365:]=1e8
 assert np.array_equal(a,matrix(u[:,:365],meta(),365,'M3'),equal_nan=True)
 with pytest.raises(ValueError):diagnostics(u,365)

@pytest.mark.parametrize('origin',[1885,1913,1802])
def test_final_or_unapproved_origin(origin):
 with pytest.raises(ValueError):snapshots(origin)

def test_same_targets_and_matched_columns():
 u=np.tile(np.arange(1801),(2,1));arrays={a:training_arrays(u,meta(),1801,a) for a in ['M0','M1','M2','M3']}
 for a,(x,y) in arrays.items():
  assert np.array_equal(y,arrays['M0'][1])
  assert np.array_equal(x[:,:13],arrays['M0'][0])
 assert np.array_equal(arrays['M1'][0],arrays['M3'][0][:,:20])
 assert np.array_equal(arrays['M2'][0][:,-4:],arrays['M3'][0][:,-4:])
 assert np.array_equal(arrays['M0'][1][:28],np.arange(365,393))
 assert max(snapshots(1801))+28<=1801

def test_pilot_hash_and_future_free_selection():
 m=meta(10);m['eligible']=True
 a=select_pilot(m,4);b=select_pilot(m.sample(frac=1,random_state=4),4)
 assert a.item_id.tolist()==b.item_id.tolist()
 m.loc[m.item_id==a.item_id.iloc[0],'eligible']=False
 assert a.item_id.iloc[0] not in select_pilot(m,4).item_id.tolist()

@pytest.mark.parametrize('bad',['missing','negative','short'])
def test_invalid_history(bad):
 u=np.ones((2,365))
 if bad=='missing':u[0,0]=np.nan
 elif bad=='negative':u[0,0]=-1
 else:u=u[:,:364]
 with pytest.raises(ValueError):diagnostics(u,365)

def test_no_price_revenue_abc_columns():
 for a in ['M0','M1','M2','M3']:
  assert not any(any(word in n for word in ['price','revenue','abc']) for n in feature_names(a))

def test_each_snapshot_recomputed():
 u=np.ones((2,1801));u[0,365:]=0
 x,y=training_arrays(u,meta(),1801,'M3');block=2*28
 assert x[0,-4]==1 # X at365
 assert not np.array_equal(x[0,13:20],x[block,13:20])
