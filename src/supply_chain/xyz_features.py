"""Past-only XYZ ablation features. No prices, ABC or final-holdout loaders."""
import hashlib
import numpy as np
import pandas as pd
from .baselines import ORIGINS,H
BASE=['lag1','lag7','lag28','mean7','mean28','item_hash','store_hash','category_hash','department_hash','weekday','month','day_of_year','horizon']
CONT=['cv','adi','mean','positive_days','positive_frequency','observed_days','undefined']
LABEL=['xyz_X','xyz_Y','xyz_Z','xyz_U']

def feature_names(arm):
 if arm not in ['M0','M1','M2','M3']:raise ValueError('Unknown arm')
 return BASE+(CONT if arm in ['M1','M3'] else [])+(LABEL if arm in ['M2','M3'] else [])

def snapshots(origin):
 if origin not in ORIGINS:raise ValueError('Final/unapproved origin forbidden')
 return list(range(365,origin-H+1,28))

def hash_code(value,buckets):return int(hashlib.sha256(str(value).encode()).hexdigest()[:16],16)%buckets

def diagnostics(history,snapshot):
 u=np.asarray(history,float)
 if u.ndim!=2 or u.shape[1]!=snapshot or snapshot<365:raise ValueError('History must end exactly at snapshot with365 days minimum')
 if not np.isfinite(u).all() or (u<0).any() or not (u==np.floor(u)).all():raise ValueError('Incomplete/invalid past observations')
 mean=u.mean(axis=1);std=u.std(axis=1,ddof=0);pos=(u>0).sum(axis=1)
 cv=np.divide(std,mean,out=np.full(len(u),np.nan),where=mean>0);adi=np.divide(snapshot,pos,out=np.full(len(u),np.nan),where=pos>0)
 xyz=np.where(pos==0,'U',np.where(np.round(adi,12)>=1.32,'Z',np.where(np.round(cv,12)<=.5,'X',np.where(np.round(cv,12)<=1,'Y','Z'))))
 return pd.DataFrame(dict(cv=cv,adi=adi,mean=mean,positive_days=pos,positive_frequency=pos/snapshot,observed_days=snapshot,undefined=pos==0,xyz=xyz))

def matrix(history,meta,snapshot,arm):
 u=np.asarray(history,float);d=diagnostics(u,snapshot)
 if len(meta)!=len(u):raise ValueError('Metadata/cohort mismatch')
 # Static fixed hashes, no target fitting. Hash collisions are disclosed.
 static=np.array([[hash_code(r.item_id,64),hash_code(r.store_id,16),hash_code(r.cat_id,4),hash_code(r.dept_id,8)] for r in meta.itertuples()],float)
 x0=np.column_stack([u[:,-1],u[:,-7],u[:,-28],u[:,-7:].mean(axis=1),u[:,-28:].mean(axis=1),static])
 x=np.repeat(x0,H,axis=0);dates=pd.date_range(pd.Timestamp('2011-01-29')+pd.Timedelta(days=snapshot),periods=H)
 calendar=np.column_stack([dates.dayofweek,dates.month,dates.dayofyear,np.arange(1,H+1)])
 x=np.column_stack([x,np.tile(calendar,(len(u),1))])
 if arm in ['M1','M3']:x=np.column_stack([x,np.repeat(d[CONT].to_numpy(float),H,axis=0)])
 if arm in ['M2','M3']:x=np.column_stack([x,np.repeat(np.column_stack([d.xyz.eq(k) for k in ['X','Y','Z','U']]),H,axis=0)])
 return np.ascontiguousarray(x,dtype=np.float64)

def training_arrays(history,meta,origin,arm):
 u=np.asarray(history,float)
 if u.shape[1]!=origin:raise ValueError('Outer training array crosses origin')
 ts=snapshots(origin);n=len(u)*H*len(ts);x=np.empty((n,len(feature_names(arm))),dtype=np.float64);y=np.empty(n,dtype=np.float64);i=0
 for t in ts:
  block=matrix(u[:,:t],meta,t,arm);end=i+len(block);assert t+H<=origin
  x[i:end]=block;y[i:end]=u[:,t:t+H].reshape(-1);i=end
 return x,y

def select_pilot(metadata,limit=256):
 if not {'item_id','store_id','eligible'}<=set(metadata.columns):raise ValueError('Past eligibility required')
 eligible=metadata[metadata.eligible].copy()
 eligible['selection_hash']=[hashlib.sha256(('week2-c-pilot|'+i+'|'+s).encode()).hexdigest() for i,s in zip(eligible.item_id,eligible.store_id)]
 return eligible.sort_values(['selection_hash','item_id','store_id']).head(limit).reset_index(drop=True)
