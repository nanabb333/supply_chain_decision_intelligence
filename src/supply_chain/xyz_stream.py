"""Bounded-memory exact ridge sufficient statistics; development origins only."""
import numpy as np
import pandas as pd
from .xyz_features import diagnostics,hash_code,snapshots,CONT,LABEL
from .baselines import H,allowed_origin
LAMBDAS=(1e-6,1e-4,1e-2,1.)
HASHES=(('item_id',64),('store_id',16),('cat_id',4),('dept_id',8))
SERIES_NAMES=['lag1','lag7','lag28','mean7','mean28']+[f'{name}_hash_{i}' for name,b in HASHES for i in range(b)]+CONT+LABEL
CALENDAR_NAMES=['weekday','month','day_of_year','horizon']
NAMES=SERIES_NAMES+CALENDAR_NAMES

def indices(arm):
 if arm not in ('M0','M1','M2','M3'):raise ValueError('Unknown arm')
 return np.array(list(range(97))+(list(range(97,104)) if arm in ('M1','M3') else [])+(list(range(104,108)) if arm in ('M2','M3') else [])+list(range(108,112)))

def static_features(meta):
 a=np.zeros((len(meta),92));offset=0
 for name,b in HASHES:
  a[np.arange(len(meta)),offset+np.array([hash_code(v,b) for v in meta[name]])]=1;offset+=b
 return a

def factors(history,meta,t,static=None):
 u=np.asarray(history,float)
 if u.ndim!=2 or u.shape[1]!=t or len(meta)!=len(u):raise ValueError('History must end exactly at snapshot')
 d=diagnostics(u,t)
 if static is None:static=static_features(meta)
 if static.shape!=(len(u),92):raise ValueError('Static feature mismatch')
 base=np.column_stack([u[:,-1],u[:,-7],u[:,-28],u[:,-7:].mean(axis=1),u[:,-28:].mean(axis=1)])
 cont=d[CONT].to_numpy(float);cont[:,:2]=np.nan_to_num(cont[:,:2],nan=0.)
 labels=np.column_stack([d.xyz.eq(k) for k in ('X','Y','Z','U')])
 s=np.column_stack([base,static,cont,labels])
 dates=pd.date_range(pd.Timestamp('2011-01-29')+pd.Timedelta(days=t),periods=H)
 c=np.column_stack([dates.dayofweek,dates.month,dates.dayofyear,np.arange(1,H+1)]).astype(float)
 return s,c

def empty_stats():
 p=len(NAMES)
 return dict(n=np.array(0,dtype=np.int64),sx=np.zeros(p),xx=np.zeros((p,p)),xy=np.zeros(p),sy=np.array(0.),yy=np.array(0.))

def add_block(stats,s,c,y):
 s=np.asarray(s,float);c=np.asarray(c,float);y=np.asarray(y,float)
 if s.shape[1]!=108 or c.shape!=(28,4) or y.shape!=(len(s),28):raise ValueError('Unmatched row/horizon dimensions')
 if not all(np.isfinite(a).all() for a in (s,c,y)):raise ValueError('Nonfinite sufficient-statistic input')
 n=len(s);ss=s.sum(axis=0);cs=c.sum(axis=0)
 stats['n']+=n*H;stats['sx'][:108]+=H*ss;stats['sx'][108:]+=n*cs
 stats['xx'][:108,:108]+=H*(s.T@s)
 cross=np.outer(ss,cs);stats['xx'][:108,108:]+=cross;stats['xx'][108:,:108]+=cross.T
 stats['xx'][108:,108:]+=n*(c.T@c)
 stats['xy'][:108]+=s.T@y.sum(axis=1);stats['xy'][108:]+=c.T@y.sum(axis=0)
 stats['sy']+=y.sum();stats['yy']+=np.square(y).sum()

def solve(stats,arm,penalty):
 if penalty not in LAMBDAS:raise ValueError('Unapproved penalty')
 ix=indices(arm);n=int(stats['n'])
 if n<=0:raise ValueError('Empty training data')
 mean=stats['sx'][ix]/n;ym=float(stats['sy'])/n
 cov=stats['xx'][np.ix_(ix,ix)]/n-np.outer(mean,mean)
 var=np.diag(cov).copy()
 if var.min() < -1e-6:raise ValueError('Numerical covariance failure')
 scale=np.sqrt(np.maximum(var,0));scale[scale<1e-10]=1
 gram=cov/np.outer(scale,scale);rhs=(stats['xy'][ix]/n-mean*ym)/scale
 a=gram+penalty*np.eye(len(ix));beta=np.linalg.solve(a,rhs)
 residual=float(np.max(np.abs(a@beta-rhs)))
 if not np.isfinite(beta).all() or residual>1e-6:raise ValueError('Failed ridge normal-equation residual')
 coef=beta/scale;intercept=ym-mean@coef
 return dict(indices=ix,coef=coef,intercept=np.array(intercept),mean=mean,scale=scale,penalty=np.array(penalty),residual=np.array(residual))

def forecast(model,s,c):
 # Predict without allocating the complete112-column daily panel.
 coef=np.zeros(112);coef[model['indices']]=model['coef']
 p=(s@coef[:108])[:,None]+(c@coef[108:])[None,:]+float(model['intercept'])
 if not np.isfinite(p).all():raise ValueError('Nonfinite predictions')
 return np.maximum(p,0.)

def validate_origin_history(u,origin):
 allowed_origin(origin)
 if u.ndim!=2 or u.shape[1]!=origin:raise ValueError('Origin history mismatch')
 if not np.isfinite(u).all() or (u<0).any() or (u!=np.floor(u)).any():raise ValueError('Past-only eligibility failure')
 return snapshots(origin)
