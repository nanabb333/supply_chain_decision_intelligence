"""Development-only paired robustness helpers; no model training/data loader."""
import numpy as np
import pandas as pd
ORIGINS=(1801,1829,1857)
TIE=1e-10
COMPARISONS=(('M3','M1'),('M1','MA28'),('M3','MA28'))

def check_origin(o):
 if o not in ORIGINS:raise ValueError('Final/unapproved origin forbidden')

def validate_keys(frame,expected=None):
 keys=['item_id','store_id']
 if frame.duplicated(keys).any():raise ValueError('Duplicate paired keys')
 actual=set(map(tuple,frame[keys].to_numpy()))
 if expected is not None and actual!=expected:raise ValueError('Unmatched cohort')
 return actual

def validate_values(predictions,targets):
 shape=np.asarray(targets[0]).shape
 if len(shape)!=2 or shape[1]!=28:raise ValueError('Incorrect horizon')
 if any(np.asarray(y).shape!=shape or not np.array_equal(y,targets[0]) for y in targets):raise ValueError('Different targets')
 if any(np.asarray(p).shape!=shape or not np.isfinite(p).all() or (p<0).any() for p in predictions):raise ValueError('Incomplete/negative predictions')
 if not np.isfinite(targets[0]).all() or (targets[0]<0).any():raise ValueError('Invalid targets')

def relative(delta,reference):return float(100*delta/reference) if reference>0 else np.nan

def describe(frame,left='M3',right='M1',tol=TIE):
 l=frame[left+'_mae'].to_numpy(float);r=frame[right+'_mae'].to_numpy(float)
 if not len(l) or not np.isfinite(l).all() or not np.isfinite(r).all():raise ValueError('Empty/nonfinite comparison')
 d=l-r;mean=float(d.mean());w=int((d < -tol).sum());v=int((d>tol).sum());ties=len(d)-w-v
 return dict(comparison=left+'_minus_'+right,n=len(d),left_mae=float(l.mean()),reference_mae=float(r.mean()),delta_mae=mean,relative_delta_pct=relative(mean,float(r.mean())),reduction_pct=-relative(mean,float(r.mean())),improved=w,worsened=v,tied=ties,improved_pct=100*w/len(d),worsened_pct=100*v/len(d),tied_pct=100*ties/len(d),median_delta=float(np.median(d)),p01=float(np.quantile(d,.01)),p05=float(np.quantile(d,.05)),p25=float(np.quantile(d,.25)),p75=float(np.quantile(d,.75)),p95=float(np.quantile(d,.95)),p99=float(np.quantile(d,.99)),left_bias=float(frame[left+'_bias'].mean()),reference_bias=float(frame[right+'_bias'].mean()),bias_difference=float((frame[left+'_bias']-frame[right+'_bias']).mean()))

def collapse_origins(frame):
 if set(frame.origin)!=set(ORIGINS) or frame.duplicated(['item_id','store_id','origin']).any():raise ValueError('Invalid origin coverage')
 counts=frame.groupby(['item_id','store_id']).origin.nunique()
 if not (counts==3).all():raise ValueError('Unmatched origins')
 cols=[c for c in frame if c.endswith('_mae') or c.endswith('_bias')]
 return frame.groupby(['item_id','store_id'],as_index=False)[cols].mean()

def bands(cohort,quartiles):
 f=cohort.copy()
 if (~f.origin.isin(ORIGINS)).any():raise ValueError('Future subgroup origin')
 if not f.positive_frequency.between(0,1).all():raise ValueError('Invalid historical frequency')
 f['frequency_band']=np.where(f.positive_frequency<.1,'[0,0.1)',np.where(f.positive_frequency<.5,'[0.1,0.5)','[0.5,1]'))
 f['adi_band']=np.where(f.adi.isna(),'undefined',np.where(f.adi<1.32,'<1.32','>=1.32'))
 f['cv_band']=np.where(f.cv.isna(),'undefined',np.where(f.cv<=.5,'<=0.5',np.where(f.cv<=1,'(0.5,1]','>1')))
 f['intensity']=''
 for origin in ORIGINS:
  mask=f.origin==origin
  q=np.asarray(quartiles.loc[origin,['q25','q50','q75']],float)
  f.loc[mask,'intensity']=np.array(['Q1','Q2','Q3','Q4'])[np.searchsorted(q,f.loc[mask,'train_mean'].to_numpy(),side='left')]
 return f

def cluster_draws(matrix,mode='row',draws=2000,seed=42):
 """Equal-size crossed block mean sensitivity; rows/cols must be balanced clusters."""
 x=np.asarray(matrix,float)
 if x.ndim!=2 or not x.size or not np.isfinite(x).all():raise ValueError('Invalid cluster matrix')
 if mode not in ('row','column','crossed'):raise ValueError('Invalid cluster mode')
 rng=np.random.default_rng(seed);out=np.empty(draws);n,m=x.shape
 for start in range(0,draws,100):
  size=min(100,draws-start)
  if mode in ('row','crossed'):
   ri=rng.integers(0,n,(size,n));means=x[ri].mean(axis=1)
  else:means=np.broadcast_to(x.mean(axis=0),(size,m))
  if mode in ('column','crossed'):
   ci=rng.integers(0,m,(size,m));out[start:start+size]=np.take_along_axis(means,ci,axis=1).mean(axis=1)
  else:out[start:start+size]=means.mean(axis=1)
 return out

def verify_manifest(root,manifest,sha):
 bad=[p for p,h in manifest.items() if not (root/p).is_file() or sha(root/p)!=h]
 if bad:raise ValueError('Protected files changed: '+str(bad))
 return len(manifest)
