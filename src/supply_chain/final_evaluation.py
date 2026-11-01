"""E2-B scoped final adapter. Existing development guards remain unchanged."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd
from . import xyz_stream as ridge
from .xyz_features import diagnostics
from .baselines import META,training_stats,score
DIGEST='95466483b6e04f960a4f050dbed2bd0eb83de9641e227614aab372bb07cd910c'
ORIGIN=1885;H=28;N=30490;MODELS=('MA28','M1','M3')
SNAPSHOTS=tuple(range(365,1858,28))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def verify_protocol(root):
 root=Path(root);a=root/'reports/week2/checkpoint-e2a-v1'
 m=json.loads((a/'FROZEN_PROTOCOL_MANIFEST.json').read_text())
 digest=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if digest!=DIGEST or any(sha(root/p)!=v for p,v in m.items()):raise ValueError('Frozen protocol mismatch')
 return json.loads((a/'frozen-protocol.json').read_text())

def atomic_json(path,obj):
 path=Path(path)
 if path.exists():raise FileExistsError(path)
 temp=path.with_name(path.name+'.tmp')
 with temp.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
 # Exclusive publication prevents accidental replacement, even on a concurrent run.
 os.link(temp,path);temp.unlink()

def publish(temp,dest):
 if Path(dest).exists():raise FileExistsError(dest)
 os.link(temp,dest);Path(temp).unlink()

def history_valid(u):
 u=np.asarray(u,float)
 if u.ndim!=2 or u.shape[1]!=ORIGIN:raise ValueError('History must end at1885')
 return (np.isfinite(u)&(u>=0)&(u==np.floor(u))).all(axis=1)

def final_factors(u,meta):
 if not history_valid(u).all():raise ValueError('Invalid history')
 return ridge.factors(u,meta,ORIGIN)

def history_columns():return META+[f'd_{i}' for i in range(1,ORIGIN+1)]
def target_columns():return ['item_id','store_id']+[f'd_{i}' for i in range(1886,1914)]

def require_prediction_manifest(out):
 out=Path(out);p=out/'FINAL_PREDICTION_MANIFEST.json'
 if not p.exists():raise ValueError('Targets locked: no frozen prediction manifest')
 m=json.loads(p.read_text())
 if m['protocol_sha256']!=DIGEST or not m['coverage_validated']:raise ValueError('Invalid prediction release')
 if any(sha(out/k)!=v for k,v in m['files'].items()):raise ValueError('Prediction/artifact checksum mismatch')
 return m

def final_bands(f):
 f=f.copy();q=f.train_mean.quantile([.25,.5,.75]).to_numpy()
 f['intensity']=np.array(['Q1','Q2','Q3','Q4'])[np.searchsorted(q,f.train_mean,side='left')]
 f['frequency_band']=np.where(f.positive_frequency<.1,'[0,0.1)',np.where(f.positive_frequency<.5,'[0.1,0.5)','[0.5,1]'))
 f['cv_band']=np.where(f.cv.isna(),'undefined',np.where(f.cv<=.5,'<=0.5',np.where(f.cv<=1,'(0.5,1]','>1')))
 f['adi_band']=np.where(f.adi.isna(),'undefined',np.where(f.adi<1.32,'<1.32','>=1.32'))
 return f,q

def long_forecasts(meta,arrays):
 if set(arrays)!=set(MODELS):raise ValueError('Unmatched model set')
 frames=[]
 for model in MODELS:
  a=np.asarray(arrays[model],float)
  if a.shape!=(len(meta),H) or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid forecasts')
  frames.append(pd.DataFrame({'item_id':np.repeat(meta.item_id.to_numpy(),H),'store_id':np.repeat(meta.store_id.to_numpy(),H),'origin':ORIGIN,'model':model,'horizon':np.tile(np.arange(1,29),len(meta)),'target_day':np.tile(np.arange(1886,1914),len(meta)),'prediction':a.ravel()}))
 return pd.concat(frames,ignore_index=True)

def paired_metrics(wide,left,right,tol=1e-10):
 from .forecast_robustness import describe
 return describe(wide,left,right,tol)
