"""Development-only univariate baselines. No final-origin or price access."""
import numpy as np
import pandas as pd
ORIGINS=(1801,1829,1857)
H=28
META=['item_id','dept_id','cat_id','store_id','state_id']
ALPHAS=(.05,.1,.2)

def allowed_origin(origin):
    if origin not in ORIGINS:raise ValueError('Only accepted development origins allowed; final holdout locked')

def load_blocks(path,start,end,chunk=512,limit=None):
    if not 1<=start<=end<=1885:raise ValueError('Final holdout access forbidden')
    if chunk<1 or (limit is not None and limit<1):raise ValueError('Invalid loader bounds')
    cols=META+[f'd_{j}' for j in range(start,end+1)]
    yield from pd.read_csv(path,usecols=cols,chunksize=chunk,nrows=limit)

def eligible(history,origin):
    allowed_origin(origin);u=np.asarray(history,dtype=float)
    if u.ndim!=2 or u.shape[1]!=origin:raise ValueError('History must contain exactly d1 through origin')
    valid=np.isfinite(u)&(u>=0)&(u==np.floor(u))
    return valid.all(axis=1)&(u.shape[1]>=365)

def predict(history,origin):
    u=np.asarray(history,dtype=float)
    if not eligible(u,origin).all():raise ValueError('Incomplete/invalid model history')
    n=len(u);out={'naive':np.repeat(u[:,-1:],H,axis=1),'seasonal7':np.tile(u[:,-7:],(1,4)),
        'ma7':np.repeat(u[:,-7:].mean(axis=1)[:,None],H,axis=1),'ma28':np.repeat(u[:,-28:].mean(axis=1)[:,None],H,axis=1)}
    z=np.zeros((n,3));p=np.ones((n,3));last=np.zeros(n,dtype=int);alpha=np.array(ALPHAS)
    for j in range(u.shape[1]):
        pos=u[:,j]>0;first=pos&(last==0);later=pos&~first
        z[first]=u[first,j,None];p[first]=j+1
        z[later]=(1-alpha)*z[later]+alpha*u[later,j,None]
        p[later]=(1-alpha)*p[later]+alpha*(j+1-last[later,None])
        last[pos]=j+1
    for k,a in enumerate(ALPHAS):
        v=z[:,k]/p[:,k]
        out[f'croston_{a:g}']=np.repeat(v[:,None],H,axis=1)
        out[f'sba_{a:g}']=np.repeat(((1-a/2)*v)[:,None],H,axis=1)
    if any(not np.isfinite(v).all() or (v<0).any() for v in out.values()):raise ValueError('Method failure')
    return out

def training_stats(u):
    mean=u.mean(axis=1);sd=u.std(axis=1,ddof=0);positive=(u>0).sum(axis=1);diff=np.diff(u,axis=1)
    cv=np.divide(sd,mean,out=np.full(len(u),np.nan),where=mean>0)
    adi=np.divide(u.shape[1],positive,out=np.full(len(u),np.nan),where=positive>0)
    return dict(train_mean=mean,cv=cv,adi=adi,positive_frequency=positive/u.shape[1],all_zero=positive==0,weak_croston=positive<=1,
                scale_abs=np.abs(diff).mean(axis=1),scale_sq=(diff**2).mean(axis=1))

def score(y,p,stats):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=H:raise ValueError('Mismatched horizon/cohort')
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid predictions')
    if np.isinf(y).any() or (y[np.isfinite(y)]<0).any():raise ValueError('Invalid actuals')
    mask=np.isfinite(y);e=np.where(mask,p-y,0);n=mask.sum(axis=1);ab=np.abs(e);sq=e**2;actual=np.where(mask,y,0).sum(axis=1)
    avg=lambda x:np.divide(x,n,out=np.full(len(y),np.nan),where=n>0)
    mae=avg(ab.sum(axis=1));mse=avg(sq.sum(axis=1));sa=stats['scale_abs'];ss=stats['scale_sq'];zero=mask&(y==0);pos=mask&(y>0)
    return pd.DataFrame(dict(n=n,sum_abs=ab.sum(axis=1),sum_sq=sq.sum(axis=1),sum_error=e.sum(axis=1),actual_sum=actual,mae=mae,bias=avg(e.sum(axis=1)),
       abs_total_error=np.abs(e.sum(axis=1)),mase=np.divide(mae,sa,out=np.full(len(y),np.nan),where=sa>0),
       rmsse=np.sqrt(np.divide(mse,ss,out=np.full(len(y),np.nan),where=ss>0)),
       zero_n=zero.sum(axis=1),positive_n=pos.sum(axis=1),zero_abs=np.where(zero,ab,0).sum(axis=1),positive_abs=np.where(pos,ab,0).sum(axis=1),
       zero_prediction_mass=np.where(zero,p,0).sum(axis=1)))
