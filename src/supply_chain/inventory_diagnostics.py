"""D2 descriptive cohort diagnostics only; no simulator or forecast fitting."""
import numpy as np
import pandas as pd

def compare_numeric(sample,population):
    a=np.asarray(sample,dtype=float);b=np.asarray(population,dtype=float)
    if np.isinf(a).any() or np.isinf(b).any():raise ValueError('Infinite values must be explicitly resolved')
    x=a[np.isfinite(a)];y=b[np.isfinite(b)]
    if not len(x) or not len(y):return dict(sample_n=len(x),population_n=len(y),sample_missing=len(a)-len(x),population_missing=len(b)-len(y),ks_distance=None,standardized_mean_difference=None)
    grid=np.unique(np.r_[x,y]);ks=np.max(np.abs(np.searchsorted(np.sort(x),grid,side='right')/len(x)-np.searchsorted(np.sort(y),grid,side='right')/len(y)))
    out=dict(sample_n=len(x),population_n=len(y),sample_missing=len(a)-len(x),population_missing=len(b)-len(y),ks_distance=float(ks),standardized_mean_difference=float((x.mean()-y.mean())/y.std(ddof=0)) if y.std(ddof=0)>0 else None)
    for name,z in [('sample',x),('population',y)]:
        for k,v in [('mean',z.mean()),('std',z.std(ddof=0)),('min',z.min()),('p25',np.quantile(z,.25)),('median',np.median(z)),('p75',np.quantile(z,.75)),('p90',np.quantile(z,.9)),('p99',np.quantile(z,.99)),('max',z.max())]:out[f'{name}_{k}']=float(v)
    return out

def compare_category(a,b):
    a=pd.Series(a).fillna('UNDEFINED').astype(str);b=pd.Series(b).fillna('UNDEFINED').astype(str)
    if not len(a) or not len(b):raise ValueError('Empty cohort')
    ac=a.value_counts();bc=b.value_counts();rows=[]
    for key in sorted(set(ac.index)|set(bc.index)):
        av=int(ac.get(key,0));bv=int(bc.get(key,0));rows.append(dict(level=key,sample_count=av,population_count=bv,sample_share=av/len(a),population_share=bv/len(b),difference_pp=100*(av/len(a)-bv/len(b))))
    return rows,sum(abs(r['sample_share']-r['population_share']) for r in rows)/2

def attach_sample(population,ids):
    keys=['item_id','store_id']
    if population.duplicated(keys).any() or ids.duplicated(keys).any():raise ValueError('Duplicate keys')
    merged=population.merge(ids[keys].assign(selected=True),on=keys,how='left',validate='one_to_one')
    merged['selected']=merged.selected.eq(True)
    if merged.selected.sum()!=len(ids):raise ValueError('Missing selected IDs')
    return merged
