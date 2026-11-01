"""Population descriptive diagnostics on bounded wide-sales rows, no daily panel."""
import numpy as np
import pandas as pd

RUN_BINS=[('1',1,1),('2-6',2,6),('7-27',7,27),('28+',28,None)]

def profile_row(values):
    u=np.asarray(values,dtype=float)
    if u.ndim!=1 or len(u)==0:raise ValueError('Nonempty daily vector required')
    observed=np.isfinite(u)
    if np.isinf(u).any() or ((u[observed]<0)|(u[observed]!=np.floor(u[observed]))).any():raise ValueError('Invalid unit observation')
    z=u==0;pos=np.flatnonzero(u>0);zero_indices=np.diff(np.r_[False,z,False].astype(np.int8))
    starts=np.flatnonzero(zero_indices==1);ends=np.flatnonzero(zero_indices==-1)-1;lengths=ends-starts+1
    first=int(pos[0]) if len(pos) else None;last=int(pos[-1]) if len(pos) else None
    internal=lengths[(starts>first)&(ends<last)] if len(pos) else np.array([],dtype=int)
    mean=float(np.nanmean(u)) if observed.any() else np.nan;std=float(np.nanstd(u,ddof=0)) if observed.any() else np.nan
    cv=std/mean if mean>0 else np.nan;adi=int(observed.sum())/len(pos) if len(pos) else np.nan
    eligible=bool(observed.all() and observed.sum()>=28 and mean>0)
    cvclass=('X' if round(cv,12)<=.5 else 'Y' if round(cv,12)<=1 else 'Z') if eligible else 'U'
    xyz='Z' if eligible and round(adi,12)>=1.32 else cvclass
    r=dict(calendar_days=len(u),observed_days=int(observed.sum()),missing_sales=int((~observed).sum()),
        units=int(np.nansum(u)) if observed.any() else None,positive_days=len(pos),zero_days=int(z.sum()),
        positive_frequency=len(pos)/observed.sum() if observed.any() else np.nan,
        zero_frequency=z.sum()/observed.sum() if observed.any() else np.nan,mean_units=mean,std_units=std,cv=cv,adi=adi,
        first_positive_index=first,last_positive_index=last,
        leading_zero_days=int(z[:first].sum()) if len(pos) else None,
        trailing_zero_days=int(z[last+1:].sum()) if len(pos) else None,
        internal_zero_days=int(internal.sum()) if len(pos) else None,
        positive_span_days=last-first+1 if len(pos) else None,
        span_defined=bool(len(pos)),longest_zero_run=int(lengths.max()) if len(lengths) else 0,
        internal_runs=len(internal),longest_internal_zero_run=int(internal.max()) if len(internal) else 0,
        xyz_eligible=eligible,xyz=xyz,cv_class=cvclass,
        adi_rule_triggered=bool(eligible and round(adi,12)>=1.32),
        xyz_exclusion_reason='' if eligible else ('Missing sales' if not observed.all() else 'Fewer than 28 observed days' if len(u)<28 else 'No positive sales'))
    for name,lo,hi in RUN_BINS:
        match=internal[(internal>=lo)&(internal<=hi if hi is not None else True)]
        r[f'internal_runs_{name}']=len(match);r[f'internal_days_{name}']=int(match.sum())
    return r,internal

def ecdf_distance(sample,population):
    a=np.sort(np.asarray(sample,dtype=float));b=np.sort(np.asarray(population,dtype=float))
    a=a[np.isfinite(a)];b=b[np.isfinite(b)]
    if not len(a) or not len(b):return np.nan
    grid=np.unique(np.r_[a,b]);return float(np.max(np.abs(np.searchsorted(a,grid,side='right')/len(a)-np.searchsorted(b,grid,side='right')/len(b))))

def categorical_comparison(pop,field):
    counts=pop[field].value_counts(dropna=False);s=pop[pop.selected][field].value_counts(dropna=False)
    rows=[]
    for value,n in counts.items():
        k=int(s.get(value,0));p=n/len(pop);q=k/int(pop.selected.sum())
        rows.append(dict(dimension=field,value=str(value),population_series=int(n),sample_series=k,population_share=p,sample_share=q,
            share_difference_pp=100*(q-p),relative_share_difference=q/p-1 if p else None,selection_coverage=k/n))
    return pd.DataFrame(rows)

def numerical_comparison(pop,fields):
    rows=[]
    for field in fields:
        a=pop.loc[pop.selected,field].dropna().astype(float);b=pop[field].dropna().astype(float)
        pooled_std=b.std(ddof=0)
        values={'mean':(a.mean(),b.mean()),'median':(a.median(),b.median())}
        values.update({f'p{int(q*100)}':(a.quantile(q),b.quantile(q)) for q in [.1,.25,.75,.9,.95,.99]})
        for stat,(s,p) in values.items():
            rows.append(dict(metric=field,statistic=stat,sample=float(s),population=float(p),absolute_difference=float(s-p),relative_difference=float(s/p-1) if p!=0 else None,sample_defined=len(a),population_defined=len(b)))
        rows.append(dict(metric=field,statistic='ecdf_max_distance',sample=None,population=None,absolute_difference=ecdf_distance(a,b),relative_difference=None,sample_defined=len(a),population_defined=len(b)))
        rows.append(dict(metric=field,statistic='mean_difference_in_population_sd',sample=None,population=None,absolute_difference=float((a.mean()-b.mean())/pooled_std) if pooled_std>0 else None,relative_difference=None,sample_defined=len(a),population_defined=len(b)))
    return pd.DataFrame(rows)

def independent_wide_sql(days):
    cols=','.join('"'+c+'"' for c in days)
    return f'''WITH vectors AS (SELECT item_id,store_id,[{cols}] v FROM batch),
    base AS (SELECT *,list_count(v) observed_days,len(v) calendar_days,list_sum(v) units,
    list_count(list_filter(v,x->x=0)) zero_days,list_count(list_filter(v,x->x>0)) positive_days,
    list_avg(v) mean_units,list_stddev_pop(v) std_units,
    list_position(list_transform(v,x->x>0),true)-1 first_positive_index,
    len(v)-list_position(list_transform(list_reverse(v),x->x>0),true) last_positive_index FROM vectors),
    m AS (SELECT *,std_units/nullif(mean_units,0) cv,observed_days::DOUBLE/nullif(positive_days,0) adi,
    calendar_days-observed_days missing_sales,
    CASE WHEN first_positive_index IS NOT NULL THEN list_count(list_filter(list_slice(v,1,first_positive_index),x->x=0)) END leading_zero_days,
    CASE WHEN last_positive_index IS NOT NULL THEN list_count(list_filter(list_slice(v,last_positive_index+2,len(v)),x->x=0)) END trailing_zero_days
    FROM base)
    SELECT * EXCLUDE(v),zero_days-leading_zero_days-trailing_zero_days internal_zero_days,
    CASE WHEN missing_sales>0 OR observed_days<28 OR NOT coalesce(mean_units>0,false) THEN 'U'
      WHEN round(adi,12)>=1.32 OR round(cv,12)>1 THEN 'Z' WHEN round(cv,12)<=.5 THEN 'X' ELSE 'Y' END xyz
    FROM m ORDER BY item_id,store_id'''
