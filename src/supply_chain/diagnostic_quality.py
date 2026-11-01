"""Explainable diagnostics using accepted C/D summaries; no raw sales scan."""
import numpy as np
import pandas as pd

HORIZON='2011-01-29 to 2016-04-24; 1,913 days'
LIMITS='Purposive overlapping sample; descriptive rules, not validated operational utility. Zero sales do not prove stockouts or absent demand.'
VARIANTS={'Baseline':(.5,1.,1.32),'CV 0.4 / 0.8':(.4,.8,1.32),'CV 0.6 / 1.2':(.6,1.2,1.32),'ADI 1.20':(.5,1.,1.2),'ADI 1.50':(.5,1.,1.5),'CV only (ablation)':(.5,1.,None)}
KEYS=['item_id','store_id']

def scope(frame,store='All',category='All',department='All',item='All'):
    out=frame
    for field,value in [('store_id',store),('cat_id',category),('dept_id',department),('item_id',item)]:
        if value!='All':out=out[out[field].eq(value)]
    return out.copy()

def labels(frame,x=.5,y=1.,adi=1.32):
    eligible=frame.xyz_eligible.fillna(False).to_numpy(bool)
    cv=frame.cv.round(12).to_numpy();interval=frame.adi.round(12).to_numpy()
    classes=np.where(cv<=x,'X',np.where(cv<=y,'Y','Z'))
    if adi is not None:classes=np.where(interval>=adi,'Z',classes)
    return pd.Series(np.where(eligible,classes,'U'),index=frame.index)

def cohorts(frame):return [('Selected sample',frame[frame.selected]),('M5 population',frame)]

def quality(frame):
    distributions=[];summaries=[]
    for cohort,g in cohorts(frame):
        eligible=g[g.xyz_eligible];counts=eligible.xyz.value_counts().reindex(['X','Y','Z'],fill_value=0)
        n=len(eligible);shares=counts/n if n else counts.astype(float)*np.nan
        positive=shares[shares>0];hhi=float((positive**2).sum()) if n else np.nan
        trigger=eligible.adi.round(12).ge(1.32);effective=trigger&eligible.cv_class.ne('Z')
        summaries.append(dict(cohort=cohort,series=len(g),eligible=n,excluded=len(g)-n,
            eligibility_rate=n/len(g) if len(g) else np.nan,largest_class_share=float(shares.max()) if n else np.nan,
            hhi=hhi,effective_classes=1/hhi if n else np.nan,
            normalized_entropy=float(-(positive*np.log(positive)).sum()/np.log(3)) if n else np.nan,
            adi_trigger_count=int(trigger.sum()),adi_trigger_rate=float(trigger.mean()) if n else np.nan,
            adi_effective_count=int(effective.sum()),adi_effective_rate=float(effective.mean()) if n else np.nan,
            mean_positive_frequency=g.positive_frequency.mean(),mean_zero_frequency=g.zero_frequency.mean(),
            median_units=g.units.median(),horizon=HORIZON,limitations=LIMITS))
        for cls in ['X','Y','Z','U']:
            count=int(g.xyz.eq(cls).sum())
            distributions.append(dict(cohort=cohort,xyz=cls,series=count,total_series=len(g),eligible_series=n,
                share_all=count/len(g) if len(g) else np.nan,share_eligible=count/n if n and cls!='U' else np.nan))
    return pd.DataFrame(summaries),pd.DataFrame(distributions)

def sensitivity(frame):
    rows=[]
    for cohort,g in cohorts(frame):
        base=labels(g)
        for name,(x,y,adi) in VARIANTS.items():
            alt=labels(g,x,y,adi);n=int(g.xyz_eligible.sum())
            rows.append(dict(cohort=cohort,variant=name,series=len(g),eligible=n,X=int(alt.eq('X').sum()),Y=int(alt.eq('Y').sum()),Z=int(alt.eq('Z').sum()),U=int(alt.eq('U').sum()),Z_share=int(alt.eq('Z').sum())/n if n else np.nan,changed=int(alt.ne(base).sum()),changed_share=float(alt.ne(base).mean()) if len(g) else np.nan,cv_x=x,cv_y=y,adi_threshold=adi,horizon=HORIZON,limitations=LIMITS))
    return pd.DataFrame(rows)

def within_z(frame):
    rows=[]
    for cohort,g in cohorts(frame):
        z=g[g.xyz.eq('Z')&g.xyz_eligible].copy()
        trigger=z.adi.round(12).ge(1.32)
        z['rule_group']=np.where(z.cv_class.ne('Z'),'ADI only',np.where(trigger,'CV and ADI','CV only'))
        for name in ['ADI only','CV only','CV and ADI']:
            d=z[z.rule_group.eq(name)];r=dict(cohort=cohort,rule_group=name,series=len(d),Z_denominator=len(z),share_Z=len(d)/len(z) if len(z) else np.nan)
            for f in ['cv','adi','positive_frequency','units','internal_zero_days','longest_internal_zero_run']:
                for suffix,q in [('p10',.1),('median',.5),('p90',.9)]:r[f+'_'+suffix]=d[f].quantile(q)
            rows.append(r)
    return pd.DataFrame(rows)

def zero_bands(frame):
    rows=[]
    for cohort,g in cohorts(frame):
        total_runs=int(g.internal_runs.sum());total_days=int(g.internal_zero_days.sum())
        for band in ['1','2-6','7-27','28+']:
            count=int(g['internal_runs_'+band].sum());days=int(g['internal_days_'+band].sum())
            rows.append(dict(cohort=cohort,band=band,runs=count,zero_days=days,run_share=count/total_runs if total_runs else np.nan,zero_day_share=days/total_days if total_days else np.nan,series_with_run=int(g['internal_runs_'+band].gt(0).sum()),series=len(g)))
    return pd.DataFrame(rows)

def quantiles(frame):
    rows=[]
    for cohort,g in cohorts(frame):
        for metric in ['cv','adi']:
            for q in np.linspace(0,1,101):rows.append(dict(cohort=cohort,metric=metric,quantile=q,value=g.loc[g.xyz_eligible,metric].quantile(q)))
    return pd.DataFrame(rows)

def explanations(transitions):
    t=transitions.copy()
    for name in ['cv','adi','revenue_rank','observed_days','revenue','zero_days']:
        t[name+'_delta']=t[name]-t[name+'_baseline']
    descriptions=[];categories=[]
    for r in t.itertuples():
        tags=[];parts=[]
        if r.abc_changed:
            if r.abc_thresholds_changed:
                tags.append('ABC threshold sensitivity');parts.append('ABC cutoff changed; a new ABC label does not by itself imply a new revenue rank.')
            else:
                tags.append('Revenue-share sensitivity');parts.append('ABC label changes with eligible revenue shares in the alternative observation window.')
        if r.revenue_rank_changed:
            tags.append('Revenue-ranking sensitivity');parts.append('Revenue rank changes within the original 90-series portfolio (rank delta = alternative minus baseline; negative means a higher rank).')
        if r.cv_class_changed:
            tags.append('CV sensitivity');parts.append('CV-only band changes under '+('different CV cutoffs.' if r.cv_thresholds_changed else 'different observed-day coverage.'))
        if r.adi_rule_triggered_changed or r.adi_override_effective_changed:
            tags.append('ADI intermittency override');parts.append('ADI trigger or its effective override changes; an effective override changes a non-Z CV label to Z.')
        if r.observation_coverage_changed:
            tags.append('Observation coverage');parts.append('Observation dates/counts differ; comparisons describe window sensitivity, not a forecast.')
        if r.possible_lifecycle_boundary_influence:
            tags.append('Possible lifecycle-boundary influence');parts.append('The positive-sales-span window excludes boundary zeros. This retrospective proxy does not establish listing dates or availability.')
        if r.eligibility_changed:tags.append('Eligibility change');parts.append('Classification eligibility changes; inspect exclusions before comparing labels.')
        if sum([r.abc_changed,r.cv_class_changed,(r.adi_rule_triggered_changed or r.adi_override_effective_changed),r.eligibility_changed])>1 and r.segment_changed:
            tags.append('Multiple interacting changes');parts.append('Multiple components move together; unique causal attribution is not identified.')
        if not r.segment_changed:parts.append('Final label is stable for this comparison; numerical metrics can still change.')
        descriptions.append(' '.join(parts) or 'Observed label change; inspect rule components.')
        categories.append('; '.join(tags) or 'No observed label change')
    t['explanation']=descriptions;t['explanation_categories']=categories
    t['lifecycle_counts_scope']='Baseline full-history counts; not alternative-window counts'
    t['cohort']='Selected sample';t['limitations']=LIMITS+' Window/rank comparisons use the original 90-series portfolio. Lifecycle boundaries are retrospective proxies.'
    return t

def window_summary(t):
    rows=[]
    for scenario,g in t[t.scenario.isin(['baseline','last365','positive_span','priced_days'])].groupby('scenario',sort=False):
        rows.append(dict(scenario=scenario,cohort='Selected sample',series=len(g),changed=int(g.segment_changed.sum()),xyz_changed=int(g.xyz_changed.sum()),abc_changed=int(g.abc_changed.sum()),changed_share=float(g.segment_changed.mean()) if len(g) else np.nan,eligible=int(g.classification_eligible.sum()),observed_days=int(g.observed_days.sum()),limitations='Population window sensitivity unavailable; sample ranks retain full 90-series portfolio context.'))
    return pd.DataFrame(rows)

def transition_matrix(t):
    return pd.crosstab(t.segment_baseline,t.segment).reindex(index=['AX','AY','AZ','BX','BY','BZ','CX','CY','CZ','U'],columns=['AX','AY','AZ','BX','BY','BZ','CX','CY','CZ','U'],fill_value=0).fillna(0).astype(int)

def export_csv(frame,scope_description):
    result=frame.copy();result['export_scope']=scope_description
    result['evidence_limitations']=LIMITS
    return result.to_csv(index=False).encode('utf-8')
