"""Read-only final artifact acceptance checks; never invokes fitting or scoring."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from supply_chain.final_evaluation import sha,require_prediction_manifest,verify_protocol
R=Path(__file__).resolve().parents[1];O=R/'reports/week2/checkpoint-e2b-v1'

def test_committed_stages_and_release_order():
 verify_protocol(R);require_prediction_manifest(O)
 for name in ['COHORT_COMPLETE.json','FIT_COMPLETE.json','PREDICT_COMPLETE.json','SCORE_COMPLETE.json','ANALYSIS_COMPLETE.json']:
  r=json.loads((O/name).read_text());assert r['status']=='PASS'
  assert all(sha(O/k)==v for k,v in r['files'].items())
 assert (O/'FINAL_PREDICTION_MANIFEST.json').stat().st_mtime_ns < (O/'score-STARTED.json').stat().st_mtime_ns
 assert not list(O.glob('*FAILED*')) and not list(O.glob('*MEMORY_STOP*'))

def test_independent_metrics_and_matched_counts():
 a=pd.read_csv(O/'FINAL_MODEL_COMPARISON.csv').set_index('model').sort_index();b=pd.read_csv(O/'independent-model-metrics.csv').set_index('model').sort_index()
 np.testing.assert_allclose(a[['macro_mae','bias']],b[['macro_mae','bias']],rtol=1e-10,atol=1e-12)
 assert a.series.eq(30490).all() and a.target_observations.eq(853720).all()
 p=pd.read_csv(O/'FINAL_PAIRED_COMPARISONS.csv');assert (p.improved+p.worsened+p.tied).eq(30490).all()
 for r in p.itertuples():
  l,right=r.comparison.split('_minus_');assert np.isclose(r.delta_mae,a.loc[l,'macro_mae']-a.loc[right,'macro_mae'])

def test_frozen_practical_and_degenerate_uncertainty():
 c=verify_protocol(R);p=pd.read_csv(O/'FINAL_PRACTICAL_THRESHOLDS.csv')
 assert sorted(p[p.scale=='relative_pct'].threshold.unique())==c['relative_reduction_grid_pct']
 assert sorted(p[p.scale=='absolute_units_per_series_day'].threshold.unique())==c['absolute_reduction_grid_units']
 assert p.passes.eq(p.reduction>=p.threshold).all()
 u=pd.read_csv(O/'FINAL_UNCERTAINTY.csv');deg=u[u.method=='calendar_28d'];assert len(deg)==3 and deg.p025.isna().all() and deg.p975.isna().all() and deg.draws.eq(0).all()
 assert u[u.method!='calendar_28d'].draws.eq(2000).all()
