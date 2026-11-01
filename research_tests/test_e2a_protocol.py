"""Frozen-protocol and access-guard checks only; no final model execution."""
import importlib.util
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit_e2a',R/'scripts/audit_week2_e2a.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_independent_frozen_protocol_audit():
 assert len(m.audit())>=40

def test_frozen_protocol_has_no_execution_authorization():
 c=json.loads((R/'reports/week2/checkpoint-e2a-v1/frozen-protocol.json').read_text())
 assert c['status']=='FROZEN_PROTOCOL_ONLY'
 assert c['release_required'].startswith('Separate E2-B authorization')
 assert not c['final_fit_executed'] and not c['predictions_generated'] and not c['holdout_evaluated']

def test_required_specifications_present():
 for name in ['E2A_PROTOCOL_FREEZE','FROZEN_MODEL_SPECIFICATIONS','FINAL_COMPARISON_PLAN','HOLDOUT_INTEGRITY_AUDIT','FINAL_EVALUATION_EXECUTION_PLAN','PRACTICAL_SIGNIFICANCE_RULES']:
  assert (R/f'reports/week2/checkpoint-e2a-v1/{name}.md').stat().st_size>500
