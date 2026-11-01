"""Execution guards tested without reading real observations or fitting models."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest

@pytest.fixture
def runner(monkeypatch):
 # Execution dependency lives in the accepted isolated model environment.
 if importlib.util.find_spec("threadpoolctl") is None:
  site=Path(__file__).resolve().parents[1]/"reports/week2/checkpoint-c-v1/model-env/lib/python3.12/site-packages"
  monkeypatch.syspath_prepend(str(site))
 p=Path(__file__).resolve().parents[1]/'scripts/run_xyz_stream.py';spec=importlib.util.spec_from_file_location('stream_runner_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_atomic_receipts_never_overwrite(tmp_path,runner):
 p=tmp_path/'receipt.json';runner.atomic_json(p,{'status':'PASS'})
 with pytest.raises(FileExistsError):runner.atomic_json(p,{'status':'FAIL'})
 assert json.loads(p.read_text())=={'status':'PASS'}

def test_completed_job_not_reexecuted(tmp_path,runner,monkeypatch):
 monkeypatch.setattr(runner,'C',tmp_path);(tmp_path/'done').mkdir();(tmp_path/'done/execution.json').write_text('{}')
 monkeypatch.setattr(runner.sys,'argv',['run','--origin','1801','--series','256','--name','done'])
 with pytest.raises(FileExistsError):runner.main()

def test_population_requires_resource_gate(tmp_path,runner,monkeypatch):
 monkeypatch.setattr(runner,'C',tmp_path);monkeypatch.setattr(runner.sys,'argv',['run','--origin','1801','--series','30490','--name','full'])
 with pytest.raises(FileNotFoundError):runner.main()
 assert not (tmp_path/'full').exists()

def test_final_origin_rejected_before_output(tmp_path,runner,monkeypatch):
 monkeypatch.setattr(runner,'C',tmp_path);monkeypatch.setattr(runner.sys,'argv',['run','--origin','1885','--series','256','--name','forbidden'])
 with pytest.raises(ValueError):runner.main()
 assert not (tmp_path/'forbidden').exists()

def test_memory_guard_stops_at_conservative_limit(tmp_path,runner,monkeypatch):
 class Stop:
  def wait(self,t):return False
 monkeypatch.setattr(runner,'rss',lambda:runner.LIMIT)
 def halt(code):raise RuntimeError(code)
 monkeypatch.setattr(runner.os,'_exit',halt)
 with pytest.raises(RuntimeError,match='90'):runner.monitor(tmp_path,Stop())
 assert json.loads((tmp_path/'MEMORY_STOP.json').read_text())['threshold']==3*1024**3
