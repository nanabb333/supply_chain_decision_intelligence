"""Validated read-only input / versioned diagnostic output configuration.
Existing Phase 1.5 entrypoints and defaults are deliberately unchanged.
"""
from pathlib import Path
import json,math,re
from .robustness import DEFAULT_THRESHOLDS

def _unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError(f'Duplicate configuration key: {k}')
        d[k]=v
    return d

def overlaps(a,b):return a==b or a in b.parents or b in a.parents

def load_config(path):
    path=Path(path).resolve();c=json.loads(path.read_text(),object_pairs_hook=_unique)
    fields={'root','raw_directory','processed_directory','duckdb_path','analytical_output','report_output','scenarios'}
    if set(c)!=fields:raise ValueError('Configuration fields missing or unknown')
    root=(path.parent/c['root']).resolve();c['root']=str(root)
    for k in fields-{'root','scenarios'}:
        if not isinstance(c[k],str) or not c[k]:raise ValueError(f'Invalid path {k}')
        c[k]=str((root/c[k]).resolve())
    raw,proc,db=map(Path,[c['raw_directory'],c['processed_directory'],c['duckdb_path']])
    if not raw.is_dir() or not proc.is_dir() or not db.is_file():raise ValueError('Configured inputs do not exist')
    if db!=proc/'analytics.duckdb':raise ValueError('Ambiguous database versus processed version')
    outputs=[Path(c[k]) for k in ['analytical_output','report_output']]
    if overlaps(*outputs):raise ValueError('Analytical/report output directories overlap')
    protected=[raw,proc,root/'reports/phase1_5',root/'docs/phase1_5',root/'app',root/'src',root/'scripts',root/'tests']
    if any(overlaps(o,p) for o in outputs for p in protected):raise ValueError('Output overlaps protected inputs/artifacts')
    specs=c['scenarios'];names=[]
    if not isinstance(specs,list) or not specs:raise ValueError('Scenarios required')
    for s in specs:
        if set(s)-{'name','window','thresholds'} or not {'name','window'}<=set(s):raise ValueError('Invalid scenario fields')
        if not re.fullmatch('[a-z][a-z0-9_]*',s['name']):raise ValueError('Invalid scenario name')
        if s['window'] not in ['full','last365','positive_span','priced_days']:raise ValueError('Invalid window')
        if set(s.get('thresholds',{}))-set(DEFAULT_THRESHOLDS):raise ValueError('Unknown threshold')
        t=DEFAULT_THRESHOLDS|s.get('thresholds',{})
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in t.values()):raise ValueError('Thresholds must be finite numbers')
        if not (0<t['a']<t['b']<1 and 0<=t['x']<t['y'] and t['intermittent']>=1):raise ValueError('Invalid thresholds')
        if s['name']=='baseline' and (s['window']!='full' or t!=DEFAULT_THRESHOLDS):raise ValueError('Baseline definition must match accepted rules')
        names.append(s['name'])
    if len(set(names))!=len(names) or names.count('baseline')!=1:raise ValueError('Unique scenarios and one baseline required')
    return c

def fresh_target(config,checkpoint):
    if checkpoint not in ['lifecycle','robustness']:raise ValueError('Unsupported checkpoint')
    target=Path(config['analytical_output'])/checkpoint
    if target.exists():raise FileExistsError(f'Output already exists; choose a new run version: {target}')
    return target
