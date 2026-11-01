"""Decode selected rows from the SAME byte buffers whose accepted SHA256 was verified.

Preparation/validation only; no engine import, scoring or inventory decisions.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib, io, json, csv, math
import pyarrow as pa
import pyarrow.parquet as pq
from .inventory_policy import ForecastInput, MatchedInputs, MODELS, quantity

B = 'reports/week3/checkpoint-b-v1/'


def verified_bytes(root, path, manifest):
    root=Path(root).resolve(); p=(root/path).resolve()
    if not p.is_relative_to(root) or path not in manifest:
        raise ValueError('Path outside protected allowlist')
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest[path]:
        raise ValueError(f'Source hash mismatch: {path}')
    return raw


@dataclass(frozen=True)
class BoundSeries:
    forecasts: MatchedInputs
    demand: tuple[int, ...]
    demand_source: str
    demand_source_sha256: str
    data_kind: str = 'M5_SOURCE_VERIFICATION_ONLY'


def decode_selected(raw, keys, *, kind):
    columns=['item_id','store_id','origin']
    if kind=='MA28': columns+=['model']
    columns += [f'{"y" if kind=="demand" else "f"}_{j}' for j in range(1,22 if kind=='demand' else 29)]
    table=pq.read_table(pa.BufferReader(raw),columns=columns)
    items=table['item_id'].to_pylist(); stores=table['store_id'].to_pylist()
    models=table['model'].to_pylist() if kind=='MA28' else None
    indices=[n for n,key in enumerate(zip(items,stores)) if key in keys and (models is None or models[n]=='ma28')]
    return table.take(pa.array(indices,type=pa.int64())).to_pylist()


def load_frozen_pilot(root, protected):
    """Return frozen128-series inputs; DOES NOT execute a pilot or compute outcomes."""
    receipt=json.loads(verified_bytes(root,B+'PILOT_INPUT_MANIFEST.json',protected))
    selected_raw=verified_bytes(root,B+'pilot-series.csv',protected)
    if hashlib.sha256(selected_raw).hexdigest()!=receipt['selected_ids_sha256']:
        raise ValueError('Selected IDs do not match frozen receipt')
    selection=list(csv.DictReader(io.StringIO(selected_raw.decode())))
    keys=tuple((r['item_id'],r['store_id']) for r in selection)
    if len(keys)!=128 or len(set(keys))!=128 or receipt['origin']!=1801 or receipt['episode_days']!=21 or receipt['policies']!=list(MODELS) or receipt['scenarios']!=['primary'] or receipt['execution_authorized'] is not False:
        raise ValueError('Frozen pilot scope changed')
    source_hashes=receipt['protected_source_hashes']
    if any(protected.get(p)!=h for p,h in source_hashes.items()):
        raise ValueError('Receipt hashes differ from accepted protected manifest')
    import re
    rows={m:{} for m in (*MODELS,'demand')}; used=[]
    for path in sorted(source_hashes):
        if re.fullmatch(r'reports/week2/checkpoint-b-v1/full-v1/cohorts/1801-[0-9]+\.parquet',path):
            continue  # B's frozen IDs already hash-bound to its accepted cohort selection.
        if re.fullmatch(r'reports/week2/checkpoint-b-v1/full-v1/forecasts/1801-[0-9]+\.parquet',path): kind='MA28'
        elif re.fullmatch(r'reports/week2/checkpoint-b-v1/full-v1/targets/1801-[0-9]+\.parquet',path): kind='demand'
        elif path=='reports/week2/checkpoint-c-v2/population-1801/M1-c0-forecasts.parquet': kind='M1'
        elif path=='reports/week2/checkpoint-c-v2/population-1801/M3-c0-forecasts.parquet': kind='M3'
        else: raise ValueError('Unaccepted model/origin/source path')
        raw=verified_bytes(root,path,protected)
        decoded=decode_selected(raw,set(keys),kind=kind)
        used.append(dict(path=path,sha256=source_hashes[path],selected_rows=len(decoded),kind=kind))
        for row in decoded:
            key=(row['item_id'],row['store_id'])
            if row['origin']!=1801 or key in rows[kind]:
                raise ValueError('Duplicate key or wrong fixed origin')
            prefix='y' if kind=='demand' else 'f'
            values=tuple(row[f'{prefix}_{j}'] for j in range(1,22 if kind=='demand' else 29))
            for v in values:
                quantity(v,'saved units')
                if kind=='demand' and v!=int(v): raise ValueError('Fractional observed units')
            rows[kind][key]=(values,path)
    if any(set(m)!=set(keys) for m in rows.values()): raise ValueError('Missing matched forecast/demand keys')
    bound=[]
    for item,store in keys:
        forecasts=[]
        for model in MODELS:
            values,path=rows[model][(item,store)]
            forecasts.append(ForecastInput(item,store,1801,model,values,path,protected[path],1801))
        demand,path=rows['demand'][(item,store)]
        bound.append(BoundSeries(MatchedInputs(tuple(forecasts)).validate(protected),tuple(int(v) for v in demand),path,protected[path]))
    return tuple(bound), tuple(used)
