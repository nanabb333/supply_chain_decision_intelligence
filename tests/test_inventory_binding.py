import csv,hashlib,io,json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from supply_chain.inventory_binding import B,verified_bytes,load_frozen_pilot,decode_selected

@pytest.fixture
def source_tree(tmp_path):
    manifest={}
    def write(path,raw):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        manifest[path]=hashlib.sha256(raw).hexdigest()
    keys=[(f'SYNTHETIC_{i}','store') for i in range(128)]
    sources={}
    for kind in ('MA28','M1','M3','demand'):
        path=(f'reports/week2/checkpoint-c-v2/population-1801/{kind}-c0-forecasts.parquet' if kind in ('M1','M3') else
              f'reports/week2/checkpoint-b-v1/full-v1/{"targets" if kind=="demand" else "forecasts"}/1801-000.parquet')
        rows=[]
        for item,store in keys:
            row=dict(item_id=item,store_id=store,origin=1801)
            if kind=='MA28':row['model']='ma28'
            row.update({f'{"y" if kind=="demand" else "f"}_{j}':2. for j in range(1,29)})
            rows.append(row)
        if kind=='MA28':rows.append(dict(rows[0],model='naive',f_1=999.))
        sink=pa.BufferOutputStream();pq.write_table(pa.Table.from_pylist(rows),sink)
        write(path,sink.getvalue().to_pybytes());sources[path]=manifest[path]
    s=io.StringIO();w=csv.writer(s);w.writerow(['selection_rank','selection_sha256','item_id','store_id'])
    for n,(i,t) in enumerate(keys,1):w.writerow([n,hashlib.sha256(f'{i}|{t}'.encode()).hexdigest(),i,t])
    write(B+'pilot-series.csv',s.getvalue().encode())
    receipt=dict(origin=1801,episode_days=21,policies=['MA28','M1','M3'],scenarios=['primary'],execution_authorized=False,
                 selected_ids_sha256=manifest[B+'pilot-series.csv'],protected_source_hashes=sources)
    write(B+'PILOT_INPUT_MANIFEST.json',json.dumps(receipt).encode())
    return tmp_path,manifest,write,receipt


def test_hash_bound_decode_and_same_paths(source_tree):
    root,m,_,_=source_tree
    bound,used=load_frozen_pilot(root,m)
    assert len(bound)==128 and len(used)==4
    assert all(b.demand==(2,)*21 for b in bound)
    assert all(f.values==(2.,)*28 for b in bound for f in b.forecasts.forecasts)
    assert all(b.data_kind=='M5_SOURCE_VERIFICATION_ONLY' for b in bound)


def test_tamper_and_outside_path(source_tree):
    root,m,_,_=source_tree;p=next(p for p in m if 'forecasts/' in p)
    (root/p).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):load_frozen_pilot(root,m)
    with pytest.raises(ValueError):verified_bytes(root,'../escape',{'../escape':'a'*64})


def test_wrong_manifest_and_final_scope(source_tree):
    root,m,write,receipt=source_tree
    receipt['origin']=1885;write(B+'PILOT_INPUT_MANIFEST.json',json.dumps(receipt).encode())
    with pytest.raises(ValueError,match='scope'):load_frozen_pilot(root,m)


def test_missing_row_duplicate_and_bad_units(source_tree):
    root,m,write,receipt=source_tree
    path=next(p for p in m if 'M1-c0' in p)
    original=pq.read_table(root/path).to_pylist()
    for rows in (original[:-1],original+[original[0]], [dict(original[0],f_1=-1.)]+original[1:]):
        sink=pa.BufferOutputStream();pq.write_table(pa.Table.from_pylist(rows),sink)
        write(path,sink.getvalue().to_pybytes());receipt['protected_source_hashes'][path]=m[path]
        write(B+'PILOT_INPUT_MANIFEST.json',json.dumps(receipt).encode())
        with pytest.raises(ValueError):load_frozen_pilot(root,m)


def test_byte_snapshot_not_reopened(source_tree):
    root,m,_,_=source_tree;path=next(p for p in m if 'M3-c0' in p)
    raw=verified_bytes(root,path,m);(root/path).write_bytes(b'changed after read')
    rows=decode_selected(raw,{('SYNTHETIC_0','store')},kind='M3')
    assert len(rows)==1 and rows[0]['f_1']==2.
    with pytest.raises(ValueError):verified_bytes(root,path,m)
