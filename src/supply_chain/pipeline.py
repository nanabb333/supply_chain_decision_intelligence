"""Two-pass bounded-memory CSV ingestion; DuckDB joins and Parquet persistence."""
from pathlib import Path
import hashlib
import json
import tempfile
import re
import os
import numpy as np
import pandas as pd
import duckdb
from .segmentation import classify
from .compatibility import calendar_days

IDS = ['id','item_id','dept_id','cat_id','store_id','state_id']
FILES = ['sales_train_validation.csv', 'calendar.csv', 'sell_prices.csv']


def require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing: raise ValueError(f'{label}: missing columns {sorted(missing)}')


def unique(frame, key, label):
    if frame[key].isna().any().any() or frame.duplicated(key).any():
        raise ValueError(f'{label}: null or duplicate key {key}')


def sales_chunks(path):
    for chunk in pd.read_csv(path, chunksize=512, dtype={c:'string' for c in IDS}):
        if 'id' not in chunk and {'item_id','store_id'} <= set(chunk):
            chunk['id'] = chunk.item_id + '_' + chunk.store_id + '_validation'
        require(chunk, IDS, 'sales')
        days = [c for c in chunk if re.fullmatch(r'd_\d+', c)]
        if not days: raise ValueError('No sales day columns')
        numbers = sorted(int(d[2:]) for d in days)
        if numbers != list(range(1, max(numbers)+1)): raise ValueError('Sales day columns must be contiguous from d_1')
        for column, regex in [('item_id',r'(FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+'), ('store_id',r'(CA|TX|WI)_\d+')]:
            if not chunk[column].str.fullmatch(regex).fillna(False).all(): raise ValueError(f'Invalid {column}')
        if chunk[IDS].isna().any().any(): raise ValueError('Null product identifier')
        if not (chunk.item_id.str.rsplit('_', n=1).str[0] == chunk.dept_id).all(): raise ValueError('Department mismatch')
        if not (chunk.dept_id.str.rsplit('_', n=1).str[0] == chunk.cat_id).all(): raise ValueError('Category mismatch')
        if not (chunk.store_id.str.split('_').str[0] == chunk.state_id).all(): raise ValueError('State mismatch')
        if not (chunk.id == chunk.item_id + '_' + chunk.store_id + '_validation').all(): raise ValueError('Sales id mismatch')
        vals = chunk[days].apply(pd.to_numeric, errors='raise')
        observed = vals.to_numpy(dtype=float)
        valid = np.isnan(observed) | (np.isfinite(observed) & (observed >= 0) & (observed == np.floor(observed)))
        if not valid.all(): raise ValueError('Sales must be nonnegative finite integers or missing')
        chunk[days] = vals
        yield chunk, days


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def run(raw, output, n=60, seed=42, synthetic=False):
    raw, output = Path(raw), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory is not empty; use a new versioned output to preserve provenance')
    receipt_path = raw/'source_receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipt:
        for name in FILES:
            if fingerprint(raw/name) != receipt['files'][name]['sha256']:
                raise ValueError(f'Source checksum mismatch: {name}')
    if n != 0 and not 30 <= n <= 100: raise ValueError('Use 30–100 series or 0 for all')
    for name in FILES:
        if not (raw/name).is_file(): raise FileNotFoundError(f'Missing {raw/name}; see docs/DATA.md')
    if (raw/'SYNTHETIC_FIXTURE.json').exists() and not synthetic:
        raise ValueError('Synthetic fixture requires --synthetic flag')
    metadata=[]
    raw_missing=0; raw_zero=0; raw_units=0; raw_max=0
    for chunk, days in sales_chunks(raw/FILES[0]):
        raw_missing += int(chunk[days].isna().sum().sum())
        raw_zero += int(chunk[days].eq(0).sum().sum())
        raw_units += float(chunk[days].sum().sum())
        raw_max = max(raw_max, float(chunk[days].max().max()))
        meta=chunk[IDS].copy()
        meta['mean_sales']=chunk[days].mean(axis=1)
        meta['zero_share']=chunk[days].eq(0).mean(axis=1)
        meta['profile']=np.where(meta.zero_share >= .5, 'intermittent', np.where(meta.mean_sales >= 5, 'higher-volume','lower-volume'))
        metadata.append(meta)
    meta=pd.concat(metadata, ignore_index=True)
    unique(meta, ['item_id','store_id'], 'sales')
    unique(meta, ['id'], 'sales')
    # Stable stratified round robin: shuffle within store/category/profile, then take one per stratum per round.
    selected=meta.sort_values(['item_id','store_id']).sample(frac=1, random_state=seed)
    selected['round']=selected.groupby(['store_id','cat_id','profile']).cumcount()
    selected=selected.sort_values(['round','store_id','cat_id','profile','item_id'])
    selected=selected.head(n) if n else selected
    chosen=set(selected.id)
    cal=pd.read_csv(raw/FILES[1], dtype={'d':'string'})
    cal = calendar_days(cal)
    require(cal,['d','date','wm_yr_wk'],'calendar')
    unique(cal,['d'],'calendar'); unique(cal,['date'],'calendar')
    cal['date']=pd.to_datetime(cal.date, errors='raise')
    if cal.date.isna().any() or not cal.d.str.fullmatch(r'd_\d+').all(): raise ValueError('Invalid calendar date/day')
    cal['day_number']=cal.d.str[2:].astype(int)
    cal=cal.sort_values('day_number')
    if not cal.date.diff().dropna().eq(pd.Timedelta(days=1)).all() or not cal.day_number.diff().dropna().eq(1).all():
        raise ValueError('Calendar day/date mapping is not contiguous')
    if not set(days).issubset(set(cal.d)): raise ValueError('Calendar missing sales days')
    cal['wm_yr_wk']=pd.to_numeric(cal.wm_yr_wk, errors='raise')
    if cal.wm_yr_wk.isna().any() or not np.equal(cal.wm_yr_wk, np.floor(cal.wm_yr_wk)).all(): raise ValueError('Invalid calendar week')
    cal['wm_yr_wk']=cal.wm_yr_wk.astype('int64')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage=Path(tmp)
        parts=stage/'parts'; parts.mkdir()
        source_units=0.; source_missing=0; count=0
        for chunk, days in sales_chunks(raw/FILES[0]):
            subset=chunk[chunk.id.isin(chosen)]
            if subset.empty: continue
            source_units += subset[days].sum().sum()
            source_missing += int(subset[days].isna().sum().sum())
            long=subset.melt(id_vars=IDS, value_vars=days, var_name='d', value_name='units')
            long['units']=long.units.astype('Int64')
            long.to_parquet(parts/f'{count}.parquet',index=False); count+=1
        con=duckdb.connect(str(stage/'analytics.duckdb'))
        try:
            con.execute("SET memory_limit='1GB'")
            con.execute('CREATE TABLE dim_calendar AS SELECT d, cast(date as DATE) date, wm_yr_wk FROM cal')
            con.execute('CREATE TABLE dim_series AS SELECT id, item_id, dept_id, cat_id, store_id, state_id, profile FROM selected')
            pricefile=str(raw/FILES[2])
            con.execute("CREATE TEMP TABLE raw_prices AS SELECT * FROM read_csv(?, all_varchar=true)",[pricefile])
            cols={r[0] for r in con.execute('DESCRIBE raw_prices').fetchall()}
            if not {'item_id','store_id','wm_yr_wk','sell_price'} <= cols: raise ValueError('Price schema missing columns')
            bad=con.execute("""SELECT count(*) FROM raw_prices WHERE item_id IS NULL OR store_id IS NULL
              OR NOT regexp_full_match(item_id, '(FOODS|HOBBIES|HOUSEHOLD)_[0-9]+_[0-9]+')
              OR NOT regexp_full_match(store_id, '(CA|TX|WI)_[0-9]+')
              OR try_cast(wm_yr_wk AS DOUBLE) IS NULL OR NOT isfinite(try_cast(wm_yr_wk AS DOUBLE))
              OR try_cast(wm_yr_wk AS DOUBLE) != floor(try_cast(wm_yr_wk AS DOUBLE))
              OR (sell_price IS NOT NULL AND (try_cast(sell_price AS DOUBLE) IS NULL
              OR NOT isfinite(try_cast(sell_price AS DOUBLE)) OR try_cast(sell_price AS DOUBLE)<=0))""").fetchone()[0]
            if bad: raise ValueError(f'Invalid price keys/values: {bad}')
            dup=con.execute('SELECT count(*) FROM (SELECT item_id, store_id, cast(wm_yr_wk AS BIGINT) FROM raw_prices GROUP BY ALL HAVING count(*)>1)').fetchone()[0]
            if dup: raise ValueError('Duplicate price key')
            con.execute('''CREATE TABLE weekly_prices AS SELECT p.item_id,p.store_id,cast(p.wm_yr_wk AS BIGINT) wm_yr_wk,
                cast(sell_price AS DOUBLE) sell_price FROM raw_prices p JOIN dim_series s USING(item_id,store_id)''')
            con.execute('''CREATE TABLE fact_sales AS SELECT item_id,store_id,d,units FROM read_parquet(?)''',[str(parts/'*.parquet')])
            con.execute('''CREATE VIEW daily_sales AS SELECT s.*, f.d,c.date,c.wm_yr_wk,f.units,p.sell_price,
                CASE WHEN f.units=0 THEN 0.0 ELSE f.units*p.sell_price END revenue,
                f.units IS NULL missing_sales, p.sell_price IS NULL missing_price,
                f.units>0 AND p.sell_price IS NULL unpriced_positive_sale
                FROM fact_sales f JOIN dim_series s USING(item_id,store_id)
                JOIN dim_calendar c USING(d) LEFT JOIN weekly_prices p USING(item_id,store_id,wm_yr_wk)''')
            summary=con.execute('''SELECT item_id,store_id,cat_id, sum(units) units,sum(revenue) revenue,
                count(units) observed_days,count(*)-count(units) missing_sales,count(*) FILTER(WHERE units>0) positive_days,
                count(*) FILTER(WHERE unpriced_positive_sale) unpriced_positive_days,
                avg(units) mean_units,stddev_pop(units) std_units FROM daily_sales GROUP BY ALL''').df()
            segments=classify(summary)
            con.execute('CREATE TABLE sku_segments AS SELECT * FROM segments')
            totals=con.execute('''SELECT count(*),sum(units),sum(revenue),count(*) FILTER(WHERE missing_sales),
                count(*) FILTER(WHERE missing_price),count(*) FILTER(WHERE unpriced_positive_sale),min(date),max(date),
                count(*) FILTER(WHERE units=0) FROM daily_sales''').fetchone()
            if totals[0] != len(selected)*len(days) or (totals[1] or 0) != source_units or totals[3] != source_missing:
                raise ValueError('Source-to-database reconciliation failed')
            if con.execute('SELECT count(*) FROM (SELECT item_id,store_id,d FROM fact_sales GROUP BY ALL HAVING count(*)>1)').fetchone()[0]:
                raise ValueError('Duplicate daily key')
            report={'source_kind':'synthetic_fixture' if synthetic else ('M5_official_archive' if receipt and receipt.get('source_kind')=='M5_official_archive' else 'M5_user_supplied'), 'series':len(selected),'rows':totals[0],
                'units':totals[1], 'known_price_revenue':totals[2], 'missing_sales':totals[3], 'missing_price_rows':totals[4],
                'unpriced_positive_days':totals[5], 'date_start':str(totals[6]),'date_end':str(totals[7]),'zero_sales_rows':totals[8],
                'seed':seed,'source_units':source_units,'source_reconciliation':True,
                'files':{name: {'sha256':fingerprint(raw/name),'bytes':(raw/name).stat().st_size} for name in FILES}}
            report['raw_profile']={'series':len(meta),'sales_days':len(days),'missing_sales':raw_missing,'zero_sales':raw_zero,'units':raw_units,'max_daily_units':raw_max,'stores':int(meta.store_id.nunique()),'items':int(meta.item_id.nunique()),'categories':int(meta.cat_id.nunique()),'departments':int(meta.dept_id.nunique()),'calendar_rows':len(cal),'price_rows':con.execute('SELECT count(*) FROM raw_prices').fetchone()[0],'price_nulls':con.execute('SELECT count(*) FROM raw_prices WHERE sell_price IS NULL').fetchone()[0],'duplicate_sales_keys':0,'duplicate_price_keys':0,'duplicate_calendar_keys':0}
            report['source_receipt']=receipt
            report['transformations']=['Derived missing id as item_id + store_id + validation suffix','Derived missing d from date minus 2011-01-29 + 1; official guide p.5','No imputation of sales or prices']
            con.execute('CREATE TABLE run_metadata AS SELECT ? source_kind, ? seed',[report['source_kind'],seed])
            dest=str(stage/'daily_sales.parquet').replace("'","''")
            con.execute(f"COPY daily_sales TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            segments.to_parquet(stage/'segments.parquet',index=False)
            selected.to_csv(stage/'selection.csv',index=False)
            (stage/'quality_report.json').write_text(json.dumps(report,indent=2))
        finally: con.close()
        output.mkdir(parents=True,exist_ok=True)
        for file in stage.iterdir():
            if file.is_file(): os.replace(file, output/file.name)
    return report
