"""UI-independent analytical queries; all filter values are bound parameters."""
import duckdb
import pandas as pd


def connect(database):
    return duckdb.connect(str(database), read_only=True)


def load_sales(database, stores=None, categories=None, items=None, start=None, end=None, segments=None):
    clauses, params=[], []
    for name, values in [('store_id',stores),('cat_id',categories),('item_id',items),('segment',segments)]:
        if values:
            column = 's.segment' if name == 'segment' else f'd.{name}'
            clauses.append(f'{column} IN ({",".join("?" for _ in values)})'); params.extend(values)
    if start is not None: clauses.append('date >= ?'); params.append(start)
    if end is not None: clauses.append('date <= ?'); params.append(end)
    where=' AND '.join(clauses) or 'TRUE'
    with connect(database) as con:
        return con.execute(f'SELECT d.*,s.segment FROM daily_sales d JOIN sku_segments s USING(item_id,store_id) WHERE {where} ORDER BY date,item_id,store_id',params).df()


def get_sales_kpis(frame, previous=None):
    units=frame.units.sum(min_count=1)
    revenue=frame.revenue.sum(min_count=1)
    priced=frame.loc[frame.sell_price.notna() & frame.units.notna(), 'units'].sum(min_count=1)
    daily=frame.groupby('date').units.sum(min_count=1)
    incomplete=frame.groupby('date').missing_sales.any()
    daily=daily.mask(incomplete)
    growth=None
    if previous is not None and not previous.empty and not frame.empty and not frame.missing_sales.any() and not previous.missing_sales.any():
        prev=previous.units.sum(min_count=1)
        if pd.notna(prev) and prev > 0: growth=float(units/prev-1)
    return dict(units=units,revenue=revenue,asp=revenue/priced if pd.notna(priced) and priced>0 else None,
        growth=growth,volatility=daily.std(ddof=0) if len(daily.dropna()) else None,
        missing_sales=int(frame.missing_sales.sum()),unpriced_positive_days=int(frame.unpriced_positive_sale.sum()))


def daily_series(frame):
    g=frame.groupby('date').agg(units=('units',lambda x:x.sum(min_count=1)),revenue=('revenue',lambda x:x.sum(min_count=1)),
        missing_sales=('missing_sales','any'), unpriced_positive_sale=('unpriced_positive_sale','any'))
    g.loc[g.missing_sales,'units']=float('nan')
    g.loc[g.missing_sales | g.unpriced_positive_sale,'revenue']=float('nan')
    return g.reset_index()
