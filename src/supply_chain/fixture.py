"""Synthetic test data ONLY. Never presented as Walmart observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def generate(path, n=60, days=365, seed=42):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.glob('*.csv')):
        raise ValueError('Fixture directory already contains CSVs; use a new directory')
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2015-01-01', periods=days)
    cal = pd.DataFrame({'d': [f'd_{i+1}' for i in range(days)], 'date': dates, 'wm_yr_wk': np.arange(days)//7 + 1000})
    rows, prices = [], []
    for i in range(n):
        cat = ['FOODS', 'HOBBIES', 'HOUSEHOLD'][i % 3]
        dept = cat + '_1'
        item = f'{dept}_{i+1:03d}'
        store = ['CA_1','TX_1','WI_1'][i % 3]
        pattern = i % 5
        values = rng.poisson((i % 9 + 1) * np.where(dates.dayofweek >= 5, 1.4, 1.0)).astype(float)
        if pattern == 0: values[:] = 4
        if pattern == 1: values[rng.random(days) < .8] = 0
        if pattern == 2: values *= np.where(np.arange(days) > days//2, 2, 1)
        if i == 0: values[:] = 0
        if i == 1: values[30] = np.nan
        row = dict(id=f'{item}_{store}_validation', item_id=item, dept_id=dept, cat_id=cat, store_id=store, state_id=store[:2])
        row.update({f'd_{j+1}': value for j, value in enumerate(values)})
        rows.append(row)
        for week in cal.wm_yr_wk.unique():
            if i == 2 and week == 1005: continue
            prices.append(dict(item_id=item, store_id=store, wm_yr_wk=week, sell_price=round(1.5 + i*.2 + (week % 3)*.1, 2)))
    pd.DataFrame(rows).to_csv(path/'sales_train_validation.csv', index=False)
    cal.to_csv(path/'calendar.csv', index=False)
    pd.DataFrame(prices).to_csv(path/'sell_prices.csv', index=False)
    (path/'SYNTHETIC_FIXTURE.json').write_text(json.dumps({'synthetic': True, 'seed': seed, 'series':n, 'days':days}, indent=2))
