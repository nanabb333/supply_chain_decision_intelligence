"""Reproducible item-store segmentation with explicit evidence eligibility."""
import numpy as np
import pandas as pd


def classify(summary: pd.DataFrame, a=.8, b=.95, x=.5, y=1., intermittent=1.32):
    if not (0 < a < b < 1 and 0 <= x < y and intermittent >= 1):
        raise ValueError('Invalid segmentation thresholds')
    s = summary.copy().sort_values(['revenue', 'item_id', 'store_id'], ascending=[False, True, True], na_position='last')
    s['abc'] = 'U'
    s['xyz'] = 'U'
    s['abc_reason'] = 'Missing sales or positive-sales price coverage'
    s['xyz_reason'] = 'Missing sales, insufficient history, or all-zero sales'
    valid = (s.missing_sales == 0) & (s.unpriced_positive_days == 0) & (s.revenue > 0)
    total = s.loc[valid, 'revenue'].sum()
    s['eligible_revenue_share'] = np.nan
    s['cumulative_share'] = np.nan
    if total > 0:
        shares = s.loc[valid, 'revenue'] / total
        prior = shares.cumsum() - shares
        s.loc[valid, 'eligible_revenue_share'] = shares
        s.loc[valid, 'cumulative_share'] = shares.cumsum()
        s.loc[valid, 'abc'] = np.select([prior < a, prior < b], ['A', 'B'], default='C')
        s.loc[valid, 'abc_reason'] = 'Complete sales and positive-sales price coverage'
    s.loc[(s.missing_sales == 0) & (s.units == 0), 'abc_reason'] = 'No positive revenue'
    s['cv'] = s.std_units / s.mean_units.replace(0, np.nan)
    s['adi'] = s.observed_days / s.positive_days.replace(0, np.nan)
    usable = (s.missing_sales == 0) & (s.observed_days >= 28) & (s.mean_units > 0)
    s.loc[usable, 'xyz'] = np.select([s.loc[usable, 'cv'] <= x, s.loc[usable, 'cv'] <= y], ['X', 'Y'], default='Z')
    s.loc[usable & (s.adi >= intermittent), 'xyz'] = 'Z'
    s.loc[usable, 'xyz_reason'] = 'Daily CV; intermittent series forced to Z'
    s['segment'] = s.abc + s.xyz
    return s.reset_index(drop=True)
