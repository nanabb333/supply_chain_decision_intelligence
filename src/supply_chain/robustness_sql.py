"""Independent DuckDB expressions for Phase 1.6 reconciliation.
Does not call production lifecycle/window/classifier helpers.
"""
import numpy as np
import pandas as pd

LIFECYCLE_SQL='''
WITH bounds AS (
 SELECT item_id,store_id,min(date) FILTER(WHERE units>0) first_positive_date,
 max(date) FILTER(WHERE units>0) last_positive_date FROM daily_sales GROUP BY ALL
), runs AS (
 SELECT *,sum(CASE WHEN units=0 THEN 0 ELSE 1 END) OVER(PARTITION BY item_id,store_id ORDER BY date) run_id
 FROM daily_sales
), lengths AS (
 SELECT item_id,store_id,run_id,count(*) n FROM runs WHERE units=0 GROUP BY ALL
), longest AS (SELECT item_id,store_id,max(n) longest_zero_run FROM lengths GROUP BY ALL)
SELECT d.item_id,d.store_id,b.first_positive_date,b.last_positive_date,
 count(*) total_calendar_days,count(units) total_observed_days,count(*)-count(units) missing_sales_days,
 count(*) FILTER(WHERE units>0) positive_sales_days,count(*) FILTER(WHERE units=0) zero_sales_days,
 CASE WHEN b.first_positive_date IS NOT NULL THEN count(*) FILTER(WHERE units=0 AND date<b.first_positive_date) END leading_zero_days,
 CASE WHEN b.first_positive_date IS NOT NULL THEN count(*) FILTER(WHERE units=0 AND date>b.last_positive_date) END trailing_zero_days,
 CASE WHEN b.first_positive_date IS NOT NULL THEN count(*) FILTER(WHERE units=0 AND date BETWEEN b.first_positive_date AND b.last_positive_date) END internal_zero_days,
 coalesce(l.longest_zero_run,0) longest_zero_run,avg(units) full_mean_units,stddev_pop(units) full_std_units,
 stddev_pop(units)/nullif(avg(units),0) full_cv,
 count(units)::DOUBLE/nullif(count(*) FILTER(WHERE units>0),0) full_adi,
 date_diff('day',b.first_positive_date,b.last_positive_date)+1 positive_span_days,
 CASE WHEN b.first_positive_date IS NOT NULL THEN count(units) FILTER(WHERE date<b.first_positive_date OR date>b.last_positive_date) END outside_span_observed_days,
 CASE WHEN b.first_positive_date IS NOT NULL THEN count(units) FILTER(WHERE date<b.first_positive_date OR date>b.last_positive_date)::DOUBLE/nullif(count(units),0) END outside_span_observed_share
FROM daily_sales d JOIN bounds b USING(item_id,store_id) LEFT JOIN longest l USING(item_id,store_id)
GROUP BY d.item_id,d.store_id,b.first_positive_date,b.last_positive_date,l.longest_zero_run
ORDER BY d.item_id,d.store_id
'''

SCENARIO_SQL='''
WITH bounds AS (
 SELECT item_id,store_id,min(date) FILTER(WHERE units>0) first_pos,max(date) FILTER(WHERE units>0) last_pos
 FROM daily_sales GROUP BY ALL
), chosen AS (
 SELECT d.* FROM daily_sales d JOIN bounds b USING(item_id,store_id)
 WHERE $window='full'
 OR ($window='last365' AND date>=(SELECT max(date)-INTERVAL 364 DAY FROM daily_sales))
 OR ($window='positive_span' AND date BETWEEN b.first_pos AND b.last_pos)
 OR ($window='priced_days' AND sell_price IS NOT NULL)
), agg AS (
 SELECT item_id,store_id,count(*) calendar_rows,count(units) observed_days,
 count(*)-count(units) missing_sales,count(*) FILTER(WHERE units>0) positive_days,
 count(*) FILTER(WHERE units=0) zero_days,
 count(*) FILTER(WHERE units>0 AND sell_price IS NULL) unpriced_positive_days,
 min(date) window_start,max(date) window_end,sum(units) units,sum(revenue) revenue,
 avg(units) mean_units,stddev_pop(units) std_units
 FROM chosen GROUP BY ALL
), filled AS (
 SELECT b.item_id,b.store_id,a.* EXCLUDE(item_id,store_id,calendar_rows,observed_days,missing_sales,positive_days,zero_days,unpriced_positive_days),
 coalesce(calendar_rows,0) calendar_rows,coalesce(observed_days,0) observed_days,
 coalesce(missing_sales,0) missing_sales,coalesce(positive_days,0) positive_days,
 coalesce(zero_days,0) zero_days,coalesce(unpriced_positive_days,0) unpriced_positive_days
 FROM bounds b LEFT JOIN agg a USING(item_id,store_id)
), eligibility AS (
 SELECT *, coalesce(missing_sales=0 AND unpriced_positive_days=0 AND revenue>0,false) abc_eligible,
 coalesce(missing_sales=0 AND observed_days>=28 AND mean_units>0,false) xyz_eligible,
 std_units/nullif(mean_units,0) cv, observed_days::DOUBLE/nullif(positive_days,0) adi
 FROM filled
), ranked AS (
 SELECT *,sum(revenue) FILTER(WHERE abc_eligible) OVER() eligible_total,
 coalesce(sum(CASE WHEN abc_eligible THEN revenue ELSE 0 END) OVER(ORDER BY revenue DESC NULLS LAST,item_id,store_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) prior_revenue,
 sum(revenue) OVER() portfolio_total
 FROM eligibility
), classes AS (
 SELECT *,CASE WHEN NOT abc_eligible THEN 'U' WHEN round(prior_revenue/eligible_total,12)<round($a,12) THEN 'A' WHEN round(prior_revenue/eligible_total,12)<round($b,12) THEN 'B' ELSE 'C' END abc,
 CASE WHEN NOT xyz_eligible THEN 'U' WHEN round(cv,12)<=round($x,12) THEN 'X' WHEN round(cv,12)<=round($y,12) THEN 'Y' ELSE 'Z' END cv_class,
 coalesce(xyz_eligible AND round(adi,12)>=round($intermittent,12),false) adi_rule_triggered
 FROM ranked
), final AS (
 SELECT *,CASE WHEN adi_rule_triggered THEN 'Z' ELSE cv_class END xyz,
 adi_rule_triggered AND cv_class<>'Z' adi_override_effective,
 CASE WHEN abc_eligible THEN revenue/eligible_total END eligible_revenue_share,
 revenue/portfolio_total portfolio_revenue_share,
 positive_days::DOUBLE/nullif(observed_days,0) positive_sales_frequency
 FROM classes
)
SELECT *,abc||xyz segment,abc_eligible AND xyz_eligible classification_eligible FROM final ORDER BY item_id,store_id
'''

def assert_matching(actual,expected,columns):
    a=actual.sort_values(['item_id','store_id']).reset_index(drop=True)
    e=expected.sort_values(['item_id','store_id']).reset_index(drop=True)
    assert a[['item_id','store_id']].equals(e[['item_id','store_id']])
    for col in columns:
        if 'date' in col or col in ['window_start','window_end']:
            assert pd.to_datetime(a[col]).astype('datetime64[ns]').equals(pd.to_datetime(e[col]).astype('datetime64[ns]')),col
        elif pd.api.types.is_numeric_dtype(a[col]) or pd.api.types.is_bool_dtype(a[col]) or pd.api.types.is_numeric_dtype(e[col]):
            np.testing.assert_allclose(a[col].to_numpy(dtype=float,na_value=np.nan),e[col].to_numpy(dtype=float,na_value=np.nan),equal_nan=True,rtol=1e-10,atol=1e-8,err_msg=col)
        else:assert a[col].fillna('').equals(e[col].fillna('')),col
