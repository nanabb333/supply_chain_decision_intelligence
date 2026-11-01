-- CTE, calendar JOIN, time aggregation, LAG window; incomplete periods are flagged.
WITH monthly AS (
 SELECT date_trunc('month', c.date) AS month_start, sum(f.units) units,
 count(*)-count(f.units) missing_observations
 FROM fact_sales f JOIN dim_calendar c USING(d) GROUP BY 1
), compared AS (
 SELECT *, lag(units) OVER(ORDER BY month_start) previous_units,
 lag(missing_observations) OVER(ORDER BY month_start) previous_missing FROM monthly
)
SELECT *, CASE WHEN missing_observations=0 AND previous_missing=0
 THEN units/nullif(previous_units,0)-1 END unit_growth FROM compared ORDER BY month_start;
