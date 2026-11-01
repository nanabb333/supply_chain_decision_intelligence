SELECT item_id,store_id,date,units,
 CASE WHEN count(units) OVER w=28 THEN avg(units) OVER w END trailing_28_day_mean
FROM daily_sales WINDOW w AS (
 PARTITION BY item_id,store_id ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
) ORDER BY item_id,store_id,date;
