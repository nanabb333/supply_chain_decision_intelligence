WITH totals AS (
 SELECT cat_id,item_id,store_id,sum(revenue) known_revenue,
 count(*) FILTER(WHERE missing_sales OR unpriced_positive_sale) incomplete_days
 FROM daily_sales GROUP BY ALL
)
SELECT *, dense_rank() OVER(PARTITION BY cat_id ORDER BY known_revenue DESC NULLS LAST) category_rank
FROM totals ORDER BY cat_id,category_rank,item_id,store_id;
