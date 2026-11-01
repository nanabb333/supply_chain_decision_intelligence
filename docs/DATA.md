# Data dictionary and synthetic inputs

Acquire competition inputs independently under the applicable terms; see [execution guide](REPRODUCIBILITY.md) and [attribution](../DATA_SOURCES_AND_ATTRIBUTION.md). No source records or empirical acceptance outputs are included.

Sales are validated in chunks, selected series are transformed from wide daily columns, calendar joins use `d`, and prices join on `(item_id, store_id, wm_yr_wk)`. Duplicate keys, invalid dates, negative quantities and inconsistent hierarchy fail validation. Missing sales and prices are not silently imputed. A descriptive sample is not representative by construction. Full-history segmentation must not be used as a forecasting feature without recomputing it at the historical cutoff.

## Data dictionary

| Field | Grain/type | Source / derivation |
|---|---|---|
| id | series/string | Original ID when supplied; otherwise derived item_store_validation |
| item_id | series/string | Product identifier from sales; product may appear in multiple stores |
| dept_id, cat_id | series/string | Department/category from sales; validated hierarchy |
| store_id, state_id | series/string | Store/state from sales; validated prefix |
| profile | series/string | Reproducible sample stratum from full-history mean/zero share |
| d | day/string | Wide day-column name, transformed without filling nulls |
| date | day/SQL DATE | calendar.csv mapping, consecutive daily dates |
| wm_yr_wk | week/integer | M5 retail-week join key, not ISO week |
| units | item-store-day/nullable BIGINT | Original observed count, melted from sales CSV |
| sell_price | item-store-week/nullable DOUBLE | sell_prices.csv observed weekly average selling price, USD |
| revenue | item-store-day/nullable DOUBLE | units × sell_price; exactly zero for observed zero sales |
| missing_sales | day/boolean | units IS NULL |
| missing_price | day/boolean | price absent or null after left join |
| unpriced_positive_sale | day/nullable boolean | units > 0 and sell_price missing |
| abc, xyz, segment | series/string | Full-history classification; U identifies unclassifiable evidence |
| cv, adi | series/nullable DOUBLE | Population std/mean; observed days/positive days |
| eligible_revenue_share, cumulative_share | series/nullable DOUBLE | Share among ABC-eligible series only |
| abc_reason, xyz_reason | series/string | Eligibility/decision explanation |
| source_kind, seed | run/string, integer | Explicit provenance and sampling seed |

SQL tables: `dim_series`, `dim_calendar`, `weekly_prices`, `fact_sales`, `sku_segments`, `run_metadata`. Analytical view: `daily_sales`. The daily Parquet export has equivalent joined fields. Quality report includes source file hashes, dates, counts, source reconciliation, zero counts, missing sales and missing prices. `segments.parquet` is reusable, subject to cutoff-specific recomputation for future backtests.

## Synthetic fixture

`fixture` produces 60 generated series, 365 days (2015-01-01–2015-12-31), seed 42. It includes all-zero, stable, intermittent, changing-volume profiles, one missing observation and a missing price week. These are **test scenarios only**, not observed M5 data or business insights. A marker file and explicit `--synthetic` flag prevent silent fixture use. Never remove the marker to imply real observations.

