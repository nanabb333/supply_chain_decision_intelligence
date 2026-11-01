# KPI dictionary

All sales measures describe recorded sales, not unconstrained customer demand. Dollar values are USD selling-price estimates, not audited net receipts. The selected item–store portfolio and selected dates define the denominator unless stated otherwise.

| KPI | Formula / denominator | Missingness and interpretation |
|---|---|---|
| Units sold | Sum of recorded daily units | Null observations remain null; totals may be partial. Zero means observed zero sales, not proof of zero demand. All-null totals are unavailable. |
| Known-price sales revenue | Sum(units × observed weekly sell_price) | Positive units without a price contribute null, not zero. Zero observed units contribute zero even if price missing. No price imputation; no tax, discount or returns reconciliation. |
| Average selling price | Known-price revenue / units with valid observed prices | Unit weighted. Excludes unpriced units; undefined when priced-unit denominator is zero. |
| Unit sales growth | Current units / preceding equal-length period units − 1 | Same item/store/category/segment filters. N/A when baseline zero, prior period outside history, or either period has missing sales. |
| Category sales contribution | Category known-price revenue / selected portfolio known-price revenue | Partial if unpriced sales exist. Changes denominator with filters; zero denominator is unavailable. |
| Sales volatility | Population standard deviation of complete daily portfolio unit totals | Not averaged SKU CV. Days with missing component sales are excluded. Segmentation uses per-series daily CV instead. |
| Weekly/monthly sales patterns | Average complete daily portfolio units grouped by weekday or month-of-year | Controls for differing day counts. Descriptive pattern, not causal seasonality proof or a forecast; incomplete days excluded. |
| Top/bottom SKUs | Rank by recorded units within selected dates and population | Item–store grain, lexical ties; missing days shown alongside. Partial observations may distort rank. |

## Definitions only — inventory data unavailable

- **Inventory turnover:** period cost of goods sold / average inventory value on a matching cost basis. Requires inventory valuations and COGS.
- **Days of supply:** available on-hand units / expected daily demand rate, or days until projected depletion from a varying forecast. Requires current stock and demand assumptions; undefined at zero rate.
- **Fill rate:** demand units fulfilled immediately / total requested demand units. Recorded sales cannot establish unmet demand.
- **Stockout rate:** defined here as stockout SKU-days / eligible SKU-days. Requires observed availability records; distinguish from order-based service level and fill rate.

These four metrics are deliberately not shown as observations.

## ABC–XYZ policy

Classification unit: item–store series, over the entire ingested history. Full-history segments stay fixed while dashboard sales dates change. This descriptive module does not forecast; the separate forecasting modules compute cutoff-specific segments to avoid hindsight leakage.

**ABC:** rank complete series by known-price revenue descending, breaking ties by item/store ID. Eligible series require no missing sales, no positive-sales days missing prices, and positive total revenue. Compute cumulative share among eligible revenue only. Class A has prior cumulative share below 80%; B below 95%; C the rest. Thus the SKU crossing a threshold remains in the preceding class. A single dominant SKU is A. Zero-revenue or incomplete series are U, with a reason. Eligible contributions sum to 100%; all-series known revenue reconciles to the daily fact, including excluded-series known revenue.

**XYZ:** daily CV = population standard deviation / mean units, including observed zero days. X ≤0.5, Y ≤1.0, Z >1.0. ADI = observed days / days with positive sales. ADI ≥1.32 overrides CV to Z to highlight intermittent sales. All-zero, missing-sales, or <28-observation series are U. These thresholds are illustrative heuristics; they do not imply normally distributed demand or optimal inventory policies.

**Sensitivity:** dashboard sliders recompute classifications across the full ingested portfolio and show changed labels without overwriting persisted baseline segments. Higher ABC cutoffs retain more SKUs in A/B; higher CV cutoffs tend to move series toward X/Y, except ADI-overridden series. Selection and date horizon also affect classification.

**Operational meaning:** A deserves greatest revenue-focused review priority, B moderate, C lower. X supports testing stable-sales baselines; Y requires seasonal/variation review; Z warrants intermittent-sales and availability investigation. U requires evidence repair. None is a replenishment recommendation.

