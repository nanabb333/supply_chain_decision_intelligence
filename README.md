# Supply Chain Decision Intelligence

**From Demand Forecasting to Inventory Decisions**

An independent business analytics project examining how retail demand forecasting, inventory policies, and operating assumptions influence service–cost tradeoffs.

**Project status:** Completed local research; documentation-only portfolio candidate. Not yet published.

## Business Problem

Does a more accurate demand forecast necessarily lead to a better inventory decision?

Forecasting models are often evaluated by prediction accuracy. However, businesses ultimately need to decide how much inventory to hold, when to replenish, and how to balance customer service against operating costs.

This project connects predictive analytics with inventory decision-making to investigate that relationship.

## Key Findings

**1. More complex forecasting models did not necessarily perform better.**

A simple 28-day moving-average benchmark achieved the lowest macro mean absolute error among the three evaluated approaches in the accepted final evaluation.

**2. Higher simulated service levels came with tradeoffs.**

In the primary inventory scenario, an alternative forecasting policy achieved a higher simulated fill rate but also incurred higher hypothetical operating costs than the moving-average policy.

**3. Hypothetical cost rankings depended on operating assumptions.**

Sensitivity analysis showed that the policy with the lowest hypothetical cost could change when opening-inventory assumptions changed. This highlights the importance of evaluating operational robustness rather than selecting a policy based on a single scenario.

## Research Scope

| Component | Scope |
|---|---|
| Forecasting evaluation | 30,490 product-store time series |
| Retrospective demand diagnostics | 90 selected series |
| Inventory-policy simulation | 128 selected product-store series |
| Forecast horizon | 28 days |
| Inventory simulation | 21-day episode |
| Operating scenarios | 9 comparisons |
| Opening-inventory configurations | 3 methods |

The forecasting and inventory analyses use different evaluation populations and windows. Their results should not be interpreted as a matched causal comparison.

## Analytical Workflow

**Data Engineering → Demand Analysis → Forecasting → Inventory Simulation → Business Intelligence**

- **Data Engineering:** Processed historical retail sales, calendar, and pricing data using Python, Parquet, DuckDB, and SQL.
- **Demand Analysis:** Examined sales patterns and ABC–XYZ segmentation, including sensitivity to historical observation windows.
- **Forecasting:** Compared a moving-average benchmark with two feature-based forecasting approaches.
- **Inventory Simulation:** Evaluated three replenishment policies under hypothetical operating assumptions.
- **Sensitivity Analysis:** Compared policy outcomes across operating scenarios and alternative inventory initializations.
- **Business Intelligence:** Developed a five-view Streamlit dashboard and supporting business and technical reports for local review.

## Tools and Technologies

Python · SQL · DuckDB · Pandas · Parquet · Streamlit · Forecast Evaluation · Inventory Simulation · Scenario Analysis

## Research Limitations

Historical sales were used as a proxy for demand. Actual inventory availability, supplier lead times, and operating costs were not observed.

Inventory results are hypothetical simulations, not realized business savings or evidence of operational deployment.

The selected inventory cohort is not established as representative of the full forecasting population. The final forecast evaluation also had prior descriptive exposure and should not be characterized as a pristine blind test.

The project does not claim causal business improvement, production readiness, or verified fresh-machine reproducibility.

## Data Sources and Attribution

The project uses historical retail data from the M5 Forecasting Competition.

See [Data Sources and Attribution](DATA_SOURCES_AND_ATTRIBUTION.md) for dataset provenance and rights limitations.

No raw or processed datasets, research figures, model artifacts, or underlying code are included in this proposed documentation-only release.

## Author Contribution and AI-Assisted Development

The project owner directed the research scope, approved analytical designs and assumptions, and reviewed project decisions.

OpenAI Codex assisted with implementation, automation, debugging, testing, validation, and documentation.

The project represents independent research direction supported by AI-assisted development. It does not imply that every line of code was manually written or that an external independent audit was performed.

## Release Scope

This proposed portfolio release contains only the project README and the data attribution document.

The research pipeline, dashboard, datasets, figures, PDFs, and simulation outputs remain local.

Public release is pending final rights review and explicit author authorization.
