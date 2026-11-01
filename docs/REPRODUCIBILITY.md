# Reproducibility and execution boundaries

## Installation

Use Python 3.12 (the validation host used 3.12.14 ARM64). From this checkout:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

Requirements list direct dependencies, not a hash-locked universal environment. A fresh-machine installation has not been verified. The process-tree watchdog needs OS permission to inspect processes; restricted sandboxes may reject its test. No CUDA dependency is required. POSIX process-group handling is used; Windows portability is unverified.

## Synthetic demonstration — no competition data

```sh
python examples/synthetic/demo.py --output outputs/synthetic-demo
SUPPLY_CHAIN_DB=outputs/synthetic-demo/processed/analytics.duckdb python -m streamlit run app/streamlit_app.py
```

Choose a fresh output directory on every invocation. This generates artificial sales, calendar and prices, executes CSV → Parquet → DuckDB, scores an artificial 28-day forecast, and independently reconciles three artificial 21-day inventory episodes. It does not train M1/M3 or reproduce empirical research findings. The runnable demo is the original sales dashboard, not a synthetic replacement for the five-view research portfolio.

## Obtain M5 independently

Visit the [official competition data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), sign in, read and accept applicable terms if eligible, and download through Kaggle. No credentials should be stored in this repository. Place these files locally:

```text
data/raw/m5-organizers-2026-10-13/
  sales_train_validation.csv
  calendar.csv
  sell_prices.csv
```

The historical directory name is a code convention, not a claim that new downloads have the accepted research hashes. Keep `sales_train_evaluation.csv` separate; do not substitute it silently.

Validate and ingest a descriptive sample (this command performs schema, duplicate, key, nonnegative-value and join checks):

```sh
python -m supply_chain.cli ingest --raw data/raw/m5-organizers-2026-10-13 --output data/processed/local-m5 --series 90
SUPPLY_CHAIN_DB=data/processed/local-m5/analytics.duckdb python -m streamlit run app/streamlit_app.py
```

This fresh sample is not guaranteed to be the accepted 90-series retrospective cohort. See [data definitions](DATA.md) and [KPIs](KPIS.md). Do not use full-history descriptive segments as past-only forecasting features.

## Forecasting and inventory code

`src/supply_chain/baselines.py`, `xyz_features.py` and `xyz_stream.py` contain development evaluation, past-only features and streamed ridge calculations. `final_evaluation.py` enforces the original frozen final protocol. The following historical coordinator entry points are included for inspection:

```sh
python scripts/run_week2_baselines.py --output outputs/new-baselines --pilot
python scripts/run_xyz_stream.py --origin 1801 --series 256 --name new-development
python scripts/run_week3_c2_pilot.py
python scripts/run_week3_d1.py
python scripts/run_week3_d3.py
```

**These are not a verified end-to-end public reproduction recipe.** The baseline command requires the independently acquired data. Streamed/model and inventory coordinators additionally require original manifests, compute-amendment records, matched predictions and frozen one-time authorization artifacts that are not distributed. They must fail if those inputs are absent. Never manufacture approvals, relax hashes, or rerun the original final holdout to make them work. No empirical coordinator was executed for this release. The synthetic demo above is the supported inventory execution command.

## Five-view research dashboard

```sh
python -m streamlit run app/portfolio_app.py
```

The source covers Executive Overview, Demand & Data Diagnostics, Forecasting Performance, Inventory Policy Comparison, and Robustness & Sensitivity. It requires accepted evidence under `reports/`, which is excluded. Source availability does not mean the research dashboard is functional in a clean public checkout. Use the synthetic sales dashboard for a runnable demonstration. `research_tests/` similarly holds evidence-dependent tests for inspection; `python -m pytest -q` runs the portable synthetic suite only.
