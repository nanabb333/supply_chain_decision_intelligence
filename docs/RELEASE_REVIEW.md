# Release review — 2026-11-01

## Scope and exclusions

Original Python modules, SQL, Streamlit entry points, hypothetical scenario configurations, six historical coordinators, synthetic tests, and methodological documentation are included. The synthetic example calls the original analytical functions; no empirical research was rerun. Original source modules and accepted analytical results were not modified.

| Excluded material | Reason |
|---|---|
| Raw M5 CSVs, processed Parquet/DuckDB, calendars, prices, row-level records | Redistribution permission not verified; never copied into release |
| Forecast arrays, model artifacts, cohort lists, simulation ledgers, cached evidence | Could disclose or reconstruct competition-derived records; excluded |
| Final Business Report (8 pages), Technical Appendix (3 pages), five empirical figures, detailed aggregate tables | Original author-generated outputs, but applicable derived-content permission remains unresolved; no asset-level legal clearance claimed |
| Private logs, receipts, credentials, absolute local paths, virtual environments | Privacy, security and machine-specific content; excluded |
| Historical report/publication writers and artifact-generating audit scripts | Depend on excluded evidence and mutate historical output locations; unnecessary for public computational interfaces |
| One historical PDF-publication test | Contains an absolute personal runtime path and requires excluded PDF evidence; excluded rather than edited |
| Original lock snapshots | Machine-specific snapshots; replaced by explicit direct dependency ranges, not claimed to be a universal lockfile |

The existing qualitative README narrative is retained and distinguished from results reproducible here. Existing public commit history is preserved. Its pre-existing author contact metadata is not rewritten; new release commits use a GitHub noreply identity. No claim of comprehensive legal clearance or a pristine history is made.

## Validation

- Initial staging run: 261 passed, 11 failed, 12 errors. Failures traced to a missing KPI document, absent empirical dashboard evidence, and OS sandbox process-monitoring restrictions; the log is retained locally.
- KPI documentation was supplied. Evidence-dependent tests are retained unchanged under `research_tests/`, explicitly outside the portable suite. No assertions were weakened.
- Final portable suite: **263 passed** in 3.38 seconds, including synthetic pipeline/SQL reconciliation, temporal and inventory guards, independent accounting and Streamlit AppTest. The watchdog test ran with OS process access.
- Synthetic demo completed CSV → Parquet → DuckDB, synthetic 28-day forecast scoring, three artificial 21-day policy episodes and independent event reconciliation.
- Dependency check reported no broken installed requirements. Threadpoolctl and setuptools were absent from the research environment and installed separately for release validation; the accepted environment was not modified. Dependency ranges are declared, but a clean environment installation and all-platform compatibility are not verified.
- Syntax, staged-file, bounded secret/privacy, restricted-file, relative-link and Git-history content scans are recorded in the local release audit. Pattern scans are not a guarantee of absence of every possible secret.
- No raw M5 data, empirical models, ledger, PDF or figure is included. No empirical forecasting, final scoring, inventory evaluation or report rebuilding occurred.

## What runs and what does not

The synthetic sales dashboard is tested. The five-view portfolio and other research dashboard sources require excluded evidence and are included for inspection; they are not advertised as runnable without those artifacts. Frozen historical runners retain their guards and require unavailable manifests; they are not a verified fresh-machine reproduction path. See [execution guide](REPRODUCIBILITY.md).

## Rights and dependency boundaries

The official Kaggle rules page yielded no readable text during the 2026-11-01 review. Unverified permissions are not replaced by a research-use disclaimer. No vendor implementation, winning competition solution, font, logo or binary dependency is bundled. Python dependencies install separately under their own licenses. No LICENSE was selected for the author's work and no blanket license covering third-party data is supplied. See [attribution](../DATA_SOURCES_AND_ATTRIBUTION.md).
