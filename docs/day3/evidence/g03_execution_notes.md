# G03 Execution Notes

Date: 2026-04-21

## G03.1 Split manifests are reusable and leakage-safe

- Deterministic split manifest:
  - `artifacts/reports/day3/split_manifest.json`
- Tabular split export:
  - `artifacts/reports/day3/split_manifest.csv`
- Labeling and split validation summary:
  - `artifacts/reports/day3/labeling_summary.json`
- Feature leakage report:
  - `artifacts/reports/day3/feature_leakage_report.json`

## G03.2 MVP training pipeline runs end-to-end

- Feature build summary:
  - `artifacts/reports/day3/feature_builder_summary.json`
- Training metrics summary:
  - `artifacts/reports/day3/training_metrics.json`
- Focused Day 3 tests:
  - `artifacts/reports/day3/day3_tests.xml`

## G03.3 Training artifacts are reproducible from saved config and manifest

- Training manifest:
  - `artifacts/reports/day3/training_manifest.json`
- Model artifacts:
  - `artifacts/models/day3/linear_model_calibrated.pkl`
  - `artifacts/models/day3/tree_model_calibrated.pkl`
  - `artifacts/models/day3/late_fusion_config.json`
  - `artifacts/models/day3/feature_columns.json`
  - `artifacts/models/day3/scaler_metadata.json`

## Consolidated Result

- Day 3 pipeline tasks T08, T09, and T10 execute end-to-end.
- Day 3 focused test suite passed: 8 passed.
- G03 gate criteria satisfied with linked evidence artifacts.