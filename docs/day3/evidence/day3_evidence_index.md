# Day 3 Evidence Index

Version: v1.0.0  
Date: 2026-04-21

## T08 Unified Feature Builder

### T08.1 Join EEG, optical, and IMU quality features

- Implementation:
  - `src/neuroguage/feature_builder.py`
  - `scripts/day3_build_features.py`
- Input fixtures:
  - `tests/fixtures/day3/eeg_window_features_sessions.csv`
  - `tests/fixtures/day3/optical_imu_window_features_sessions.csv`
- Output artifact:
  - `artifacts/features/day3/unified_feature_matrix.csv`

### T08.2 Train-only normalization fit and val/test transform

- Normalization strategy:
  - train-only z-score fit with split-specific transform
- Output artifact:
  - `artifacts/features/day3/unified_feature_matrix_normalized.csv`
- Normalization metadata:
  - `artifacts/reports/day3/normalization_metadata.json`

### T08.3 Persist feature set with dataset and split compatibility metadata

- Dataset tags included in persisted matrix:
  - `dataset_version`
  - `feature_schema_version`
  - `split_compatibility_marker`
- Summary artifact:
  - `artifacts/reports/day3/feature_builder_summary.json`

### T08.4 Leakage checks for target contamination and session overlap

- Leakage report artifact:
  - `artifacts/reports/day3/feature_leakage_report.json`

## T09 Labeling and Session-Held-Out Split Utility

### T09.1 Generate weak labels from task phase, subjective ratings, and performance drift

- Implementation:
  - `src/neuroguage/labeling.py`
  - `scripts/day3_build_labels_and_splits.py`
- Annotation fixture:
  - `tests/fixtures/day3/session_annotations.csv`
- Output labeled matrix:
  - `artifacts/features/day3/labeled_feature_matrix.csv`

### T09.2 Optional temporal smoothing

- Configured smoothing:
  - enabled with rolling window = 3
- Evidence:
  - `artifacts/reports/day3/labeling_summary.json`

### T09.3 Session-level split isolation with deterministic seeding

- Split policy:
  - deterministic seed = 42
  - session-held-out train/val/test assignment
- Evidence artifacts:
  - `artifacts/reports/day3/split_manifest.json`
  - `artifacts/reports/day3/split_manifest.csv`

### T09.4 Export split manifests for repeatable experiments

- Exported split manifest artifacts:
  - `artifacts/reports/day3/split_manifest.json`
  - `artifacts/reports/day3/split_manifest.csv`

## T10 MVP Model Training and Calibration

### T10.1 Linear baseline sweep (Logistic Regression or LDA)

- Implementation:
  - `src/neuroguage/training.py`
  - `scripts/day3_train_mvp.py`
- Selected linear model:
  - LDA with solver `lsqr`, shrinkage `auto`

### T10.2 Tree baseline and validation comparison

- Selected tree model:
  - Random Forest with `n_estimators=220`, `max_depth=8`, `min_samples_leaf=2`
- Metrics summary:
  - `artifacts/reports/day3/training_metrics.json`

### T10.3 Calibrated late-fusion scoring

- Calibrated model artifacts:
  - `artifacts/models/day3/linear_model_calibrated.pkl`
  - `artifacts/models/day3/tree_model_calibrated.pkl`
- Fusion config:
  - `artifacts/models/day3/late_fusion_config.json`

### T10.4 Persist models, scalers, hyperparameters, and training metadata

- Persisted training artifacts:
  - `artifacts/models/day3/feature_columns.json`
  - `artifacts/models/day3/scaler_metadata.json`
  - `artifacts/reports/day3/training_manifest.json`
  - `artifacts/reports/day3/training_metrics.json`

## G03 Day 3 Exit Gate

- Gate execution summary:
  - `docs/day3/evidence/g03_execution_notes.md`
- Focused Day 3 test report:
  - `artifacts/reports/day3/day3_tests.xml`
- Day 3 test files:
  - `tests/test_feature_builder_day3.py`
  - `tests/test_labeling_day3.py`
  - `tests/test_training_day3.py`