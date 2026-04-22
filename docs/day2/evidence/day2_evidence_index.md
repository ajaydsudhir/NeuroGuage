# Day 2 Evidence Index

Version: v1.0.0  
Date: 2026-04-21

## T04 Multimodal Ingestion and Timeline Alignment

### T04.1 Ingestion adapters for EEG, optical, and IMU

- Implementation:
  - `src/neuroguage/ingestion.py`
  - `scripts/day2_sync_session.py`
- Tests:
  - `tests/test_ingestion_alignment.py`

### T04.2 Timestamp normalization to common clock

- Implementation:
  - `src/neuroguage/ingestion.py` (`normalize_modality_timestamps`)
- Evidence artifact:
  - `artifacts/reports/day2/alignment_diagnostics.json`

### T04.3 Alignment diagnostics for drift, dropouts, packet delay

- Implementation:
  - `src/neuroguage/ingestion.py` (`_compute_modality_diagnostics`)
- Evidence artifact:
  - `artifacts/reports/day2/alignment_diagnostics.json`

### T04.4 Synchronized session artifact and validation summary

- Synchronized session artifact:
  - `artifacts/processed/day2/synchronized_session.csv`
- Validation summary:
  - `artifacts/reports/day2/synchronization_validation_summary.json`

## T05 Windowing and Buffering Layer

### T05.1 Fixed-size sliding windows for fast EEG updates

- Implementation:
  - `src/neuroguage/windowing.py` (`build_windowing_layers`)
- Artifact:
  - `artifacts/processed/day2/eeg_fast_windows.csv`

### T05.2 Slower rolling windows for optical context features

- Implementation:
  - `src/neuroguage/windowing.py` (`build_windowing_layers`)
- Artifact:
  - `artifacts/processed/day2/optical_slow_windows.csv`

### T05.3 Preserve mapping from each window to absolute timeline index

- Evidence columns in both window artifacts:
  - `absolute_timeline_start_idx`
  - `absolute_timeline_end_idx`

### T05.4 Boundary tests for start, end, missing packets

- Tests:
  - `tests/test_windowing.py`
- Summary artifact:
  - `artifacts/reports/day2/windowing_summary.json`

## T06 EEG Preprocessing and Features

### T06.1 Filtering and artifact-aware segmentation

- Implementation:
  - `src/neuroguage/eeg_features.py`
  - `scripts/day2_eeg_features.py`
- Tests:
  - `tests/test_eeg_features.py`

### T06.2 Relative band powers and key ratios

- Feature artifact:
  - `artifacts/processed/day2/eeg_window_features.csv`

### T06.3 Numerical safeguards for unstable spectra and clipping

- Implementation safeguards include:
  - NaN sanitization
  - value clipping
  - low-sample and low-power fallback guards

### T06.4 Persist per-window EEG features with quality flags

- Persisted artifact:
  - `artifacts/processed/day2/eeg_window_features.csv`
- Summary:
  - `artifacts/reports/day2/eeg_feature_summary.json`

## T07 Optical and IMU Quality Features

### T07.1 Optical normalization and trend/slope/mean-change features

- Implementation:
  - `src/neuroguage/optical_imu_features.py`
  - `scripts/day2_optical_imu_features.py`

### T07.2 IMU-derived motion artifact scoring per window

- Feature columns:
  - `imu_motion_score`

### T07.3 Confidence scores for quality-based down-weighting

- Feature columns:
  - `confidence_score`
  - `downweight_factor`

### T07.4 Persist per-window optical and quality feature tables

- Persisted artifact:
  - `artifacts/processed/day2/optical_imu_window_features.csv`
- Summary:
  - `artifacts/reports/day2/optical_imu_feature_summary.json`

## G02 Day 2 Exit Gate

- Gate execution summary:
  - `docs/day2/evidence/g02_execution_notes.md`
- Consolidated test report:
  - `artifacts/reports/day2/day2_tests.xml`
