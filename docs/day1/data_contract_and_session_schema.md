# Day 1 Data Contract and Session Schema

Version: v1.0.0  
Date: 2026-04-21

## T03.1 Canonical Columns, Units, and Sampling Assumptions

### EEG

- Required columns:
  - `session_id`
  - `timestamp_utc`
  - `eeg_tp9`, `eeg_af7`, `eeg_af8`, `eeg_tp10`, `eeg_aux`
- Units:
  - EEG channels: microvolts
  - `timestamp_utc`: ISO-8601 UTC
- Sampling assumption: 256 Hz

### Optical

- Required columns:
  - `session_id`
  - `timestamp_utc`
  - `optics_lo_nir`, `optics_ro_nir`, `optics_lo_ir`, `optics_ro_ir`
  - `optics_lo_red`, `optics_ro_red`, `optics_lo_amb`, `optics_ro_amb`
- Units:
  - Optical channels: arbitrary units (a.u.)
  - `timestamp_utc`: ISO-8601 UTC
- Sampling assumption: 64 Hz

### IMU

- Required columns:
  - `session_id`
  - `timestamp_utc`
  - `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`
- Units:
  - Acceleration: g
  - Gyroscope: deg/s
  - `timestamp_utc`: ISO-8601 UTC
- Sampling assumption: 52 Hz

## T03.2 Mandatory Metadata Fields

- `session_id`
- `task_type` (allowed: `baseline_rest`, `sustained_attention`, `n_back`, `code_comprehension`)
- `start_timestamp_utc`
- `end_timestamp_utc`
- `subjective_fatigue_score` in [0, 10]
- Optional performance metrics captured as fields in the schema and may be null for unsupported tasks:
  - `reaction_time_ms`
  - `accuracy`

## T03.3 Missing-Value and Fallback Policy

- Global timestamp rule: no null timestamps allowed in any modality.
- Modality max missing ratio thresholds:
  - EEG: <= 0.20
  - Optical: <= 0.35
  - IMU: <= 0.25
- Fallback behavior:
  - EEG above threshold: reject session for training/live update.
  - Optical above threshold: continue with down-weighted optical contribution.
  - IMU above threshold: continue with disabled motion gating and degraded confidence flag.

## T03.4 Schema Validation Enforcement

- Canonical machine-readable contract: `configs/data_contract.json`
- Fail-fast validator implementation: `src/neuroguage/schema_validator.py`
- CLI wrapper: `scripts/validate_schema.py`
- Validator evidence outputs are stored under `artifacts/reports/day1`.
