# Architecture

## Core Source Modules

- `cli.py`
  - All command entry points (`run-live`, `train`, `evaluate`, `train-dual-head`, schema/env utilities).
- `pipeline.py`
  - Offline data path: sync, windows, features, labels, training, evaluation, schema helpers.
- `dual_head.py`
  - Dual-head model, weak-label derivation, train/load/predict utilities.
- `inference.py`
  - MVP late-fusion online risk inference.
- `guardrails.py`
  - Neuro-symbolic state machine and mismatch intervention logic.
- `live_app.py`
  - Live orchestration loop, calibration, artifact writing, dashboard snapshot generation.
- `muse_stream.py`
  - Muse/OpenMuse/LSL stream management, replay and decode helpers.
- `dashboard.py`
  - Matplotlib real-time UI.
- `lifecycle.py`
  - Demo, hardening, promotion, and release utility flows.

## Experiment Flow

1. **Train**
   - `scripts/train.py`
   - Optional dual-head: `scripts/train_dual_head.py --model-variant muse_demo|research_eegnet`
2. **Run**
   - `scripts/run_demo.py` for classroom defaults
   - `scripts/run_live.py` for explicit control
3. **Review**
   - `artifacts/reports/live/*_summary.json`
   - `artifacts/reports/live/*_session_log.csv`
   - mismatch and transition artifacts for neuro-symbolic behavior

## Variant Support

- **Demo variant**: `muse_demo` (lighter and classroom-friendly).
- **Research variant**: `research_eegnet` (higher-capacity dual-head architecture).

Both variants go through the same guardrail layer for intervention decisions.
