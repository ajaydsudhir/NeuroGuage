# Pipeline Outline

## Goal
Detect cognitive fatigue risk in real time with Muse S Athena and trigger clear intervention states.

## Phase 1 — EEG Attention Signal
Mostly implemented in `src/neuroguage/pipeline.py` and `src/neuroguage/dual_head.py`.

- Ingest synchronized EEG windows.
- Extract EEG band/ratio/quality features.
- Produce attention-oriented signals used by both MVP and dual-head paths.

## Phase 2 — Optical/IMU Fatigue-Related Signal
Mostly implemented in `src/neuroguage/pipeline.py` and `src/neuroguage/dual_head.py`.

- Extract optical trend proxies and IMU motion quality.
- Build metabolic-readiness proxy features.
- Train/serve dual-head metabolic branch (`research_eegnet` or `muse_demo`).

## Phase 3 — Neuro-Symbolic Intervention
Implemented in `src/neuroguage/guardrails.py`.

- Map risk + mismatch conditions into GREEN/YELLOW/RED/BLACK states.
- Use persistence and hysteresis to reduce oscillation.
- Emit explicit transition rationale and artifacts.

## Runtime Orchestration
Implemented in `src/neuroguage/live_app.py`.

- Acquire Muse streams (`src/neuroguage/muse_stream.py`).
- Run inference (`src/neuroguage/inference.py`, optional `src/neuroguage/dual_head.py`).
- Apply calibration, guardrails, and dashboard rendering.
- Persist run artifacts for analysis and replay.
