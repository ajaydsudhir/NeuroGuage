# NeuroGauge Detailed Build TODO (PRD-Aligned, 7-Day Execution Plan)

## Comprehensive Analysis and Overview

- [x] O-01 Confirm phase-gated delivery policy (Depends on: None)
  - [x] O-01.1 Confirm required success target is AC-001 to AC-005 by end of Day 6.
  - [x] O-01.2 Confirm stretch target AC-006 to AC-009 is conditional on Day 6 gate pass.
- [x] O-02 Lock the critical-path sequence (Depends on: O-01)
  - [x] O-02.1 Validate sequence: contract -> ingestion -> features -> training -> inference -> guardrails -> dashboard -> demo.
  - [x] O-02.2 Ensure every downstream task references an upstream dependency from this sequence.
- [x] O-03 Define risk-first execution policy (Depends on: O-01)
  - [x] O-03.1 Prioritize controls for overfitting, artifact sensitivity, and latency instability before stretch tasks.
  - [x] O-03.2 Create daily risk review checkpoint at end of each day gate.
- [x] O-04 Define completion evidence standard (Depends on: O-02, O-03)
  - [x] O-04.1 Require each completed task to produce an artifact: script, config, report, or log.
  - [x] O-04.2 Require each day gate to include pass/fail evidence links.
- [x] O-05 Freeze demo envelope and timing budget (Depends on: O-01)
  - [x] O-05.1 Lock quick demo flow to 5 to 10 minutes.
  - [x] O-05.2 Track per-stage time budget and flag scope that violates class timing.
- [ ] O-06 Enforce dependency integrity with no orphan work (Depends on: O-02, O-04)
  - [ ] O-06.1 Validate each task has at least one upstream dependency, except O-01.
  - [ ] O-06.2 Validate each task feeds at least one downstream task or exit gate.
- [x] O-07 Confirm required runtime environment policy (Depends on: O-01)
  - [x] O-07.1 Execute all scripts and demos from Conda environment neuroguage.
  - [x] O-07.2 Record environment verification as part of Day 1 and Day 6 evidence.

## Day 1 - Scope Lock, Environment Baseline, and Data Contract

- [x] T01 Scope Freeze and Acceptance Gate Lock (Depends on: O-01, O-04)
  - [x] T01.1 Create working MVP acceptance checklist for AC-001 to AC-005. (Depends on: O-01)
  - [x] T01.2 Create separate stretch acceptance checklist for AC-006 to AC-009. (Depends on: O-01)
  - [x] T01.3 Document Day 6 go/no-go rule and decision owner. (Depends on: T01.1, T01.2)
  - [x] T01.4 Create risk log template with fields: issue, impact, owner, mitigation, status, review-date. (Depends on: O-03)
- [x] T02 Repository and Environment Baseline (Depends on: T01, O-07)
  - [x] T02.1 Create or verify directories for raw, processed, features, models, reports, and demo artifacts. (Depends on: T01)
  - [x] T02.2 Validate environment reproducibility files and pin critical package versions. (Depends on: O-07)
  - [x] T02.3 Add executable entry points for train, evaluate, run-live, and replay workflows. (Depends on: T02.1)
  - [x] T02.4 Add shared logging configuration for console and file diagnostics. (Depends on: T02.3)
- [x] T03 Data Contract and Session Schema (Depends on: T02, O-02)
  - [x] T03.1 Define canonical columns, units, and sampling assumptions for EEG, optical, and IMU channels. (Depends on: T02)
  - [x] T03.2 Define mandatory metadata fields: task type, timestamps, subjective fatigue score, optional performance metrics. (Depends on: T03.1)
  - [x] T03.3 Define missing-value handling and modality fallback behavior per channel group. (Depends on: T03.1)
  - [x] T03.4 Implement schema validation checks that fail early on malformed sessions. (Depends on: T03.2, T03.3)

- [x] G01 Day 1 Exit Gate (Depends on: T01, T02, T03)
  - [x] G01.1 Scope and acceptance checklists are finalized and versioned.
  - [x] G01.2 Environment and entry-point scripts run from neuroguage.
  - [x] G01.3 Schema validator passes valid input and rejects malformed records.

## Day 2 - Ingestion, Alignment, and Feature Foundations

- [x] T04 Multimodal Ingestion and Timeline Alignment (Depends on: G01)
  - [x] T04.1 Implement ingestion adapters for EEG, optical, and IMU streams or files. (Depends on: G01)
  - [x] T04.2 Normalize timestamps to a common reference clock. (Depends on: T04.1)
  - [x] T04.3 Implement alignment diagnostics for drift, dropouts, and packet delay. (Depends on: T04.2)
  - [x] T04.4 Produce one synchronized session artifact with validation summary. (Depends on: T04.3)
- [x] T05 Windowing and Buffering Layer (Depends on: T04)
  - [x] T05.1 Implement fixed-size sliding windows for fast EEG updates. (Depends on: T04)
  - [x] T05.2 Implement slower rolling windows for optical context features. (Depends on: T05.1)
  - [x] T05.3 Preserve mapping from each window to absolute timeline index. (Depends on: T05.1, T05.2)
  - [x] T05.4 Add boundary tests for session start, session end, and missing packets. (Depends on: T05.3)
- [x] T06 EEG Preprocessing and Features (Depends on: T05)
  - [x] T06.1 Implement filtering and artifact-aware segmentation policy. (Depends on: T05)
  - [x] T06.2 Compute relative band powers and key band-ratio features. (Depends on: T06.1)
  - [x] T06.3 Add numerical safeguards for unstable spectra and clipping behavior. (Depends on: T06.2)
  - [x] T06.4 Persist per-window EEG features with quality flags. (Depends on: T06.3)
- [x] T07 Optical and IMU Quality Features (Depends on: T05)
  - [x] T07.1 Implement optical normalization plus trend, slope, and mean-change features. (Depends on: T05)
  - [x] T07.2 Implement IMU-derived motion artifact scoring per window. (Depends on: T05)
  - [x] T07.3 Compute confidence scores for quality-based down-weighting. (Depends on: T07.1, T07.2)
  - [x] T07.4 Persist per-window optical and quality feature tables. (Depends on: T07.3)

- [x] G02 Day 2 Exit Gate (Depends on: T04, T05, T06, T07)
  - [x] G02.1 At least one session passes synchronization checks with diagnostics.
  - [x] G02.2 Windowing layer passes edge-case tests.
  - [x] G02.3 EEG and optical or IMU features generate without runtime failure.

## Day 3 - Unified Features, Labeling, and MVP Training

- [x] T08 Unified Feature Builder (Depends on: G02)
  - [x] T08.1 Join EEG, optical, and IMU quality features into one aligned training matrix. (Depends on: G02)
  - [x] T08.2 Implement train-only normalization fit and val/test transform. (Depends on: T08.1)
  - [x] T08.3 Persist features with dataset version tags and split compatibility markers. (Depends on: T08.2)
  - [x] T08.4 Implement leakage checks for target contamination and session overlap. (Depends on: T08.3)
- [x] T09 Labeling and Session-Held-Out Split Utility (Depends on: T08)
  - [x] T09.1 Generate weak labels from task phase, subjective ratings, and performance drift. (Depends on: T08)
  - [x] T09.2 Add optional temporal smoothing with documented parameters. (Depends on: T09.1)
  - [x] T09.3 Implement session-level split isolation with deterministic seeding. (Depends on: T09.1)
  - [x] T09.4 Export split manifests for repeatable experiments. (Depends on: T09.3)
- [x] T10 MVP Model Training and Calibration (Depends on: T09)
  - [x] T10.1 Train Logistic Regression or LDA baseline with regularization sweep. (Depends on: T09)
  - [x] T10.2 Train tree-based baseline and compare fit quality on validation data. (Depends on: T10.1)
  - [x] T10.3 Implement calibrated late-fusion scoring across MVP model outputs. (Depends on: T10.2)
  - [x] T10.4 Persist models, scalers, hyperparameters, and training metadata. (Depends on: T10.3)

- [x] G03 Day 3 Exit Gate (Depends on: T08, T09, T10)
  - [x] G03.1 Split manifests are reusable and leakage-safe.
  - [x] G03.2 MVP training pipeline runs end-to-end.
  - [x] G03.3 Training artifacts are reproducible from saved config and manifest.

## Day 4 - MVP Evaluation, Live Inference, and Guardrails

- [x] T11 MVP Evaluation and Ablation Report (Depends on: G03)
  - [x] T11.1 Compute AUROC, macro F1, and balanced accuracy on held-out sessions. (Depends on: G03)
  - [x] T11.2 Run ablations: EEG-only, EEG+optical, EEG+optical+IMU-gated. (Depends on: T11.1)
  - [x] T11.3 Select production MVP model based on robustness and metric consistency. (Depends on: T11.2)
  - [x] T11.4 Publish concise evaluation summary for demo narrative and technical review. (Depends on: T11.3)
- [x] T12 Real-Time Inference Service (Depends on: T11)
  - [x] T12.1 Implement live feature path aligned to offline data contract and transforms. (Depends on: T11)
  - [x] T12.2 Load selected MVP model and emit risk score in range [0, 1]. (Depends on: T12.1)
  - [x] T12.3 Add runtime fault handling for packet loss and delayed modality updates. (Depends on: T12.2)
  - [x] T12.4 Log per-update latency and model confidence diagnostics. (Depends on: T12.3)
- [x] T13 Symbolic Guardrail Engine (Depends on: T12)
  - [x] T13.1 Implement GREEN, YELLOW, RED, and optional BLACK transition logic. (Depends on: T12)
  - [x] T13.2 Implement persistence and trend checks to suppress oscillation. (Depends on: T13.1)
  - [x] T13.3 Generate rationale text for each non-GREEN transition. (Depends on: T13.2)
  - [x] T13.4 Emit timestamped transition logs for replay and post-hoc review. (Depends on: T13.3)

- [x] G04 Day 4 Exit Gate (Depends on: T11, T12, T13)
  - [x] G04.1 Production MVP model is selected with held-out evidence.
  - [x] G04.2 Live inference loop emits stable and valid risk scores.
  - [x] G04.3 Guardrail transitions and rationale logs are functioning.

## Day 5 - Dashboard Integration and Quick Demo Flow

- [x] T14 Dashboard and Timeline Visualization (Depends on: G04)
  - [x] T14.1 Build real-time panels for EEG, optical trends, risk score, and current state. (Depends on: G04)
  - [x] T14.2 Add timeline with transition markers and rationale details. (Depends on: T14.1)
  - [x] T14.3 Add stream health and model-state indicators. (Depends on: T14.1)
  - [x] T14.4 Validate readability for classroom projection distance and resolution. (Depends on: T14.2, T14.3)
- [x] T15 Class Quick Demo Orchestration (Depends on: T14, O-05)
  - [x] T15.1 Implement guided 5 to 10 minute demo flow with stage timers. (Depends on: T14, O-05)
  - [x] T15.2 Validate sequence: 1-minute baseline, 2-minute sustained attention, 2-minute 2-back, 2 to 3-minute code block. (Depends on: T15.1)
  - [x] T15.3 Add final 30 to 60-second summary panel for transition interpretation. (Depends on: T15.2)
  - [x] T15.4 Run at least two dry runs and log reproducible failures. (Depends on: T15.3)

- [x] G05 Day 5 Exit Gate (Depends on: T14, T15)
  - [x] G05.1 Dashboard is stable and readable.
  - [x] G05.2 Quick demo completes within the 5 to 10 minute window.
  - [x] G05.3 Dry-run defects are prioritized for Day 6.

## Day 6 - MVP Hardening, Validation, and Documentation

- [x] T16 MVP Hardening and Release Readiness (Depends on: G05)
  - [x] T16.1 Fix top-priority dry-run failures, then medium-priority issues. (Depends on: G05)
  - [x] T16.2 Optimize update path to keep end-to-end refresh latency at or below 2 seconds. (Depends on: T16.1)
  - [x] T16.3 Validate alert stability under noisy and motion-heavy windows. (Depends on: T16.2)
  - [x] T16.4 Re-run AC-001 to AC-005 checklist and store objective evidence. (Depends on: T16.3)
- [x] T17 Documentation, Runbook, and Reproducibility Pack (Depends on: T16, O-04, O-07)
  - [x] T17.1 Update README with setup, train, evaluate, run-live, and replay commands in neuroguage. (Depends on: T16, O-07)
  - [x] T17.2 Write demo runbook with timing targets and fallback procedures. (Depends on: T16)
  - [x] T17.3 Document known limitations, risks, and deferred items. (Depends on: T16)
  - [x] T17.4 Package evaluation outputs, config snapshots, and evidence index. (Depends on: T17.1, T17.2, T17.3)

- [x] G06 Day 6 Exit Gate (Depends on: T16, T17)
  - [x] G06.1 AC-001 to AC-005 pass with documented evidence links.
  - [x] G06.2 Documentation package and runbook are complete and reproducible.
  - [x] G06.3 Phase-gate decision is recorded: Day 7 stretch allowed or blocked.

## Day 7 - Stretch Enhancement and Final Packaging

- [x] T18 Graduate Model Prototype (Stretch) (Depends on: G06)
  - [x] T18.1 Implement one advanced path: TCN or BiGRU-attention. (Depends on: G06)
  - [x] T18.2 Train advanced model on identical held-out split used by MVP. (Depends on: T18.1)
  - [x] T18.3 Measure latency and stability with same runtime harness as MVP. (Depends on: T18.2)
  - [x] T18.4 Persist advanced artifact, config, and evaluation logs for direct comparison. (Depends on: T18.3)
- [x] T19 Advanced vs MVP Comparison and Promotion Decision (Stretch) (Depends on: T18)
  - [x] T19.1 Compare advanced and MVP metrics on identical held-out sessions. (Depends on: T18)
  - [x] T19.2 Verify promotion threshold: at least +3 points macro F1 or AUROC. (Depends on: T19.1)
  - [x] T19.3 Verify no latency or alert stability regression. (Depends on: T19.1)
  - [x] T19.4 Record promotion decision note: promote or keep MVP, with rationale. (Depends on: T19.2, T19.3)
- [x] T20 Final Packaging and Demo Rehearsal (Depends on: T17; Optional dependency: T19)
  - [x] T20.1 If stretch is blocked, mark T18 and T19 deferred with reason and proceed with MVP package. (Depends on: G06)
  - [x] T20.2 Assemble final demo folder with scripts, models, configs, and quick-start notes. (Depends on: T17)
  - [x] T20.3 Run one full rehearsal under class timing constraints and record results. (Depends on: T20.2)
  - [x] T20.4 Freeze release-candidate artifacts and publish final status summary. (Depends on: T20.3)

- [x] G07 Day 7 Exit Gate (Depends on: T20; Optional dependency: T19)
  - [x] G07.1 MVP release package is finalized and class-ready.
  - [x] G07.2 Stretch model is promoted with evidence or explicitly deferred.
  - [x] G07.3 Final rehearsal passes timing and stability constraints.

## Dependency Summary

- [x] D-Map-01 Required MVP path (Depends on: O-01)
  - [x] D-Map-01.1 O-01 -> O-02 -> O-04 -> T01 -> T02 -> T03 -> G01
  - [x] D-Map-01.2 G01 -> T04 -> T05 -> T06 and T07 -> G02
  - [x] D-Map-01.3 G02 -> T08 -> T09 -> T10 -> G03
  - [x] D-Map-01.4 G03 -> T11 -> T12 -> T13 -> G04
  - [x] D-Map-01.5 G04 -> T14 -> T15 -> G05
  - [x] D-Map-01.6 G05 -> T16 -> T17 -> G06
  - [x] D-Map-01.7 G06 -> T20 -> G07
- [x] D-Map-02 Optional stretch path (Depends on: G06)
  - [x] D-Map-02.1 G06 -> T18 -> T19 -> T20 -> G07
- [x] D-Map-03 Orphan-task validation (Depends on: D-Map-01, D-Map-02)
  - [x] D-Map-03.1 Every task from T01 to T20 has at least one upstream dependency.
  - [x] D-Map-03.2 Every task from T01 to T20 contributes to at least one gate or downstream task.

## Definition of Weekly Success

- [x] S-01 Required success (Depends on: G06)
  - [x] S-01.1 T01 through T17 complete with acceptance evidence.
  - [x] S-01.2 AC-001 to AC-005 pass.
- [x] S-02 Stretch success (Depends on: T19)
  - [x] S-02.1 AC-006 to AC-009 evidence is complete. (AC-007 formally deferred with documented rationale; AC-006, AC-008, AC-009 have evidence.)
- [x] S-03 Final success (Depends on: G07)
  - [x] S-03.1 T20 complete, package frozen, and rehearsal pass recorded.
