# eNeuroGauge Product Requirements Document (PRD)

## Document Control

- Status: Final comprehensive baseline for implementation
- Project: NeuroGauge
- Repository: NeuroGuage
- Source of truth: proposal.md and docs/TODO.md
- Required runtime environment: Conda environment named `neuroguage`

## 1. Product Overview

NeuroGauge is a real-time cognitive fatigue risk monitoring system using consumer wearable neuro-signals. The product combines neural indicators from EEG, metabolic context from optical channels, and an interpretable symbolic guardrail engine that maps model outputs to actionable fatigue risk states.

The product is delivered in two phases:

- Phase 1 MVP first: reliable, demonstrable, end-to-end system suitable for a 5 to 10 minute in-class demo.
- Phase 2 Graduate enhancement: advanced temporal multimodal models with strict promotion gates over MVP baselines.

## 2. Problem Statement

Users can experience cognitive fatigue before obvious behavior breakdown. Existing single-modality or black-box systems can be difficult to trust, difficult to interpret, and brittle under real-world signal artifacts.

NeuroGauge addresses this by:

- Combining multiple signal streams.
- Providing transparent risk-state logic.
- Enforcing session-held-out validation and latency constraints.

## 3. Goals and Non-Goals

### 3.1 Goals

- Build a real-time multimodal fatigue risk estimator.
- Provide interpretable risk transitions: GREEN, YELLOW, RED, optional BLACK.
- Support robust MVP operation on one-user longitudinal data.
- Demonstrate graduate-level model enhancement with quantitative gains over MVP.
- Produce class-ready, reproducible engineering artifacts.

### 3.2 Non-Goals

- Clinical diagnosis or medical claims.
- Population-wide deployment claims.
- Aviation certification or safety certification work.

## 4. Stakeholders

- Primary user: project owner and operator.
- Secondary users: instructor and class audience.
- Engineering stakeholders: model developer, data engineer, visualization developer.

## 5. Users and Primary Use Cases

### 5.1 Primary User

- One pilot user performing sustained cognitive tasks.

### 5.2 Core Use Cases

- Monitor fatigue risk during focused sessions.
- Trigger early warning before severe subjective fatigue.
- Review timeline of signals, model score, and guardrail transitions.
- Compare MVP and graduate model outputs on the same recorded session.

## 6. Scope

### 6.1 Phase 1 MVP In Scope

- Signal ingestion and synchronization from EEG, optical, and IMU streams.
- Preprocessing and feature extraction pipeline.
- MVP models: LDA or Logistic Regression plus tree-based baseline and calibrated late fusion.
- Symbolic guardrail logic.
- Real-time dashboard and event timeline logging.
- Class quick demo mode constrained to 5 to 10 minutes.

### 6.2 Phase 2 Graduate Enhancement In Scope

- Temporal models: TCN or BiGRU with attention.
- Optional compact multimodal Transformer if data volume supports it.
- Comparative evaluation versus MVP using identical held-out sessions.
- Ablation and failure analysis report.

### 6.3 Out of Scope

- Multi-subject domain generalization claims.
- Clinical endpoint prediction.
- Production-grade cloud deployment.

## 7. Assumptions and Constraints

- Only one user is available for direct data collection.
- Hardware: Windows laptop with RTX 4050 and Muse S Athena.
- Session-held-out evaluation is mandatory to avoid leakage.
- EEG is the primary continuous signal; optical features are contextual confidence modifiers.

### 7.1 Required Runtime Environment

- Standard project environment: Conda environment named `neuroguage`.
- All project scripts, training jobs, evaluation runs, and demo runs shall be executed from `neuroguage`.
- Default activation command on Windows: `conda activate neuroguage`.
- Environment reproducibility source files: `environment.yml` and/or `requirements.txt`.

## 8. Product Requirements

### 8.1 Functional Requirements

#### Ingestion and Synchronization

- FR-001: System shall ingest EEG channels from Muse stream in real time.
- FR-002: System shall ingest optical channels in real time.
- FR-003: System shall ingest IMU channels in real time.
- FR-004: System shall synchronize multimodal streams onto a common timeline.
- FR-005: System shall buffer data for sliding-window feature extraction.

#### Preprocessing and Feature Engineering

- FR-006: System shall apply EEG filtering and segmentation per configured windows.
- FR-007: System shall compute EEG spectral features including relative band powers and ratios.
- FR-008: System shall apply optical normalization and trend feature extraction.
- FR-009: System shall compute IMU-based motion artifact indicators.
- FR-010: System shall perform quality gating or down-weighting for artifact-heavy windows.

#### Training and Evaluation

- FR-011: System shall train MVP models on engineered multimodal features.
- FR-012: System shall use session-wise train/validation/test partitions.
- FR-013: System shall compute AUROC, macro F1, and balanced accuracy.
- FR-014: System shall export model and evaluation artifacts per run.
- FR-015: System shall support ablation comparisons for EEG-only, EEG plus optical, and EEG plus optical plus IMU-gated variants.

#### Real-Time Inference and Guardrails

- FR-016: System shall produce continuous fatigue risk score in range 0 to 1.
- FR-017: System shall map risk score to guardrail states GREEN, YELLOW, RED, optional BLACK.
- FR-018: System shall apply persistence and trend logic before state escalation.
- FR-019: System shall display rationale text for each non-GREEN transition.
- FR-020: System shall log timestamped state transitions and trigger reasons.

#### Dashboard and Replay

- FR-021: System shall display real-time EEG and optical trend panels.
- FR-022: System shall display current risk score and state badge.
- FR-023: System shall support session replay mode for post-hoc analysis.
- FR-024: System shall support side-by-side replay comparison between MVP and graduate models.

#### Phase-Gate and Model Promotion

- FR-025: System shall treat MVP as default production demo model until phase-2 promotion criteria are met.
- FR-026: System shall promote an advanced model only if macro F1 or AUROC improves by at least 3 points on held-out sessions and no real-time regression is observed.

### 8.2 Non-Functional Requirements

- NFR-001 Latency: End-to-end inference and UI refresh cycle shall be 2 seconds or less.
- NFR-002 Stability: Alert state shall not oscillate excessively due to single-window noise; persistence logic must be enforced.
- NFR-003 Reliability: Real-time demo shall complete without pipeline crash in a 5 to 10 minute class run.
- NFR-004 Reproducibility: Training and evaluation shall be repeatable from documented commands and environment files, executed from the `neuroguage` Conda environment.
- NFR-005 Explainability: Non-GREEN states shall present trigger rationale in human-readable form.
- NFR-006 Maintainability: Codebase shall be organized with clear module boundaries and documentation.
- NFR-007 Privacy: Raw session files shall remain local to project workspace unless explicitly exported.

## 9. Data Requirements

### 9.1 Data Sources

- Live collection from Muse S Athena streams.
- Public resources for feature and model prior development.

### 9.2 Session Metadata

Each session record shall include:

- Task type.
- Start and end timestamps.
- Subjective fatigue annotations.
- Optional performance metrics (reaction time, accuracy, completion quality).

### 9.3 Labeling Strategy

- Weak labels from task phase, subjective ratings, and performance drift.
- Optional temporal smoothing for label consistency.
- Session-wise separation is mandatory to prevent leakage.

## 10. Experimental and Demo Requirements

### 10.1 Standard Task Set

- Baseline eyes-open rest.
- Sustained attention reaction task.
- 2-back working memory task.
- Code comprehension block.

### 10.2 Class Quick Demo Mode (Mandatory)

Duration: 5 to 10 minutes total.

Sequence:

- 1 minute baseline rest.
- 2 minutes sustained attention.
- 2 minutes 2-back.
- 2 to 3 minutes code comprehension.
- 30 to 60 seconds for timeline and state interpretation.

### 10.3 Extended Demo Mode (Optional)

Duration: 12 to 15 minutes total.

### 10.4 Graduate Enhancement Add-On (Optional)

- Replay the same recorded quick-demo session with advanced model.
- Show side-by-side trajectory and confusion points versus MVP.

## 11. Metrics and Acceptance Criteria

### 11.1 MVP Acceptance Criteria

- AC-001 End-to-end pipeline operates from ingestion through dashboard and event logging.
- AC-002 Session-held-out evaluation reports AUROC, macro F1, balanced accuracy.
- AC-003 Real-time latency meets NFR-001.
- AC-004 Quick demo completes in 5 to 10 minutes without system failure.
- AC-005 Guardrail outputs are interpretable and include rationale text.

### 11.2 Graduate Enhancement Acceptance Criteria

- AC-006 Advanced model evaluated on identical held-out sessions as MVP.
- AC-007 Mean improvement is at least +3 points macro F1 or AUROC over MVP.
- AC-008 No regression on latency or alert stability relative to MVP.
- AC-009 Comparative report includes ablation and error analysis.

## 12. Milestones and Delivery Plan

- M1: Data ingestion and synchronization complete.
- M2: Preprocessing and feature extraction complete.
- M3: MVP model training and calibration complete.
- M4: MVP quick demo validation complete.
- M5: Graduate enhancement model training complete.
- M6: Comparative evaluation and final documentation complete.

## 13. Instructor Requirement Alignment

- Multiple tasks: covered by four-task experimental protocol.
- Training loop and optimizer: explicit in MVP and enhancement phases.
- Normalization and feature engineering: required preprocessing components.
- Reproducibility artifacts: README, environment file, run instructions.
- Development quality: multi-commit history, clear structure, documentation.

## 14. Risks and Mitigation

- R-001 Overfitting due to one-user data.
  - Mitigation: session-held-out validation, regularization, conservative claims.
- R-002 Optical channel instability.
  - Mitigation: treat optical channels as contextual, not sole trigger.
- R-003 Motion artifacts generating false positives.
  - Mitigation: IMU gating and confidence-weighted fusion.
- R-004 Label uncertainty from subjective reports.
  - Mitigation: combine multiple weak supervision signals.
- R-005 Advanced model over-complexity.
  - Mitigation: phase gate requires measurable improvement and no latency regression.

## 15. Dependencies

- Conda runtime: `neuroguage` environment (required baseline for all scripts).
- Muse streaming stack: OpenMuse and/or BrainFlow compatibility.
- Python with PyTorch and scientific libraries.
- Visualization stack for real-time dashboard and replay.

## 16. Deliverables

- D-001 Working MVP pipeline with live dashboard.
- D-002 Trained MVP model artifacts and evaluation report.
- D-003 Graduate enhancement model artifacts and comparison report.
- D-004 Demo script and runbook for class presentation.
- D-005 Documentation package: README, environment specification, future work notes.

## 17. Definition of Done

Project is done when:

- Phase 1 MVP acceptance criteria AC-001 to AC-005 are met.
- Phase 2 acceptance criteria AC-006 to AC-009 are met or formally deferred with documented rationale.
- Final repository is reproducible, documented, and demonstrable in class constraints.

## 18. Seven-Day Implementation Alignment

- Day 1 focus: scope lock, environment baseline, and data contract.
- Day 2 focus: ingestion, synchronization, and feature foundations.
- Day 3 focus: unified features, labeling, and MVP model training.
- Day 4 focus: MVP evaluation, real-time inference, and guardrail logic.
- Day 5 focus: dashboard integration and quick demo flow.
- Day 6 focus: MVP hardening, acceptance validation, and documentation freeze.
- Day 7 focus: optional graduate enhancement and final packaging.
- Daily execution is controlled by explicit day exit gates G01 through G07 defined in TODO.md.
- Task policy requires no orphan tasks: every task has upstream dependency and downstream contribution.

Execution rule:

- If AC-001 to AC-005 are not fully met by end of Day 6, Day 7 is MVP-only and Phase 2 tasks are deferred.

## 19. Requirements Traceability to TODO IDs

| PRD Requirement Group                                                                   | Primary TODO Task IDs              |
| --------------------------------------------------------------------------------------- | ---------------------------------- |
| FR-001 to FR-005 (ingestion, sync, buffering)                                           | T04, T05                           |
| FR-006 to FR-010 (preprocessing, features, quality gating)                              | T06, T07, T08                      |
| FR-011 to FR-015 (training, evaluation, ablations)                                      | T09, T10, T11                      |
| FR-016 to FR-020 (live inference, guardrails, rationale, logs)                          | T12, T13                           |
| FR-021 to FR-022 (dashboard)                                                            | T14                                |
| FR-023 to FR-024 (replay and side-by-side comparison)                                   | T18, T19, T20                      |
| FR-025 to FR-026 (phase gate and model promotion)                                       | T01, T16, T19                      |
| NFR-001 to NFR-003 (latency, stability, reliability)                                    | T15, T16, T20                      |
| NFR-004 to NFR-007 (reproducibility, explainability, maintainability, privacy)          | T02, T11, T17                      |
| Governance and execution controls (phase gating, evidence, no-orphan dependency policy) | O-01 to O-07, G01 to G07, D-Map-03 |

Traceability rule:

- No requirement is considered complete until corresponding TODO tasks and gate evidence are marked complete.
