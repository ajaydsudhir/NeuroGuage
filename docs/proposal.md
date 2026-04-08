NeuroGauge Proposal (Research-Grounded MVP)

## 1) Project Summary
NeuroGauge is a real-time cognitive fatigue estimation system using consumer wearable neuro-signals. The system combines:
- EEG features for fast neural engagement and fatigue trend signals.
- Optical hemodynamic proxy features for slower metabolic context.
- Symbolic guardrail rules for interpretable risk levels and alerts.

The MVP goal is not clinical diagnosis. The goal is early warning of likely cognitive fatigue progression during demanding work sessions.

Execution strategy is explicitly phased:
- Phase 1 (MVP first): robust end-to-end pipeline with interpretable, lightweight models that can be demonstrated reliably in class.
- Phase 2 (Graduate enhancement): advanced temporal multimodal models, compared against MVP baselines with strict validation and latency constraints.

## 2) Why This Problem Matters
Cognitive fatigue can rise before obvious behavioral failure. In high-demand tasks, delayed detection can reduce performance safety margins.

Evidence from recent literature supports the multimodal direction but also defines realistic constraints:
- Multimodal fusion can outperform single modalities in fatigue classification (for example, one controlled study reported average accuracy gains from single-modality ranges around 59.7 to 66.8 up to about 75.9 with EEG+fNIRS+ECG fusion).
- EEG often provides clearer continuous fatigue trajectories than fNIRS in longer simulated flight tasks.
- fNIRS and related optical signals remain promising but are slower, more channel-sensitive, and more artifact-sensitive in operational settings.
- Translation from lab studies to cockpit or real-world deployment still faces comfort, calibration, artifact, and certification constraints.

Design implication for this project: use EEG as the primary continuous signal, and use optical/metabolic features as a contextual confidence modifier rather than a sole decision source.

## 3) Problem Statement
Current fatigue detection prototypes often have at least one of these gaps:
- Single-modality overreliance (especially EEG-only pipelines).
- Weak interpretability of alert decisions.
- Limited handling of temporal mismatch between fast neural and slow hemodynamic signals.

NeuroGauge addresses these gaps with a layered pipeline and explicit safety rules.

## 4) Goals and Non-Goals
### Goals
- Build a working real-time multimodal fatigue risk estimator.
- Demonstrate interpretable neuro-metabolic risk logic with visible intermediate signals.
- Achieve reproducible training and evaluation workflows suitable for a class project.
- Run on local consumer hardware with open-source tooling.

### Non-Goals (MVP boundaries)
- No medical or clinical claim.
- No FAA/clinical-grade deployment validation.
- No large population-general model due to one-user collection constraint.

## 5) Users and Use Cases
Primary user:
- Single pilot user (project owner) performing sustained cognitive tasks.

MVP use cases:
- Monitor fatigue risk during 30 to 90 minute focused sessions.
- Receive graded alerts before major subjective fatigue escalation.
- Review session timeline with feature traces and predicted risk states.

## 6) System Architecture
### Layer A: Signal Acquisition
- Device: Muse S Athena.
- Streams:
  - EEG channels (TP9, AF7, AF8, TP10, AUX).
  - Optical channels (NIR, IR, RED, ambient streams).
  - Motion channels (accelerometer and gyroscope).
- Transport options: OpenMuse and/or muse-lsl/BrainFlow compatible streaming.

### Layer B: Preprocessing and Feature Engineering
- EEG preprocessing:
  - Band-pass filtering and artifact-aware segmentation.
  - Relative band powers (theta, alpha, beta), band ratios, spectral statistics.
- Optical preprocessing:
  - Ambient correction and normalization.
  - Sliding-window trend, slope, and mean-change features as metabolic proxy.
- Motion-aware quality handling:
  - Flag windows with strong movement artifact for down-weighting or exclusion.
- Windowing strategy:
  - Fast windows for EEG updates and slower rolling context windows for optical features.

### Layer C: ML Inference
- Phase 1 (MVP models, completed first):
  - Logistic Regression / LDA on engineered multimodal features.
  - Tree-based model (Random Forest or XGBoost) for nonlinear interactions.
  - Simple calibrated late-fusion ensemble for robust first deployment.
- Phase 2 (Graduate-level enhancement models):
  - Temporal Convolutional Network (TCN) or BiGRU with attention on sequential multimodal windows.
  - Optional compact multimodal Transformer encoder if data volume is sufficient after augmentation and session accumulation.
- Training rigor:
  - Session-wise train/validation/test split.
  - Optimizer: AdamW with learning-rate scheduling (for neural models).
  - Regularization: dropout, early stopping, and weight decay.
  - Report ablation: EEG-only vs EEG+optical vs EEG+optical+IMU quality gating.
  - Report model comparison: MVP models vs graduate enhancement models on identical held-out sessions.
- Fusion approach:
  - Late fusion of modality-specific scores in MVP.
  - Confidence weighting based on signal quality and modality stability.
- Output:
  - Continuous fatigue risk score in [0, 1].

Model selection rule for this project:
- Complete and validate MVP model stack first.
- Promote a graduate-level model only if it improves held-out session metrics and preserves real-time latency and alert stability.
- If advanced models overfit due to one-user data volume, keep MVP as production demo model and document the tradeoff.

### Layer D: Symbolic Guardrail Engine
Rules convert score trajectories into actionable states:
- GREEN: low risk, stable trend.
- YELLOW: elevated risk or accelerating risk trend.
- RED: sustained high risk with corroborating features.
- BLACK (optional strict mode): critical sustained high risk with degraded signal quality safeguards.

Guardrails use explicit criteria (threshold + persistence + trend) to reduce noisy one-window alerts.

## 7) Data Strategy
Constraint: only one user is available for direct data collection.

Plan:
- Use public EEG/fNIRS fatigue-related resources to inform feature/model priors.
- Collect multi-session personal calibration data with repeated tasks.
- Split by session (not random windows) to avoid leakage.
- Maintain metadata per session:
  - task type,
  - start/end timestamps,
  - self-report fatigue ratings,
  - performance metrics when available.

## 8) Experimental Protocol (MVP)
Locked class-demo task set (multi-task requirement):
- Task 1: Baseline eyes-open rest (2 to 3 minutes).
  - Purpose: collect calm reference window for per-session normalization.
- Task 2: Sustained attention reaction task (5 minutes).
  - Example: continuous response or simple vigilance/reaction-time task.
  - Purpose: induce sustained attention load with measurable drift.
- Task 3: 2-back working memory task (5 minutes).
  - Purpose: induce higher executive load and short-term memory strain.
- Task 4: Code comprehension block (6 to 8 minutes).
  - Purpose: represent realistic focused class/project behavior.

Why this set is demo-friendly:
- Short enough to run live in class.
- Covers low, medium, and high cognitive demand patterns.
- Easy to repeat across sessions for comparable model evaluation.

Class quick demo mode (strict 5 to 10 minutes total, MVP-first):
- 1 minute baseline rest.
- 2 minutes sustained attention task.
- 2 minutes 2-back task.
- 2 to 3 minutes code comprehension block.
- 30 to 60 seconds for live risk timeline and guardrail-state explanation.

Optional graduate enhancement add-on (no extra live collection required):
- Replay the same demo session through the advanced model and show side-by-side risk trajectory comparison.

Session design:
- 30 to 90 minutes per session.
- Baseline rest segment before each task.
- Periodic subjective fatigue rating prompts.

In-class extended demo variant (about 12 to 15 minutes total):
- 2 minutes baseline rest.
- 3 to 4 minutes sustained attention task.
- 3 to 4 minutes 2-back task.
- 4 to 5 minutes code comprehension block.

Label strategy:
- Weak labels from task phase + self-report + performance drift.
- Optional pseudo-label smoothing for temporal consistency.

## 9) Evaluation Metrics and Success Criteria
### Offline model metrics
- AUROC, macro F1, balanced accuracy.
- Session-held-out validation as primary result.

### Real-time performance metrics
- End-to-end update latency target: <= 2 seconds per refresh cycle.
- Alert stability target: avoid rapid oscillation between adjacent states.
- Early warning objective: detect sustained rising risk before severe subjective fatigue stage in the same session.

### MVP acceptance criteria
- Reproducible training and inference pipeline runs end-to-end.
- Better-than-random and stable cross-session fatigue discrimination for the single user.
- Interpretable dashboard shows signal traces, model score, and guardrail state.

### Graduate enhancement acceptance criteria
- Advanced model shows consistent improvement over MVP baseline on held-out sessions.
- Improvement target: at least +3 percentage points in macro F1 or AUROC averaged across held-out sessions.
- No real-time regression: alert latency and stability remain within targets.
- Comparative report includes ablation and error analysis, not only headline metrics.

## 10) Real-Time Outputs
- EEG panel with channel traces and derived engagement/fatigue indicators.
- Optical/metabolic panel with normalized trend view.
- Risk state panel with GREEN/YELLOW/RED/BLACK status and rationale text.
- Session timeline log of transitions and trigger reasons.

## 11) Risk Register and Mitigation
- Risk: Overfitting due to one-user dataset.
  - Mitigation: strict session-wise validation, regularization, and phased model gating (MVP first, advanced model only after stable gains).
- Risk: Inconsistent optical/fNIRS behavior across sessions.
  - Mitigation: treat optical signals as contextual features, not sole trigger.
- Risk: Motion artifacts causing false alerts.
  - Mitigation: IMU-based artifact flags and confidence gating.
- Risk: Label noise from subjective fatigue ratings.
  - Mitigation: combine subjective, task-phase, and performance cues.

## 12) Implementation Plan and Milestones
- Milestone 1: Data ingestion and synchronized multimodal logging.
- Milestone 2: Preprocessing and feature extraction pipeline.
- Milestone 3: MVP model training, calibration, and 5 to 10 minute class demo readiness.
- Milestone 4: Graduate-level model enhancement (TCN/BiGRU-attention, optional compact Transformer).
- Milestone 5: Comparative evaluation (MVP vs enhanced), dashboard polish, documentation, and reproducibility packaging.

## 13) Tooling and Environment
- OS: Windows 11.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU.
- Python: 3.12.x in Conda environment.
- Core libraries: PyTorch, scientific Python stack, streaming stack (OpenMuse and/or BrainFlow/muse-lsl).

## 14) Instructor Requirement Mapping 
(These may be omitted if they require significant additional work but try to include them as much as possible)
- Run on multiple tasks: included in calibration protocol.
- Training loop: explicit MVP training stage followed by graduate-level enhancement stage.
- Optimizer selection: required for trainable models.
- Normalization techniques: required in preprocessing.
- Feature engineering: core project component.
- Multiple commits over time: development process requirement.
- README run instructions: required deliverable.
- requirements.txt or environment.yml: required deliverable.
- Clear directory structure: required deliverable.
- Documentation/comments and future work notes: required deliverable.

## 15) Expected Outcomes
- A functioning proof-of-concept fatigue risk monitor for one-user longitudinal sessions.
- An interpretable neuro-symbolic alerting mechanism with explicit decision logic.
- A reproducible class-project codebase and documentation package.

## 16) References Informing Scope
- PMCID: PMC4865510 (multimodal fatigue classification, controlled study).
- PMCID: PMC6032997 (hybrid EEG-fNIRS methods review).
- PMCID: PMC10036528 (EEG-fNIRS in simulated flight fatigue progression).
- PMCID: PMC12833356 (cockpit deployment review and translational constraints).

## 17) Final Phase-Gate Checklist
Phase 1 gate (MVP must pass before any model upgrade):
- End-to-end pipeline runs live with stable risk-state updates.
- Class quick demo completes in 5 to 10 minutes.
- Session-held-out metrics and alert stability meet MVP acceptance criteria.

Phase 2 gate (graduate enhancement only after Phase 1 pass):
- Advanced model trained and evaluated on identical held-out sessions.
- Improvement over MVP is consistent (target >= +3 points macro F1 or AUROC) and does not violate latency targets.
- Comparative analysis documents gains, failures, and deployment tradeoffs.