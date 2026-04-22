# NeuroGauge — Known Limitations, Risks, and Deferred Items

Version: v1.0.0 (Day 6)
Authoritative source: this document supersedes any earlier risk notes.

## 1. Known Limitations

### 1.1 Single-Subject Data (PRD R-001)
- Only one pilot user contributed longitudinal data, per `docs/PRD.md` §7.
- Classification metrics therefore reflect *subject-specific* behaviour
  only. No cross-subject generalisation claims are made.
- Mitigation applied: session-wise splits (`T09.3`, `T09.4`), regularised
  MVP baselines, conservative phase-gate threshold of +3 pts for any
  Phase 2 promotion.

### 1.2 Small Labeled-Session Volume
- The held-out feature matrix contains 24 labeled rows (see
  `artifacts/features/day3/labeled_feature_matrix.csv`).
- Held-out AUROC, macro F1, and balanced accuracy estimates have wide
  confidence intervals; numerical values should be read as *lower bounds
  on pipeline correctness*, not production-grade performance claims.

### 1.3 Weak Labels from Subjective Fatigue
- Labels derive from task phase + subjective ratings + performance drift
  (`labeling.py`). Subjective drift and end-of-task fatigue bias are not
  fully removable.
- Temporal smoothing parameters (`T09.2`) are documented but not yet
  swept across a grid; chosen defaults are fixed at v0.1.0.

### 1.4 Optical Channel Contextual Use Only
- Optical features feed the fusion head but never trigger a state change
  on their own (PRD R-002).
- The MVP selected variant at v0.1.0 is `eeg_optical`; the
  `eeg_optical_imu_gated` variant is available but scored slightly lower
  on the composite metric — see `artifacts/reports/day4/evaluation_summary.json`.

### 1.5 IMU Gating Heuristic
- Motion-artifact scoring uses a fixed-magnitude heuristic with
  `imu_motion_score = clip(std(acc) + std(gyro) / 0.5, 0, 1)`.
- The threshold `0.5` is empirically tuned against the project's pilot
  recordings only; different recording conditions may require
  recalibration.

### 1.6 Guardrail Thresholds
- Threshold values in `GuardrailConfig` (YELLOW=0.40, RED=0.65,
  BLACK=0.85) are static and not personalised.
- Persistence (`escalation_persistence=3`) and hysteresis
  (`green_recovery_threshold=0.30`, etc.) are validated in the Day 6
  stability report (`artifacts/reports/day6/stability_report.json`).

### 1.7 Dashboard and Demo Scope
- Dashboard is Matplotlib-based and tuned for classroom projection at
  >= 1280x720. It is not a production UI and does not yet support
  multi-session overlay or user interaction beyond stage advancement.
- Demo orchestrator runs in either live (stream) or dry-run (replay)
  mode; hybrid operator-paced mode is deferred to post-v0.1.0.

## 2. Risk Register Snapshot

| ID | Risk | Likelihood | Impact | Mitigation | Evidence |
|---|---|---|---|---|---|
| R-001 | Overfitting to single subject | High | High | Session-held-out validation, regularisation sweeps, conservative promotion gate | `artifacts/reports/day4/ablation_report.json` |
| R-002 | Optical channel instability | Medium | Medium | Optical used as contextual modifier only; confidence-weighted fusion | `src/neuroguage/optical_imu_features.py` |
| R-003 | Motion artifacts triggering false RED | Medium | Medium | IMU gating + `min_confidence_for_red` blocks low-confidence escalation | `artifacts/reports/day6/stability_report.json` scenario `motion_artifact` |
| R-004 | Label uncertainty from subjective reports | Medium | Medium | Multi-signal weak supervision (`T09.1`), optional temporal smoothing | `src/neuroguage/labeling.py` |
| R-005 | Advanced model over-complexity | Low | Medium | Phase gate requires +3 pts metric improvement and no latency regression | `docs/day1/day6_go_no_go_rule.md` |
| R-006 | Demo timing overrun | Low | Medium | Frozen envelope + stage timers + forced advance (see `configs/demo_timing.json`) | `artifacts/reports/day5/dry_run_*_report.json` |
| R-007 | Dashboard freeze under live load | Low | Medium | Replay-mode fallback documented in runbook | `docs/day6/demo_runbook.md` §4 F-02 |

## 3. Deferred Items (Out of MVP Scope)

The following are intentionally deferred past the v0.1.0 MVP release and
tracked as candidates for Day 7 stretch or post-MVP work.

| Deferred | Trace | Rationale |
|---|---|---|
| T18 Graduate model (TCN or BiGRU-attention) | PRD §6.2, TODO D-Map-02 | Gated behind AC-001..AC-005 pass and phase-gate GO decision |
| T19 Advanced vs MVP comparison and promotion | TODO D-Map-02 | Requires T18 artifacts |
| Cross-subject evaluation | PRD §6.3 (Out of Scope) | Single-user data only |
| Clinical endpoint prediction | PRD §3.2 | Explicit non-goal |
| Cloud deployment | PRD §6.3 | Explicit non-goal |
| Multi-session dashboard overlay | Post-v0.1.0 | UX enhancement |
| Operator-paced hybrid demo mode | Post-v0.1.0 | Scope freeze |
| Personalised guardrail thresholds | Post-v0.1.0 | Requires more subjects |
| Full grid sweep over smoothing windows | Post-v0.1.0 | Small dataset diminishing returns |

## 4. Phase-Gate Decision Criteria

Per `docs/day1/day6_go_no_go_rule.md`:

- **GO for Day 7 stretch**: AC-001 through AC-005 all pass with objective
  evidence links in `artifacts/reports/day6/acceptance_results.json`.
- **NO-GO**: Any AC-001..AC-005 item fails or lacks evidence; Day 7
  follows the MVP-only packaging path and T18/T19 are marked deferred.

The authoritative gate evidence is
`artifacts/reports/day6/day6_exit_gate.json`.

## 5. Privacy and Data Handling

- Raw session files remain local to the project workspace per NFR-007.
- The release bundle (`artifacts/release/`) contains only aggregated
  reports, models, and documentation — no raw EEG/optical/IMU samples.
