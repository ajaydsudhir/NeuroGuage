# Stretch Acceptance Checklist (AC-006 to AC-009)

Version: v1.0.0-rc1
Date: 2026-04-22

- [x] AC-006 Advanced model evaluated on identical held-out sessions as MVP.
  - Evidence link: `artifacts/reports/day7/promotion_decision.json` (`held_out_sessions`, identical `feature_columns` as `artifacts/models/day4/ablations/eeg_optical/feature_columns.json`).
  - Evidence link: `artifacts/reports/day7/advanced_predictions.csv` (per-window scores on the identical held-out session `s01`).
- [ ] AC-007 Mean improvement is at least +3 points macro F1 or AUROC over MVP.
  - Status: **FAIL** — advanced BiGRU-attention matched MVP on test macro_f1 (0.4000) and AUROC (1.0000). Delta = 0.0000. See `artifacts/reports/day7/promotion_decision.md`.
  - Deferral rationale: the single-subject test split contains only 5 windows; the PRD promotion threshold cannot be met with this data volume without overfitting. Advanced prototype is preserved; AC-007 is formally deferred to a follow-up data-collection phase (see `docs/day6/limitations_and_risks.md`).
- [x] AC-008 No regression on latency or alert stability relative to MVP.
  - Evidence link: `artifacts/reports/day7/promotion_decision.md` (latency + stability tables).
  - Evidence link: `artifacts/reports/day7/advanced_latency_benchmark.json` (max 3.314 ms < 2 s budget).
  - Evidence link: `artifacts/reports/day7/advanced_stability_report.json` (all scenarios passed).
- [x] AC-009 Comparative report includes ablation and error analysis.
  - Evidence link: `artifacts/reports/day7/promotion_decision.md` (side-by-side metrics, latency, stability; decision rationale).
  - Evidence link: `artifacts/reports/day4/ablation_report.json` (shared MVP ablation baseline).
  - Evidence link: `artifacts/reports/day7/advanced_training_log.json` (per-epoch train/val trajectory for the advanced model).

## Overall stretch outcome

**Stretch: completed with decision `KEEP_MVP`.** Advanced model artifacts are preserved under `artifacts/models/day7/advanced/` and included in the release bundle. AC-007 is formally deferred with rationale above; AC-006, AC-008, and AC-009 have documented evidence. The MVP remains the class-demo production model per FR-025.
