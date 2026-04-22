# NeuroGauge Release Evidence Index

- Version: 0.1.0-day6
- Created: 2026-04-22T21:42:25.279610+00:00
- Git commit: `eb7af4ba05f00f05058f73c198672a677bdc3bcd`

## MVP Acceptance Summary (AC-001..AC-005)

- Passed: **5 / 5**

- [PASS] **AC-001** — End-to-end pipeline operates from ingestion through dashboard and event logging.
    - End-to-end evidence: inference diagnostics, guardrail transition log, and dashboard snapshot all present.
- [PASS] **AC-002** — Session-held-out evaluation reports AUROC, macro F1, balanced accuracy.
    - Selected variant 'eeg_optical' reports macro_f1=0.4, balanced_accuracy=0.5, auroc=1.0.
- [PASS] **AC-003** — Real-time latency meets NFR-001 (<= 2 s).
    - Refresh latency: mean=0.06ms, p95=0.15ms, max=0.32ms (budget=2.0s).
- [PASS] **AC-004** — Quick demo completes in 5 to 10 minutes without system failure.
    - Day 5 gate_passed=True, timing_within_budget=True, all_passed=True, dry_runs=2.
- [PASS] **AC-005** — Guardrail outputs are interpretable and include rationale text.
    - 2 of 2 non-GREEN transitions include rationale text.

## Gate Decisions

### DAY5

```json
{
  "g05_1_dashboard_stable": true,
  "g05_2_timing_within_budget": true,
  "g05_3_dry_runs_completed": true,
  "gate_passed": true,
  "dashboard_snapshot": "artifacts\\reports\\day5\\dashboard_snapshot.png",
  "dashboard_validation": "artifacts\\reports\\day5\\dashboard_validation.json",
  "dry_run_count": 2,
  "total_defects": 0
}
```

## Included Artifacts

| Label | Path | Included | SHA-256 |
|-------|------|----------|---------|
| Day 1 data contract | `configs\data_contract.json` | YES | `7a60d8adb4d2c39d…` |
| Day 5 demo timing envelope | `configs\demo_timing.json` | YES | `f59411b3cdf8160d…` |
| Day 4 ablation model variants | `models\ablations` | YES | `-` |
| Day 3 MVP baseline models | `models\day3_baseline` | YES | `-` |
| Day 3 training reports | `reports\day3` | YES | `-` |
| Day 4 evaluation summary | `reports\day4\evaluation_summary.json` | YES | `5d07a0c3de195c1d…` |
| Day 4 ablation report | `reports\day4\ablation_report.json` | YES | `6e027ad090588c1d…` |
| Day 4 evaluation narrative | `reports\day4\evaluation_narrative.md` | YES | `deb54a891b5fa4ed…` |
| Day 4 model selection | `reports\day4\model_selection.json` | YES | `593a4f1f5143c3d1…` |
| Day 4 inference diagnostics | `reports\day4\inference_diagnostics.json` | YES | `1724ba58129e3f68…` |
| Day 4 guardrail transition log | `reports\day4\guardrail_transition_log.json` | YES | `25b8475e2a08251e…` |
| Day 4 guardrail summary | `reports\day4\guardrail_summary.json` | YES | `3f105bb6032937cb…` |
| Day 4 integration summary | `reports\day4\day4_integration_summary.json` | YES | `61062cd11aad24b9…` |
| Day 5 dashboard snapshot | `reports\day5\dashboard_snapshot.png` | YES | `37e9b41f7cd43be0…` |
| Day 5 dashboard validation | `reports\day5\dashboard_validation.json` | YES | `1f5aa3e85deac859…` |
| Day 5 demo summary | `reports\day5\demo_summary.json` | YES | `c1829bc215fd4fc0…` |
| Day 5 dry-run defect log | `reports\day5\defect_log.json` | YES | `4f53cda18c2baa0c…` |
| Day 5 exit gate evidence | `reports\day5\day5_exit_gate.json` | YES | `fbae591a2b6270c1…` |
| Day 6 latency benchmark | `reports\day6\latency_benchmark.json` | YES | `690153fc43fc58db…` |
| Day 6 alert-stability report | `reports\day6\stability_report.json` | YES | `3214b77a2bea0b7c…` |
| Day 6 dry-run defect triage | `reports\day6\defect_triage.json` | YES | `8c120d2c744a7969…` |
| Day 6 AC-001..AC-005 results | `reports\day6\acceptance_results.json` | YES | `f73dd70238a3a3d7…` |
| Day 6 exit gate evidence | `reports\day6\day6_exit_gate.json` | no | `-` |
| Updated MVP acceptance checklist | `checklists\mvp_acceptance_ac_001_to_ac_005.md` | YES | `f7aa168b0b705782…` |
| Stretch acceptance checklist | `checklists\stretch_acceptance_ac_006_to_ac_009.md` | YES | `3c876b8e00d32e74…` |
| T17.2 Demo runbook | `docs\demo_runbook.md` | YES | `315874fcf328cbc8…` |
| T17.3 Limitations and risks | `docs\limitations_and_risks.md` | YES | `a77359de37842337…` |
| Updated README | `docs\README.md` | YES | `ca5aded4309d4cd7…` |
| Conda environment spec | `env\environment.yml` | YES | `89efe8da929dcafc…` |
| Pip requirements | `env\requirements.txt` | YES | `868297d7522759df…` |
