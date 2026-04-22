# NeuroGauge Class Demo - Quick Start (T20.2)

This note ships inside the release-candidate bundle so the operator can run
the class demo end-to-end without external references.

All commands assume the required Conda environment is active:

```powershell
conda activate neuroguage
```

## 1. Bundle layout

```
artifacts/release/
  configs/                 # frozen data contract + demo timing envelope
  models/
    ablations/             # Day 4 ablation variants (MVP)
    day3_baseline/         # Day 3 MVP baseline models
    day7_advanced/         # BiGRU-attention stretch model (if stretch ran)
  reports/
    day3/ day4/ day5/ day6/ day7/
  checklists/
    mvp_acceptance_ac_001_to_ac_005.md
    stretch_acceptance_ac_006_to_ac_009.md
  docs/
    README.md              # repo quick-start (from Day 6)
    demo_runbook.md        # operator runbook with timing + fallbacks
    limitations_and_risks.md
    quick_start.md         # this file
  manifest.json            # SHA-256 index of every included artifact
  evidence_index.md        # human-readable bundle index
```

## 2. Five-minute class demo (MVP)

1. Launch dashboard + guardrail loop in replay mode:

   ```powershell
   python scripts/day5_dashboard_demo.py
   ```

2. Run the quick demo orchestrator (prints per-stage banner, logs
   transitions, renders the summary panel at the end):

   ```powershell
   python -c "from neuroguage.demo import DemoConfig, run_dry_demo; import json; print(json.dumps(run_dry_demo(DemoConfig.from_json('configs/demo_timing.json')), indent=2)[:400])"
   ```

3. Open `artifacts/release/reports/day5/dashboard_snapshot.png` on the
   projector; walk the audience through the EEG band, risk score,
   guardrail timeline, and rationale panels.

## 3. Extended rehearsal checklist

- `reports/day7/rehearsal_report.json` — final simulated run; confirm
  `within_budget == true` and `failure_count == 0`.
- `reports/day6/latency_benchmark.json` — confirm
  `within_budget == true` (NFR-001 ≤ 2 s).
- `reports/day6/stability_report.json` — confirm every scenario entry
  has `passed == true` (NFR-002).
- `checklists/mvp_acceptance_ac_001_to_ac_005.md` — expect five `[x]`
  entries with evidence links.

## 4. Optional stretch comparison

If the advanced model is present (`models/day7_advanced/`), pair the
MVP timeline with:

- `reports/day7/promotion_decision.md` — plain-English promotion note.
- `reports/day7/advanced_predictions.csv` — per-window advanced
  probabilities on the identical held-out sessions as the MVP.

## 5. Emergency fallbacks

- **Live stream down**: `scripts/day4_inference_guardrails.py` runs on
  the replay session included in the bundle; the dashboard and
  guardrail logic behave identically.
- **Dashboard fails to render**: present
  `reports/day5/dashboard_snapshot.png` as a static fallback.
- **Timing overshoot**: skip the optional "summary" stage last — the
  minimum 5-minute budget only requires the four task stages.

## 6. Smoke-test commands

```powershell
pytest -q tests/test_dashboard_day5.py tests/test_demo_day5.py
pytest -q tests/test_hardening_day6.py tests/test_release_day6.py
pytest -q tests/test_advanced_day7.py tests/test_promotion_day7.py tests/test_finalization_day7.py
```

## 7. Reproducibility

Regenerate every Day 7 artifact:

```powershell
python scripts/day7_stretch_and_final.py
```

The script prints PASS/FAIL for each T18/T19/T20 sub-step and finishes by
writing `reports/day7/day7_exit_gate.json`.
