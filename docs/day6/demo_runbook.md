# NeuroGauge Class Quick Demo Runbook

Version: v1.0.0 (Day 6)
Audience: Project Owner + Instructor + Class audience
Environment: Conda environment `neuroguage` on Windows (RTX 4050, Muse S Athena).

This runbook turns the PRD "Class Quick Demo" (Section 10.2) into a
repeatable 5 to 10 minute walkthrough with explicit timing targets,
operator actions, and fallback procedures.

## 1. Pre-Flight Checklist (T-10 minutes)

Run each item from the project root in PowerShell.

- [ ] Hardware
  - Muse S Athena charged > 70%, firmware OK, Bluetooth pairing verified.
  - Host machine on AC power; display connected to projector @ >= 1280x720.
- [ ] Environment
  - `conda activate neuroguage`
  - `python scripts/verify_environment.py` should print "OK" and list pinned
    versions for numpy, scipy, scikit-learn, torch, matplotlib.
- [ ] Artifacts
  - `artifacts/models/day4/ablations/<selected_variant>/` present.
  - `artifacts/reports/day4/evaluation_summary.json` present.
  - `artifacts/reports/day5/dashboard_snapshot.png` present (rehearsal proof).
- [ ] Storage
  - Close other GPU/CPU-heavy apps.
  - Ensure `logs/` and `artifacts/reports/day6/` are writable.
- [ ] Dry-rehearsal
  - Run `python scripts/day5_dashboard_demo.py` and confirm `G05 PASSED`
    within the last 24 h.

## 2. Timing Targets (O-05 Frozen Envelope)

Total envelope: **5 to 10 minutes** (hard cap). The frozen per-stage budget
lives in `configs/demo_timing.json` and is validated by
`neuroguage.demo.validate_demo_timing`.

| Order | Stage | Planned (s) | Tolerance (±s) | Operator Action |
|------:|-------|-----------:|---------------:|----------------|
| 1 | Baseline Rest | 60 | 12 | "Eyes open, relax, no reading." |
| 2 | Sustained Attention | 120 | 24 | Start reaction-time stimulus. |
| 3 | 2-Back Working Memory | 120 | 24 | Start n-back task on screen. |
| 4 | Code Comprehension | 150 | 30 | Show code block; ask questions at t+90s. |
| 5 | Timeline Summary | 45 | 9 | Walk the audience through guardrail timeline. |
|   | **Total** | **495** (8:15) | | Must remain in [300, 600]. |

If any stage starts running over the tolerance, the operator must short-cut
that stage (not extend) so the 10-minute hard cap is respected.

## 3. Run Sequence (Live)

1. **Start the dashboard** (replay-mode safe fallback):

   ```powershell
   conda activate neuroguage
   python scripts/day5_dashboard_demo.py
   ```

   The Matplotlib window should render within ~5 s and show all four
   real-time panels plus the guardrail timeline.

2. **Start live inference** (when Muse stream is healthy):

   ```powershell
   python scripts/run_live.py --config configs/demo_timing.json
   ```

   The log line `Inference #1 | risk=... | latency=...s` should appear
   within 2 s of stream connect.

3. **Advance stages** using the orchestrator. Stage timers drive the
   on-screen countdown and auto-log transitions to
   `artifacts/reports/day5/dry_run_<n>_report.json` on exit.

4. **Narrate guardrail transitions** as they happen:
   - GREEN -> YELLOW: "Fatigue risk is rising; confidence is still moderate."
   - YELLOW -> RED: "Sustained risk over the RED threshold; take a break."
   - No oscillation beyond 4 transitions/minute under real audience noise
     (see `artifacts/reports/day6/stability_report.json`).

5. **Timeline Summary (last 30–60 s)**: keep the dashboard visible, call
   out two or three non-GREEN transitions and read their rationale text
   aloud from the Guardrail Timeline panel.

## 4. Fallback Procedures

| Failure | Symptom | Action |
|--------|--------|--------|
| F-01 Muse stream drops | `eeg_stale` fault flag appears repeatedly | Continue demo in **replay mode** with `python scripts/replay.py --session <latest>`. Announce the switch to the class. |
| F-02 Dashboard freeze | No panel update > 3 s | Press `r` to force-redraw; if unresolved within 10 s, fall back to the pre-rendered `artifacts/reports/day5/dashboard_snapshot.png` on-screen. |
| F-03 Model load error | `InferenceError: model not found` | Switch to Day 3 baseline: `python scripts/run_live.py --model artifacts/models/day3`. |
| F-04 Excessive RED flashes (> 4/min) | Alert state oscillating visibly | Immediately stop, walk through `docs/day6/limitations_and_risks.md#alert-stability` to explain, and resume after 15 s. |
| F-05 Timing overrun | Stage elapsed > planned + tolerance | Skip to next stage by calling `advance_stage()` (Ctrl+N hotkey in demo window). Do **not** extend the envelope past 10 minutes. |
| F-06 Projector disconnect | Display loss | Use mirrored laptop screen; resume with `python scripts/day5_dashboard_demo.py`. Narrate from the snapshot. |
| F-07 Demo script crash | Python traceback | Capture stack to `logs/`, restart from the current stage using `--resume-from <stage>` on the runner. |

## 5. Post-Demo Checklist

- [ ] Copy `artifacts/reports/day5/dry_run_*.json` to `artifacts/release/`.
- [ ] Re-run `python scripts/day6_hardening_release.py` to refresh the
      evidence index.
- [ ] Archive logs under `logs/demo_<date>/`.
- [ ] Review `docs/day6/limitations_and_risks.md` for any newly observed
      issues; log them into the risk log template.

## 6. References

- `docs/PRD.md` Section 10.2 (Class Quick Demo Mode).
- `configs/demo_timing.json` (frozen stage envelope).
- `artifacts/reports/day5/dashboard_validation.json` (projection readability).
- `artifacts/reports/day6/stability_report.json` (alert stability evidence).
- `artifacts/reports/day6/latency_benchmark.json` (NFR-001 evidence).
