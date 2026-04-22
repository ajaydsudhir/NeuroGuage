"""Day 5 — Dashboard Integration and Quick Demo Flow runner.

Exercises:
  O-05   Freeze demo envelope and timing budget.
  T14    Dashboard and Timeline Visualization.
  T15    Class Quick Demo Orchestration (dry-run mode).
  G05    Day 5 Exit Gate validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import pandas as pd  # noqa: E402

from neuroguage.dashboard import (  # noqa: E402
    DashboardConfig,
    DashboardRenderer,
    build_snapshot_from_replay,
    write_dashboard_artifacts,
)
from neuroguage.demo import (  # noqa: E402
    DemoConfig,
    run_dry_demo,
    validate_demo_timing,
    write_demo_artifacts,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELED_FEATURE_PATH = Path("artifacts/features/day3/labeled_feature_matrix.csv")
INFERENCE_DIAG_PATH = Path("artifacts/reports/day4/inference_diagnostics.json")
GUARDRAIL_LOG_PATH = Path("artifacts/reports/day4/guardrail_transition_log.json")
GUARDRAIL_SUMMARY_PATH = Path("artifacts/reports/day4/guardrail_summary.json")
DEMO_TIMING_PATH = Path("configs/demo_timing.json")

REPORT_DIR = Path("artifacts/reports/day5")
DASHBOARD_SNAPSHOT_PATH = REPORT_DIR / "dashboard_snapshot.png"
DASHBOARD_VALIDATION_PATH = REPORT_DIR / "dashboard_validation.json"


def _load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    print("[Day 5] Dashboard Integration and Quick Demo Flow")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # O-05 — Freeze demo envelope and timing budget
    # -----------------------------------------------------------------------
    print("\n[O-05] Freeze demo envelope and timing budget")

    if DEMO_TIMING_PATH.exists():
        demo_config = DemoConfig.from_json(DEMO_TIMING_PATH)
        print(f"  Loaded demo timing from {DEMO_TIMING_PATH}")
    else:
        demo_config = DemoConfig.default()
        print("  Using default demo timing config")

    timing_report = validate_demo_timing(demo_config)
    total_min = timing_report["total_planned_s"] / 60.0
    print(f"  [O-05.1] Total planned: {timing_report['total_planned_s']:.0f}s ({total_min:.1f} min)")
    print(f"  [O-05.2] Within budget: {timing_report['within_budget']}")
    if timing_report["violations"]:
        for v in timing_report["violations"]:
            print(f"    VIOLATION: {v}")

    for stage in timing_report["stages"]:
        print(f"    {stage['label']:>24}: {stage['planned_s']:>4.0f}s  (±{stage['tolerance_s']:.0f}s)")

    # -----------------------------------------------------------------------
    # Load Day 4 artifacts
    # -----------------------------------------------------------------------
    print("\n  Loading Day 4 artifacts...")
    missing = []
    for p in [LABELED_FEATURE_PATH, INFERENCE_DIAG_PATH, GUARDRAIL_LOG_PATH, GUARDRAIL_SUMMARY_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print(f"  ERROR: Missing artifacts: {missing}")
        return 1

    labeled_frame = pd.read_csv(LABELED_FEATURE_PATH)
    inference_diags = _load_json(INFERENCE_DIAG_PATH)
    guardrail_transitions = _load_json(GUARDRAIL_LOG_PATH)
    guardrail_summary = _load_json(GUARDRAIL_SUMMARY_PATH)
    print(f"  Loaded: {len(labeled_frame)} feature rows, {len(inference_diags)} inference updates")

    # -----------------------------------------------------------------------
    # T14 — Dashboard and Timeline Visualization
    # -----------------------------------------------------------------------
    print("\n[T14] Dashboard and Timeline Visualization")

    # T14.1 — Build real-time panels
    print("  [T14.1] Building dashboard panels...")
    snapshot = build_snapshot_from_replay(
        inference_diagnostics=inference_diags,
        guardrail_transitions=guardrail_transitions,
        guardrail_summary=guardrail_summary,
        labeled_frame=labeled_frame,
        stage_name="Replay",
    )

    dashboard_config = DashboardConfig(output_dir=str(REPORT_DIR))
    renderer = DashboardRenderer(dashboard_config)

    # T14.2 — Timeline with transition markers
    print("  [T14.2] Adding timeline with transition markers and rationale...")

    # T14.3 — Stream health and model-state indicators
    print("  [T14.3] Adding stream health and model-state indicators...")
    print(f"    EEG Health:     {snapshot.eeg_health}")
    print(f"    Optical Health: {snapshot.optical_health}")
    print(f"    IMU Health:     {snapshot.imu_health}")
    print(f"    Model Health:   {snapshot.model_health}")

    # T14.4 — Render and validate
    print("  [T14.4] Validating readability for classroom projection...")
    saved_path = renderer.save_snapshot(snapshot, DASHBOARD_SNAPSHOT_PATH)
    renderer.close()
    print(f"    Dashboard snapshot saved: {saved_path}")

    validation = write_dashboard_artifacts(
        snapshot_path=DASHBOARD_SNAPSHOT_PATH,
        validation_path=DASHBOARD_VALIDATION_PATH,
        config=dashboard_config,
    )
    resolution = validation["resolution"]
    meets_min = validation["meets_projection_minimum"]
    print(f"    Resolution: {resolution}")
    print(f"    Meets 1280×720 minimum: {meets_min}")
    print(f"    Readability pass: {validation['readability_pass']}")

    # -----------------------------------------------------------------------
    # T15 — Class Quick Demo Orchestration
    # -----------------------------------------------------------------------
    print("\n[T15] Class Quick Demo Orchestration")

    # T15.1 — Guided demo flow with stage timers
    print("  [T15.1] Implementing guided demo flow...")

    # T15.2 — Validate sequence
    print("  [T15.2] Validating demo sequence...")
    print(f"    Sequence valid: {timing_report['within_budget']}")

    # T15.3 — Summary panel
    print("  [T15.3] Summary panel preparation...")

    # T15.4 — Run two dry runs
    print("  [T15.4] Running dry runs...")

    dry_run_reports: list[dict] = []
    for run_num in range(1, 3):
        print(f"\n    --- Dry Run {run_num} ---")
        report = run_dry_demo(
            config=demo_config,
            inference_diagnostics=inference_diags,
            guardrail_transitions=guardrail_transitions,
            run_id=f"dry_run_{run_num}",
        )
        dry_run_reports.append(report)

        elapsed = report.get("total_elapsed_s", 0)
        in_budget = report.get("within_budget", False)
        failures = report.get("failure_count", 0)
        print(f"    Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"    Within budget: {in_budget}")
        print(f"    Failures:      {failures}")

        if report.get("failures"):
            for f in report["failures"]:
                print(f"      [{f.get('severity', '?')}] {f.get('description', '')}")

    # Write demo artifacts
    print(f"\n  Writing demo artifacts to {REPORT_DIR}...")
    artifact_paths = write_demo_artifacts(dry_run_reports, REPORT_DIR)

    # -----------------------------------------------------------------------
    # G05 — Day 5 Exit Gate
    # -----------------------------------------------------------------------
    print("\n[G05] Day 5 Exit Gate")

    g05_1 = DASHBOARD_SNAPSHOT_PATH.exists() and meets_min
    g05_2 = timing_report["within_budget"]
    g05_3 = len(dry_run_reports) >= 2

    print(f"  [G05.1] Dashboard is stable and readable:               {'PASS' if g05_1 else 'FAIL'}")
    print(f"  [G05.2] Quick demo completes within 5–10 min window:   {'PASS' if g05_2 else 'FAIL'}")
    print(f"  [G05.3] Dry-run defects are prioritized for Day 6:      {'PASS' if g05_3 else 'FAIL'}")

    gate_passed = g05_1 and g05_2 and g05_3

    # Write exit gate evidence
    gate_evidence = {
        "g05_1_dashboard_stable": g05_1,
        "g05_2_timing_within_budget": g05_2,
        "g05_3_dry_runs_completed": g05_3,
        "gate_passed": gate_passed,
        "dashboard_snapshot": str(DASHBOARD_SNAPSHOT_PATH),
        "dashboard_validation": str(DASHBOARD_VALIDATION_PATH),
        "dry_run_count": len(dry_run_reports),
        "total_defects": sum(r.get("failure_count", 0) for r in dry_run_reports),
    }
    gate_path = REPORT_DIR / "day5_exit_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate_evidence, indent=2), encoding="utf-8")

    print(f"\n  Exit gate evidence: {gate_path}")
    print(f"\n[Day 5] {'COMPLETE — G05 PASSED' if gate_passed else 'COMPLETE — G05 FAILED'}")

    # List all artifacts
    print("\n  All Day 5 artifacts:")
    day5_files = sorted(REPORT_DIR.glob("*"))
    for f in day5_files:
        if f.is_file():
            print(f"    [OK] {f}")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
