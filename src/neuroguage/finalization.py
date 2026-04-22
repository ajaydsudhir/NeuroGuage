"""Day 7 — T20  Final Packaging and Demo Rehearsal.

Builds the release-candidate bundle by extending the Day 6 MVP release with
Day 7 stretch artifacts, runs one final rehearsal under the class-demo
timing constraints, and publishes the final status summary consumed by the
G07 exit gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .release import ReleaseManifest, build_release_bundle

LOGGER = logging.getLogger(__name__)


# ===========================================================================
# T20.1 — Stretch deferral gate
# ===========================================================================

@dataclass
class StretchStatus:
    stretch_attempted: bool
    deferred: bool
    reason: str
    decision: str | None = None


def resolve_stretch_status(
    day6_exit_gate: dict[str, Any] | None,
    promotion_decision: dict[str, Any] | None,
) -> StretchStatus:
    """Implement T20.1 - decide whether to proceed with stretch or defer.

    Logic:
      - If the Day 6 gate blocked the stretch, T18 and T19 are marked
        deferred and the final package proceeds with MVP only.
      - If stretch ran and `promotion_decision` exists, we follow its
        outcome (PROMOTE_ADVANCED or KEEP_MVP); neither path is "deferred"
        because both produced evidence.
      - If Day 6 allowed stretch but promotion artifacts are missing,
        stretch is considered deferred with an explicit reason.
    """
    gate = day6_exit_gate or {}
    decision = str(gate.get("phase_gate_decision", ""))
    decision_upper = decision.upper()
    blocked = (
        "NO-GO" in decision_upper
        or "NO GO" in decision_upper
        or "BLOCK" in decision_upper
        or "DEFER" in decision_upper
    )
    allowed = (not blocked) and ("GO" in decision_upper)

    if not allowed:
        return StretchStatus(
            stretch_attempted=False,
            deferred=True,
            reason=(
                f"Day 6 phase-gate decision was '{decision or 'UNKNOWN'}'. "
                "Stretch tasks T18/T19 deferred; final package ships MVP only."
            ),
            decision=None,
        )

    if promotion_decision is None:
        return StretchStatus(
            stretch_attempted=False,
            deferred=True,
            reason=(
                "Day 6 allowed stretch, but no promotion artifact "
                "(promotion_decision.json) was produced. Deferring T18/T19."
            ),
            decision=None,
        )

    outcome = str(promotion_decision.get("decision", "KEEP_MVP"))
    return StretchStatus(
        stretch_attempted=True,
        deferred=False,
        reason=(
            f"Stretch completed with decision '{outcome}'. "
            f"Rationale: {promotion_decision.get('rationale', '')}".strip()
        ),
        decision=outcome,
    )


# ===========================================================================
# T20.2 — Assemble the final demo folder
# ===========================================================================

def build_final_demo_bundle(
    release_dir: Path,
    version: str,
    commit: str | None,
    stretch: StretchStatus,
    day7_report_dir: Path,
    advanced_model_dir: Path,
) -> ReleaseManifest:
    """Assemble the final demo folder (T20.2).

    Extends the Day 6 MVP release bundle with:
      - Day 7 advanced-model artifacts (if stretch not deferred).
      - Day 7 comparison + promotion decision evidence.
      - Quick-start notes + final status summary.
    """
    release_dir = Path(release_dir)
    day7_report_dir = Path(day7_report_dir)
    advanced_model_dir = Path(advanced_model_dir)

    extras: list[tuple[Path, Path, str]] = [
        (
            day7_report_dir / "rehearsal_report.json",
            release_dir / "reports" / "day7" / "rehearsal_report.json",
            "Day 7 final rehearsal report",
        ),
        (
            day7_report_dir / "final_status_summary.json",
            release_dir / "reports" / "day7" / "final_status_summary.json",
            "Day 7 final status summary",
        ),
        (
            day7_report_dir / "final_status_summary.md",
            release_dir / "reports" / "day7" / "final_status_summary.md",
            "Day 7 final status narrative",
        ),
        (
            day7_report_dir / "day7_exit_gate.json",
            release_dir / "reports" / "day7" / "day7_exit_gate.json",
            "Day 7 exit gate evidence",
        ),
        (
            Path("docs/day7/quick_start.md"),
            release_dir / "docs" / "quick_start.md",
            "T20.2 quick-start notes for class demo",
        ),
    ]

    if not stretch.deferred:
        extras.extend(
            [
                (
                    advanced_model_dir,
                    release_dir / "models" / "day7_advanced",
                    "Day 7 advanced BiGRU-attention model artifacts",
                ),
                (
                    day7_report_dir / "advanced_training_metrics.json",
                    release_dir / "reports" / "day7" / "advanced_training_metrics.json",
                    "Day 7 advanced training metrics",
                ),
                (
                    day7_report_dir / "advanced_training_log.json",
                    release_dir / "reports" / "day7" / "advanced_training_log.json",
                    "Day 7 advanced training epoch log",
                ),
                (
                    day7_report_dir / "advanced_predictions.csv",
                    release_dir / "reports" / "day7" / "advanced_predictions.csv",
                    "Day 7 advanced held-out predictions",
                ),
                (
                    day7_report_dir / "advanced_latency_benchmark.json",
                    release_dir / "reports" / "day7" / "advanced_latency_benchmark.json",
                    "Day 7 advanced latency benchmark",
                ),
                (
                    day7_report_dir / "advanced_stability_report.json",
                    release_dir / "reports" / "day7" / "advanced_stability_report.json",
                    "Day 7 advanced stability report",
                ),
                (
                    day7_report_dir / "promotion_decision.json",
                    release_dir / "reports" / "day7" / "promotion_decision.json",
                    "Day 7 promotion decision (T19.4)",
                ),
                (
                    day7_report_dir / "promotion_decision.md",
                    release_dir / "reports" / "day7" / "promotion_decision.md",
                    "Day 7 promotion decision narrative",
                ),
            ]
        )

    manifest = build_release_bundle(
        release_dir=release_dir,
        version=version,
        commit=commit,
        extra_entries=extras,
    )
    return manifest


# ===========================================================================
# T20.3 — Final rehearsal under class-demo timing constraints
# ===========================================================================

@dataclass
class RehearsalResult:
    within_budget: bool
    total_elapsed_s: float
    stage_count: int
    failure_count: int
    latency_within_budget: bool
    stability_within_budget: bool
    timing_violations: list[str]
    report: dict[str, Any] = field(default_factory=dict)


def run_final_rehearsal(
    *,
    demo_config_path: Path | None,
    inference_diagnostics_path: Path,
    guardrail_log_path: Path,
    latency_benchmark: dict[str, Any],
    stability_reports: list[dict[str, Any]],
    run_id: str = "final_rehearsal",
) -> RehearsalResult:
    """Execute one full rehearsal under class-demo timing constraints (T20.3).

    Reuses the Day 5 demo dry-run harness so the rehearsal is an actual
    simulated end-to-end pass through the PRD-mandated quick-demo sequence
    rather than a rubber-stamp.  Includes objective latency and stability
    checks against the most recent benchmarks.
    """
    # Imported lazily to avoid dragging matplotlib into release-only imports
    from .demo import DemoConfig, run_dry_demo

    if demo_config_path and Path(demo_config_path).exists():
        config = DemoConfig.from_json(Path(demo_config_path))
    else:
        config = DemoConfig.default()

    inference_updates: list[dict[str, Any]] = []
    if inference_diagnostics_path and Path(inference_diagnostics_path).exists():
        raw = json.loads(Path(inference_diagnostics_path).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            inference_updates = raw
        elif isinstance(raw, dict):
            inference_updates = raw.get("updates", [])

    guardrail_transitions: list[dict[str, Any]] = []
    if guardrail_log_path and Path(guardrail_log_path).exists():
        raw = json.loads(Path(guardrail_log_path).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            guardrail_transitions = raw
        elif isinstance(raw, dict):
            guardrail_transitions = raw.get("transitions", [])

    report = run_dry_demo(
        config=config,
        inference_diagnostics=inference_updates,
        guardrail_transitions=guardrail_transitions,
        run_id=run_id,
        simulate_realtime=False,
    )

    latency_within_budget = bool(latency_benchmark.get("within_budget", False))
    stability_within_budget = all(bool(r.get("passed", False)) for r in stability_reports)

    timing = report.get("timing_validation", {})
    timing_violations = list(timing.get("violations", []) or [])

    return RehearsalResult(
        within_budget=bool(report.get("within_budget", False)),
        total_elapsed_s=float(report.get("total_elapsed_s", 0.0)),
        stage_count=len(report.get("stages", []) or []),
        failure_count=int(report.get("failure_count", 0)),
        latency_within_budget=latency_within_budget,
        stability_within_budget=stability_within_budget,
        timing_violations=timing_violations,
        report=report,
    )


def write_rehearsal_report(result: RehearsalResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "within_budget": result.within_budget,
        "total_elapsed_s": result.total_elapsed_s,
        "total_elapsed_min": result.total_elapsed_s / 60.0,
        "stage_count": result.stage_count,
        "failure_count": result.failure_count,
        "latency_within_budget": result.latency_within_budget,
        "stability_within_budget": result.stability_within_budget,
        "timing_violations": result.timing_violations,
        "dry_run_report": result.report,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ===========================================================================
# T20.4 — Release-candidate freeze + final status summary
# ===========================================================================

def build_final_status_summary(
    *,
    rc_version: str,
    commit: str | None,
    stretch: StretchStatus,
    rehearsal: RehearsalResult,
    acceptance_results: dict[str, Any],
    promotion_decision: dict[str, Any] | None,
    latency_benchmark: dict[str, Any],
    stability_reports: list[dict[str, Any]],
    mvp_evaluation: dict[str, Any],
    release_manifest_path: Path,
    release_dir: Path,
) -> dict[str, Any]:
    """Compose the final status summary consumed by G07.

    All fields are boolean/scalar so downstream gates and instructor-facing
    narrative can assert against them without re-parsing artifact JSON.
    """
    ac_passed = int(acceptance_results.get("passed_count", 0))
    ac_total = int(acceptance_results.get("total_count", 0))

    promoted = bool(promotion_decision.get("promoted")) if promotion_decision else False

    summary: dict[str, Any] = {
        "rc_version": rc_version,
        "commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(release_dir),
        "release_manifest": str(release_manifest_path),
        "mvp": {
            "selected_variant": mvp_evaluation.get("selected_variant"),
            "test_metrics": (
                mvp_evaluation.get("variant_scores", {})
                .get(mvp_evaluation.get("selected_variant", ""), {})
                .get("test_late_fusion", {})
            ),
        },
        "acceptance_ac_001_to_ac_005": {
            "passed_count": ac_passed,
            "total_count": ac_total,
            "all_passed": ac_passed == ac_total and ac_total > 0,
            "criteria": acceptance_results.get("criteria", []),
        },
        "latency": {
            "mean_ms": float(latency_benchmark.get("mean_s", 0.0)) * 1000,
            "p95_ms": float(latency_benchmark.get("p95_s", 0.0)) * 1000,
            "max_ms": float(latency_benchmark.get("max_s", 0.0)) * 1000,
            "within_budget": bool(latency_benchmark.get("within_budget", False)),
            "budget_s": float(latency_benchmark.get("budget_s", 2.0)),
        },
        "stability": {
            "all_passed": all(bool(r.get("passed", False)) for r in stability_reports),
            "scenarios": [
                {
                    "scenario": r.get("scenario"),
                    "passed": bool(r.get("passed", False)),
                    "transitions_per_minute": float(r.get("transitions_per_minute", 0.0)),
                    "max_consecutive_flips": int(r.get("max_consecutive_flips", 0)),
                }
                for r in stability_reports
            ],
        },
        "rehearsal": {
            "within_budget": rehearsal.within_budget,
            "total_elapsed_s": rehearsal.total_elapsed_s,
            "total_elapsed_min": rehearsal.total_elapsed_s / 60.0,
            "stage_count": rehearsal.stage_count,
            "failure_count": rehearsal.failure_count,
            "latency_within_budget": rehearsal.latency_within_budget,
            "stability_within_budget": rehearsal.stability_within_budget,
            "timing_violations": rehearsal.timing_violations,
        },
        "stretch": {
            "attempted": stretch.stretch_attempted,
            "deferred": stretch.deferred,
            "decision": stretch.decision,
            "promoted": promoted,
            "reason": stretch.reason,
        },
    }
    return summary


def render_final_status_markdown(summary: dict[str, Any]) -> str:
    """Render the T20.4 final status summary as a Markdown narrative."""
    lines: list[str] = []
    lines.append("# NeuroGauge Final Status Summary (T20.4)")
    lines.append("")
    lines.append(f"- **Release candidate**: `{summary['rc_version']}`")
    if summary.get("commit"):
        lines.append(f"- **Git commit**: `{summary['commit']}`")
    lines.append(f"- **Created**: {summary['created_at']}")
    lines.append(f"- **Release bundle**: `{summary['release_dir']}`")
    lines.append("")

    ac = summary["acceptance_ac_001_to_ac_005"]
    lines.append("## MVP Acceptance (AC-001..AC-005)")
    lines.append("")
    lines.append(f"- Passed: **{ac['passed_count']} / {ac['total_count']}**")
    for c in ac.get("criteria", []):
        mark = "PASS" if c.get("passed") else "FAIL"
        lines.append(f"- [{mark}] **{c.get('id')}** — {c.get('description')}")
    lines.append("")

    lines.append("## Runtime Quality")
    lines.append("")
    lines.append(f"- Latency max: **{summary['latency']['max_ms']:.2f} ms** (budget {summary['latency']['budget_s']*1000:.0f} ms)")
    lines.append(f"- Latency within budget: **{summary['latency']['within_budget']}**")
    lines.append(f"- Stability all scenarios passed: **{summary['stability']['all_passed']}**")
    lines.append("")

    r = summary["rehearsal"]
    lines.append("## Final Rehearsal")
    lines.append("")
    lines.append(
        f"- Total elapsed: **{r['total_elapsed_min']:.2f} min** "
        f"({r['total_elapsed_s']:.1f}s across {r['stage_count']} stages)"
    )
    lines.append(f"- Within class-demo budget: **{r['within_budget']}**")
    lines.append(f"- Failures logged: **{r['failure_count']}**")
    if r.get("timing_violations"):
        lines.append("- Timing violations:")
        for v in r["timing_violations"]:
            lines.append(f"  - {v}")
    lines.append("")

    s = summary["stretch"]
    lines.append("## Stretch Outcome")
    lines.append("")
    if s.get("deferred"):
        lines.append("- **Stretch deferred.**")
    elif s.get("promoted"):
        lines.append("- **Advanced model PROMOTED** per FR-026/AC-007/AC-008.")
    else:
        lines.append("- **MVP retained** as the production demo model; advanced prototype preserved for future work.")
    lines.append(f"- Reason: {s.get('reason', '')}")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_final_status_summary(
    summary: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(render_final_status_markdown(summary), encoding="utf-8")
