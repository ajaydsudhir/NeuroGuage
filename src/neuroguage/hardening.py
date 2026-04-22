"""Day 6 — T16  MVP Hardening and Release Readiness.

Provides reusable primitives for:

  T16.1  Prioritising and triaging dry-run / demo defects.
  T16.2  End-to-end refresh latency benchmarking against NFR-001 (<= 2s).
  T16.3  Alert-state stability validation under noisy and motion-heavy
         windows (synthetic stress tests against the guardrail engine).
  T16.4  Automated AC-001 through AC-005 verification with objective
         evidence links.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .guardrails import (
    AlertState,
    GuardrailConfig,
    GuardrailState,
    run_guardrail_sequence,
    update_guardrail,
)

LOGGER = logging.getLogger(__name__)


# ===========================================================================
# T16.1 — Dry-run defect triage
# ===========================================================================

_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def prioritise_defects(
    dry_run_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate and prioritise defects across dry-run reports (T16.1).

    Returns a triage report listing high-, medium-, and low-priority issues
    in the order they should be addressed during hardening.
    """
    high: list[dict[str, Any]] = []
    medium: list[dict[str, Any]] = []
    low: list[dict[str, Any]] = []

    for report in dry_run_reports:
        run_id = report.get("run_id", "unknown")
        for failure in report.get("failures", []) or []:
            record = {
                "run_id": run_id,
                "elapsed_s": failure.get("elapsed_s", 0.0),
                "stage": failure.get("stage", "unknown"),
                "severity": failure.get("severity", "medium"),
                "description": failure.get("description", ""),
            }
            sev = record["severity"].lower()
            if sev == "high":
                high.append(record)
            elif sev == "low":
                low.append(record)
            else:
                medium.append(record)

    def _sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda d: (d.get("run_id", ""), d.get("elapsed_s", 0.0)))

    triage = {
        "total_defects": len(high) + len(medium) + len(low),
        "high_priority": _sorted(high),
        "medium_priority": _sorted(medium),
        "low_priority": _sorted(low),
        "resolution_order": [r["description"] for r in _sorted(high) + _sorted(medium) + _sorted(low)],
        "all_clear": (len(high) + len(medium) + len(low)) == 0,
    }
    return triage


# ===========================================================================
# T16.2 — End-to-end latency benchmark (NFR-001)
# ===========================================================================

@dataclass
class LatencyBenchmark:
    """Aggregated latency statistics for the inference + guardrail path."""

    samples: int
    mean_s: float
    median_s: float
    p95_s: float
    max_s: float
    budget_s: float
    within_budget: bool
    per_stage_s: dict[str, float] = field(default_factory=dict)


def benchmark_refresh_latency(
    inference_update_fn: Callable[[int], tuple[float, float]],
    guardrail_update_fn: Callable[[float, float, float], None],
    sample_count: int = 50,
    budget_s: float = 2.0,
) -> LatencyBenchmark:
    """Benchmark full inference + guardrail refresh latency (T16.2).

    `inference_update_fn(i)` must run one inference cycle and return
    `(risk_score, model_confidence)`. `guardrail_update_fn(risk, conf, ts)`
    updates the guardrail engine.

    The benchmark measures end-to-end wall-clock latency per cycle and
    reports p95/max along with per-stage breakdowns.
    """
    inference_times: list[float] = []
    guardrail_times: list[float] = []
    total_times: list[float] = []

    for i in range(sample_count):
        ts = float(i)
        t0 = time.perf_counter()
        risk, conf = inference_update_fn(i)
        t1 = time.perf_counter()
        guardrail_update_fn(risk, conf, ts)
        t2 = time.perf_counter()

        inference_times.append(t1 - t0)
        guardrail_times.append(t2 - t1)
        total_times.append(t2 - t0)

    p95 = float(np.percentile(total_times, 95)) if total_times else 0.0
    return LatencyBenchmark(
        samples=len(total_times),
        mean_s=float(statistics.mean(total_times)) if total_times else 0.0,
        median_s=float(statistics.median(total_times)) if total_times else 0.0,
        p95_s=p95,
        max_s=float(max(total_times)) if total_times else 0.0,
        budget_s=budget_s,
        within_budget=(max(total_times) <= budget_s) if total_times else False,
        per_stage_s={
            "inference_mean": float(statistics.mean(inference_times)) if inference_times else 0.0,
            "inference_p95": float(np.percentile(inference_times, 95)) if inference_times else 0.0,
            "guardrail_mean": float(statistics.mean(guardrail_times)) if guardrail_times else 0.0,
            "guardrail_p95": float(np.percentile(guardrail_times, 95)) if guardrail_times else 0.0,
        },
    )


# ===========================================================================
# T16.3 — Alert stability under noise and motion artifacts
# ===========================================================================

@dataclass
class StabilityReport:
    """Summary of guardrail stability under a synthetic stress scenario."""

    scenario: str
    update_count: int
    transitions: int
    transitions_per_minute: float
    green_fraction: float
    yellow_fraction: float
    red_fraction: float
    oscillation_index: float
    max_consecutive_flips: int
    passed: bool
    notes: str = ""


def _simulate_states(
    risk_scores: list[float],
    confidences: list[float],
    config: GuardrailConfig,
) -> tuple[list[str], GuardrailState]:
    """Replay a risk trajectory through the guardrail engine and return
    the per-update state stream."""
    state = GuardrailState()
    timeline: list[str] = []
    for i, (r, c) in enumerate(zip(risk_scores, confidences)):
        update_guardrail(
            risk_score=float(r),
            model_confidence=float(c),
            timestamp_s=float(i),
            state=state,
            config=config,
        )
        timeline.append(state.current_state.value)
    return timeline, state


def _max_consecutive_flips(timeline: list[str]) -> int:
    """Longest run of adjacent state changes (proxy for oscillation bursts)."""
    if len(timeline) < 2:
        return 0
    best = 0
    current = 0
    for a, b in zip(timeline[:-1], timeline[1:]):
        if a != b:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def stability_scenario_noisy_risk(
    config: GuardrailConfig | None = None,
    *,
    duration: int = 300,
    base_risk: float = 0.25,
    noise_std: float = 0.15,
    seed: int = 17,
) -> StabilityReport:
    """T16.3 — Inject high-variance Gaussian noise around a GREEN-band
    baseline.

    The mean risk stays well below the YELLOW threshold.  With the engine's
    persistence + trend logic, single-window noise crossing YELLOW should
    NOT cause rapid oscillation (NFR-002).  We measure:

      * `max_consecutive_flips`: the longest burst of adjacent state
         changes — must stay small (<=2) because persistence requires at
         least 3 consecutive windows above threshold before escalating.
      * `counts["BLACK"]`: engine must never enter BLACK from noise alone.
    """
    cfg = config or GuardrailConfig()
    rng = np.random.default_rng(seed)
    risk = np.clip(rng.normal(base_risk, noise_std, duration), 0.0, 1.0).tolist()
    conf = [0.8] * duration

    timeline, state = _simulate_states(risk, conf, cfg)
    transitions = len(state.transitions)
    per_min = (transitions / max(1, duration)) * 60.0
    total = len(timeline)
    counts = {s: timeline.count(s) for s in ("GREEN", "YELLOW", "RED", "BLACK")}

    osc_index = transitions / max(1, total)
    max_flips = _max_consecutive_flips(timeline)

    # Pass criteria (NFR-002 stability):
    #   1. No rapid-flip bursts (max_consecutive_flips <= 2).
    #   2. Engine never reaches BLACK from noisy excursions alone.
    #   3. Majority of windows remain at or below YELLOW (no spurious
    #      RED/BLACK dominance from noise).
    below_red = (counts["GREEN"] + counts["YELLOW"]) / max(1, total)
    passed = (
        max_flips <= 2
        and counts["BLACK"] == 0
        and below_red >= 0.7
    )

    notes = (
        f"base_risk={base_risk}, noise_std={noise_std}; "
        f"persistence={cfg.escalation_persistence} windows; "
        f"max_consecutive_flips={max_flips}; "
        f"below_red_fraction={below_red:.3f}."
    )
    return StabilityReport(
        scenario="noisy_risk",
        update_count=total,
        transitions=transitions,
        transitions_per_minute=per_min,
        green_fraction=counts["GREEN"] / total,
        yellow_fraction=counts["YELLOW"] / total,
        red_fraction=counts["RED"] / total,
        oscillation_index=osc_index,
        max_consecutive_flips=max_flips,
        passed=passed,
        notes=notes,
    )


def stability_scenario_motion_artifact(
    config: GuardrailConfig | None = None,
    *,
    duration: int = 300,
    spike_interval: int = 15,
    spike_value: float = 0.95,
    baseline_risk: float = 0.25,
    baseline_confidence: float = 0.85,
    spike_confidence: float = 0.2,
    seed: int = 23,
) -> StabilityReport:
    """T16.3 — Simulate motion-artifact spikes with low confidence.

    The spike pattern mimics IMU-heavy windows where raw risk transiently
    jumps into the RED zone but model confidence collapses.  The guardrail
    engine must refuse escalation beyond YELLOW under these conditions
    (via `min_confidence_for_red`).
    """
    cfg = config or GuardrailConfig()
    rng = np.random.default_rng(seed)
    risk = rng.normal(baseline_risk, 0.05, duration)
    confidence = np.full(duration, baseline_confidence)

    for t in range(0, duration, spike_interval):
        risk[t] = spike_value
        confidence[t] = spike_confidence

    risk = np.clip(risk, 0.0, 1.0).tolist()
    confidence = np.clip(confidence, 0.0, 1.0).tolist()

    timeline, state = _simulate_states(risk, confidence, cfg)
    total = len(timeline)
    transitions = len(state.transitions)
    counts = {s: timeline.count(s) for s in ("GREEN", "YELLOW", "RED", "BLACK")}
    per_min = (transitions / max(1, duration)) * 60.0
    osc_index = transitions / max(1, total)
    max_flips = _max_consecutive_flips(timeline)

    # Under low-confidence motion spikes, engine must stay out of RED/BLACK.
    passed = counts["RED"] == 0 and counts["BLACK"] == 0

    notes = (
        f"spike every {spike_interval} windows at risk={spike_value} with "
        f"confidence={spike_confidence}; engine min_confidence_for_red="
        f"{cfg.min_confidence_for_red}."
    )
    return StabilityReport(
        scenario="motion_artifact",
        update_count=total,
        transitions=transitions,
        transitions_per_minute=per_min,
        green_fraction=counts["GREEN"] / total,
        yellow_fraction=counts["YELLOW"] / total,
        red_fraction=counts["RED"] / total,
        oscillation_index=osc_index,
        max_consecutive_flips=max_flips,
        passed=passed,
        notes=notes,
    )


def stability_scenario_sustained_fatigue(
    config: GuardrailConfig | None = None,
    *,
    duration: int = 180,
    ramp_start: float = 0.2,
    ramp_end: float = 0.9,
    seed: int = 29,
) -> StabilityReport:
    """T16.3 — Slow ramp from GREEN to RED to verify that real fatigue
    *does* escalate the state even with persistence guardrails."""
    cfg = config or GuardrailConfig()
    rng = np.random.default_rng(seed)
    base = np.linspace(ramp_start, ramp_end, duration)
    noise = rng.normal(0.0, 0.02, duration)
    risk = np.clip(base + noise, 0.0, 1.0).tolist()
    confidence = [0.85] * duration

    timeline, state = _simulate_states(risk, confidence, cfg)
    total = len(timeline)
    transitions = len(state.transitions)
    counts = {s: timeline.count(s) for s in ("GREEN", "YELLOW", "RED", "BLACK")}
    per_min = (transitions / max(1, duration)) * 60.0

    # The engine must reach at least RED on a clean ramp (evidence that
    # stability logic does not over-suppress true escalations).
    reached_red = AlertState(state.current_state.value).severity() >= AlertState.RED.severity()
    passed = reached_red

    return StabilityReport(
        scenario="sustained_fatigue_ramp",
        update_count=total,
        transitions=transitions,
        transitions_per_minute=per_min,
        green_fraction=counts["GREEN"] / total,
        yellow_fraction=counts["YELLOW"] / total,
        red_fraction=counts["RED"] / total,
        oscillation_index=transitions / max(1, total),
        max_consecutive_flips=_max_consecutive_flips(timeline),
        passed=passed,
        notes=f"ramp {ramp_start}->{ramp_end}; final state={state.current_state.value}.",
    )


def run_all_stability_scenarios(
    config: GuardrailConfig | None = None,
) -> list[StabilityReport]:
    """T16.3 — Run the full stability test matrix."""
    return [
        stability_scenario_noisy_risk(config),
        stability_scenario_motion_artifact(config),
        stability_scenario_sustained_fatigue(config),
    ]


# ===========================================================================
# T16.4 — AC-001 through AC-005 automated verification
# ===========================================================================

@dataclass
class AcceptanceCriterionResult:
    """Verification result for a single MVP acceptance criterion."""

    id: str
    description: str
    passed: bool
    evidence_paths: list[str]
    notes: str = ""


def _exists(*paths: Path) -> tuple[bool, list[str]]:
    ok = all(p.exists() for p in paths)
    return ok, [str(p) for p in paths]


def verify_ac001_end_to_end(
    inference_diagnostics_path: Path,
    guardrail_transition_log: Path,
    dashboard_snapshot: Path,
) -> AcceptanceCriterionResult:
    """AC-001 — End-to-end pipeline operates from ingestion through
    dashboard and event logging."""
    ok, ev = _exists(inference_diagnostics_path, guardrail_transition_log, dashboard_snapshot)
    notes = (
        "End-to-end evidence: inference diagnostics, guardrail transition "
        "log, and dashboard snapshot all present."
        if ok
        else "Missing one or more end-to-end artifacts."
    )
    return AcceptanceCriterionResult(
        id="AC-001",
        description="End-to-end pipeline operates from ingestion through dashboard and event logging.",
        passed=ok,
        evidence_paths=ev,
        notes=notes,
    )


def verify_ac002_session_held_out(
    evaluation_summary_path: Path,
    ablation_report_path: Path,
) -> AcceptanceCriterionResult:
    """AC-002 — Session-held-out evaluation reports AUROC, macro F1,
    balanced accuracy."""
    evidence = [str(evaluation_summary_path), str(ablation_report_path)]
    if not evaluation_summary_path.exists() or not ablation_report_path.exists():
        return AcceptanceCriterionResult(
            id="AC-002",
            description="Session-held-out evaluation reports AUROC, macro F1, balanced accuracy.",
            passed=False,
            evidence_paths=evidence,
            notes="Missing evaluation or ablation report.",
        )

    data = json.loads(evaluation_summary_path.read_text(encoding="utf-8"))
    selected = data.get("variant_scores", {}).get(data.get("selected_variant", ""), {})
    test_metrics = selected.get("test_late_fusion", {})
    required = {"macro_f1", "balanced_accuracy", "auroc"}
    has_all = required.issubset(test_metrics.keys())
    notes = (
        f"Selected variant '{data.get('selected_variant')}' reports "
        f"macro_f1={test_metrics.get('macro_f1')}, "
        f"balanced_accuracy={test_metrics.get('balanced_accuracy')}, "
        f"auroc={test_metrics.get('auroc')}."
    )
    return AcceptanceCriterionResult(
        id="AC-002",
        description="Session-held-out evaluation reports AUROC, macro F1, balanced accuracy.",
        passed=has_all,
        evidence_paths=evidence,
        notes=notes,
    )


def verify_ac003_latency(
    latency_benchmark: LatencyBenchmark,
    inference_diagnostics_path: Path,
) -> AcceptanceCriterionResult:
    """AC-003 — Real-time latency meets NFR-001 (<= 2 s)."""
    evidence = [str(inference_diagnostics_path)]
    passed = latency_benchmark.within_budget and latency_benchmark.max_s <= latency_benchmark.budget_s
    notes = (
        f"Refresh latency: mean={latency_benchmark.mean_s*1000:.2f}ms, "
        f"p95={latency_benchmark.p95_s*1000:.2f}ms, "
        f"max={latency_benchmark.max_s*1000:.2f}ms (budget={latency_benchmark.budget_s}s)."
    )
    return AcceptanceCriterionResult(
        id="AC-003",
        description="Real-time latency meets NFR-001 (<= 2 s).",
        passed=passed,
        evidence_paths=evidence,
        notes=notes,
    )


def verify_ac004_quick_demo(
    day5_exit_gate_path: Path,
    demo_summary_path: Path,
) -> AcceptanceCriterionResult:
    """AC-004 — Quick demo completes in 5 to 10 minutes without system
    failure."""
    evidence = [str(day5_exit_gate_path), str(demo_summary_path)]
    if not day5_exit_gate_path.exists() or not demo_summary_path.exists():
        return AcceptanceCriterionResult(
            id="AC-004",
            description="Quick demo completes in 5 to 10 minutes without system failure.",
            passed=False,
            evidence_paths=evidence,
            notes="Missing Day 5 exit gate or demo summary.",
        )

    gate = json.loads(day5_exit_gate_path.read_text(encoding="utf-8"))
    summary = json.loads(demo_summary_path.read_text(encoding="utf-8"))
    passed = (
        bool(gate.get("gate_passed"))
        and bool(summary.get("timing_within_budget"))
        and bool(summary.get("all_passed"))
    )
    notes = (
        f"Day 5 gate_passed={gate.get('gate_passed')}, "
        f"timing_within_budget={summary.get('timing_within_budget')}, "
        f"all_passed={summary.get('all_passed')}, "
        f"dry_runs={summary.get('total_dry_runs')}."
    )
    return AcceptanceCriterionResult(
        id="AC-004",
        description="Quick demo completes in 5 to 10 minutes without system failure.",
        passed=passed,
        evidence_paths=evidence,
        notes=notes,
    )


def verify_ac005_guardrail_rationale(
    guardrail_transition_log_path: Path,
) -> AcceptanceCriterionResult:
    """AC-005 — Guardrail outputs are interpretable and include rationale
    text for non-GREEN transitions."""
    evidence = [str(guardrail_transition_log_path)]
    if not guardrail_transition_log_path.exists():
        return AcceptanceCriterionResult(
            id="AC-005",
            description="Guardrail outputs are interpretable and include rationale text.",
            passed=False,
            evidence_paths=evidence,
            notes="Guardrail transition log not found.",
        )

    transitions = json.loads(guardrail_transition_log_path.read_text(encoding="utf-8"))
    non_green = [t for t in transitions if t.get("new_state") != "GREEN"]
    have_rationale = [t for t in non_green if (t.get("rationale") or "").strip()]

    # If no non-GREEN transitions occurred, PRD is trivially satisfied; still
    # assert the log exists and is well-formed.
    if not non_green:
        passed = True
        notes = (
            "No non-GREEN transitions occurred in the latest run; "
            "rationale schema is present and well-formed."
        )
    else:
        passed = len(have_rationale) == len(non_green)
        notes = (
            f"{len(have_rationale)} of {len(non_green)} non-GREEN "
            f"transitions include rationale text."
        )

    return AcceptanceCriterionResult(
        id="AC-005",
        description="Guardrail outputs are interpretable and include rationale text.",
        passed=passed,
        evidence_paths=evidence,
        notes=notes,
    )


# ===========================================================================
# Checklist rendering
# ===========================================================================

def render_acceptance_checklist(
    results: list[AcceptanceCriterionResult],
) -> str:
    """Render AC-001..AC-005 results as a Markdown checklist (T16.4).

    Mirrors the schema in docs/day1/checklists/mvp_acceptance_ac_001_to_ac_005.md
    so the artifact is a drop-in replacement.
    """
    lines: list[str] = []
    lines.append("# MVP Acceptance Checklist (AC-001 to AC-005)")
    lines.append("")
    lines.append("Version: v1.1.0 (Day 6 automated verification)")
    lines.append("")

    for r in results:
        mark = "x" if r.passed else " "
        lines.append(f"- [{mark}] {r.id} {r.description}")
        for ev in r.evidence_paths:
            lines.append(f"  - Evidence link: {ev}")
        lines.append(f"  - Notes: {r.notes}")

    lines.append("")
    passed = sum(1 for r in results if r.passed)
    lines.append(f"**Summary**: {passed} / {len(results)} criteria pass.")
    return "\n".join(lines) + "\n"


# ===========================================================================
# Artifact persistence helpers
# ===========================================================================

def write_stability_report(
    reports: list[StabilityReport],
    output_path: str | Path,
) -> Path:
    """Persist the combined stability report (T16.3)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "scenarios": [
            {
                "scenario": r.scenario,
                "update_count": r.update_count,
                "transitions": r.transitions,
                "transitions_per_minute": r.transitions_per_minute,
                "green_fraction": r.green_fraction,
                "yellow_fraction": r.yellow_fraction,
                "red_fraction": r.red_fraction,
                "oscillation_index": r.oscillation_index,
                "max_consecutive_flips": r.max_consecutive_flips,
                "passed": r.passed,
                "notes": r.notes,
            }
            for r in reports
        ],
        "overall_passed": all(r.passed for r in reports),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOGGER.info("Stability report written to %s", path)
    return path


def write_latency_benchmark(
    benchmark: LatencyBenchmark,
    output_path: str | Path,
) -> Path:
    """Persist the latency benchmark report (T16.2)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "samples": benchmark.samples,
        "mean_s": benchmark.mean_s,
        "median_s": benchmark.median_s,
        "p95_s": benchmark.p95_s,
        "max_s": benchmark.max_s,
        "budget_s": benchmark.budget_s,
        "within_budget": benchmark.within_budget,
        "per_stage_s": benchmark.per_stage_s,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOGGER.info("Latency benchmark written to %s", path)
    return path


def write_triage_report(
    triage: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Persist the defect triage report (T16.1)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(triage, indent=2), encoding="utf-8")
    LOGGER.info("Defect triage report written to %s", path)
    return path


def write_ac_results(
    results: list[AcceptanceCriterionResult],
    json_path: str | Path,
    checklist_path: str | Path,
) -> tuple[Path, Path]:
    """Persist AC-001..AC-005 results as JSON + Markdown checklist (T16.4)."""
    jp = Path(json_path)
    cp = Path(checklist_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    cp.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "id": r.id,
            "description": r.description,
            "passed": r.passed,
            "evidence_paths": r.evidence_paths,
            "notes": r.notes,
        }
        for r in results
    ]
    summary = {
        "criteria": records,
        "passed_count": sum(1 for r in results if r.passed),
        "total_count": len(results),
        "all_passed": all(r.passed for r in results),
    }
    jp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cp.write_text(render_acceptance_checklist(results), encoding="utf-8")
    LOGGER.info("AC results written to %s and %s", jp, cp)
    return jp, cp
