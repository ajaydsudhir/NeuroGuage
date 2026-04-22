"""Day 7 — T19  Advanced vs MVP Comparison and Promotion Decision.

Encodes the PRD promotion gate FR-026 / AC-007:

  "System shall promote an advanced model only if macro F1 or AUROC improves
   by at least 3 points on held-out sessions and no real-time regression is
   observed."

All evidence required for the promotion note (T19.4) is produced by this
module and persisted as machine-readable JSON + a human-readable Markdown
narrative.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROMOTION_DELTA_THRESHOLD = 0.03  # +3 points on macro F1 or AUROC
LATENCY_TOLERANCE_S = 0.050        # 50 ms tolerance vs MVP max latency
STABILITY_PERMIN_TOLERANCE = 1.0   # allow +1 transition/min vs MVP baseline


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class MetricDelta:
    name: str
    mvp: float | None
    advanced: float | None
    delta: float | None
    meets_threshold: bool


@dataclass
class LatencyComparison:
    mvp_max_s: float
    advanced_max_s: float
    mvp_p95_s: float
    advanced_p95_s: float
    within_budget_mvp: bool
    within_budget_advanced: bool
    no_regression: bool
    notes: str


@dataclass
class StabilityComparison:
    mvp_transitions_per_minute: float
    advanced_transitions_per_minute: float
    mvp_max_consecutive_flips: int
    advanced_max_consecutive_flips: int
    no_regression: bool
    notes: str


@dataclass
class PromotionDecision:
    """Full comparison + decision payload (T19.1..T19.4)."""

    held_out_sessions: list[str]
    metric_deltas: list[MetricDelta]
    latency: LatencyComparison
    stability: StabilityComparison
    metric_gate_passed: bool
    latency_gate_passed: bool
    stability_gate_passed: bool
    promoted: bool
    decision: str
    rationale: str
    mvp_metrics: dict[str, Any] = field(default_factory=dict)
    advanced_metrics: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Comparison logic
# ===========================================================================

def _extract_mvp_test_metrics(mvp_eval: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
    """Pull the test-set metrics for the selected MVP variant."""
    if variant is None:
        variant = mvp_eval.get("selected_variant", "")
    scores = mvp_eval.get("variant_scores", {})
    entry = scores.get(variant) or {}
    return entry.get("test_late_fusion", {}) or {}


def _delta(mvp: Any, adv: Any) -> float | None:
    if mvp is None or adv is None:
        return None
    try:
        return float(adv) - float(mvp)
    except (TypeError, ValueError):
        return None


def compare_metrics(
    mvp_eval_summary: dict[str, Any],
    advanced_metrics: dict[str, Any],
    *,
    threshold: float = PROMOTION_DELTA_THRESHOLD,
) -> tuple[list[MetricDelta], bool, dict[str, Any]]:
    """Compute metric deltas between MVP and advanced on identical held-out.

    Returns
    -------
    deltas : list of per-metric MetricDelta records.
    passed : True if macro F1 OR AUROC improves by >= threshold.
    mvp_test_metrics : the MVP test-set dict used for comparison.
    """
    mvp_test = _extract_mvp_test_metrics(mvp_eval_summary)
    adv_test = advanced_metrics.get("test", {})

    deltas: list[MetricDelta] = []
    for name in ("macro_f1", "auroc", "balanced_accuracy"):
        d = _delta(mvp_test.get(name), adv_test.get(name))
        promotable = name in ("macro_f1", "auroc")
        meets = bool(promotable and d is not None and d >= threshold)
        deltas.append(
            MetricDelta(
                name=name,
                mvp=(float(mvp_test[name]) if mvp_test.get(name) is not None else None),
                advanced=(float(adv_test[name]) if adv_test.get(name) is not None else None),
                delta=d,
                meets_threshold=meets,
            )
        )

    passed = any(d.meets_threshold for d in deltas if d.name in ("macro_f1", "auroc"))
    return deltas, passed, mvp_test


def compare_latency(
    mvp_bench: dict[str, Any],
    advanced_bench: dict[str, Any],
    *,
    budget_s: float = 2.0,
    tolerance_s: float = LATENCY_TOLERANCE_S,
) -> LatencyComparison:
    """Verify the advanced model does not regress real-time latency."""
    mvp_max = float(mvp_bench.get("max_s", 0.0))
    adv_max = float(advanced_bench.get("max_s", 0.0))
    mvp_p95 = float(mvp_bench.get("p95_s", 0.0))
    adv_p95 = float(advanced_bench.get("p95_s", 0.0))

    within_budget_mvp = adv_max <= float(mvp_bench.get("budget_s", budget_s))
    within_budget_adv = adv_max <= float(advanced_bench.get("budget_s", budget_s))

    no_regression = (adv_max <= mvp_max + tolerance_s) and within_budget_adv

    notes = (
        f"advanced_max={adv_max*1000:.3f}ms vs mvp_max={mvp_max*1000:.3f}ms "
        f"(tolerance={tolerance_s*1000:.1f}ms); "
        f"advanced_p95={adv_p95*1000:.3f}ms vs mvp_p95={mvp_p95*1000:.3f}ms."
    )
    return LatencyComparison(
        mvp_max_s=mvp_max,
        advanced_max_s=adv_max,
        mvp_p95_s=mvp_p95,
        advanced_p95_s=adv_p95,
        within_budget_mvp=within_budget_mvp,
        within_budget_advanced=within_budget_adv,
        no_regression=no_regression,
        notes=notes,
    )


def compare_stability(
    mvp_stability: list[dict[str, Any]],
    advanced_stability: list[dict[str, Any]],
    *,
    per_min_tolerance: float = STABILITY_PERMIN_TOLERANCE,
) -> StabilityComparison:
    """Verify the advanced model does not regress alert stability."""

    def _worst(records: list[dict[str, Any]]) -> tuple[float, int]:
        if not records:
            return 0.0, 0
        per_min = max(float(r.get("transitions_per_minute", 0.0)) for r in records)
        flips = max(int(r.get("max_consecutive_flips", 0)) for r in records)
        return per_min, flips

    mvp_per_min, mvp_flips = _worst(mvp_stability)
    adv_per_min, adv_flips = _worst(advanced_stability)

    no_regression = (
        adv_per_min <= mvp_per_min + per_min_tolerance and adv_flips <= mvp_flips + 1
    )
    notes = (
        f"worst transitions/min: mvp={mvp_per_min:.3f}, advanced={adv_per_min:.3f} "
        f"(tolerance={per_min_tolerance:.2f}); "
        f"worst max_consecutive_flips: mvp={mvp_flips}, advanced={adv_flips}."
    )
    return StabilityComparison(
        mvp_transitions_per_minute=mvp_per_min,
        advanced_transitions_per_minute=adv_per_min,
        mvp_max_consecutive_flips=mvp_flips,
        advanced_max_consecutive_flips=adv_flips,
        no_regression=no_regression,
        notes=notes,
    )


def decide_promotion(
    mvp_eval_summary: dict[str, Any],
    advanced_metrics: dict[str, Any],
    mvp_latency: dict[str, Any],
    advanced_latency: dict[str, Any],
    mvp_stability: list[dict[str, Any]],
    advanced_stability: list[dict[str, Any]],
    *,
    threshold: float = PROMOTION_DELTA_THRESHOLD,
    held_out_sessions: list[str] | None = None,
) -> PromotionDecision:
    """Apply FR-026 / AC-007 / AC-008 promotion rules."""
    deltas, metric_ok, _mvp_test = compare_metrics(
        mvp_eval_summary, advanced_metrics, threshold=threshold
    )
    latency_cmp = compare_latency(mvp_latency, advanced_latency)
    stability_cmp = compare_stability(mvp_stability, advanced_stability)

    promoted = metric_ok and latency_cmp.no_regression and stability_cmp.no_regression

    if promoted:
        decision = "PROMOTE_ADVANCED"
        rationale = (
            "Advanced BiGRU-attention met the +{thr:.0f}-point gate on macro F1 or "
            "AUROC AND showed no latency or alert-stability regression vs the MVP. "
            "Promotion is approved per FR-026, AC-007, AC-008."
        ).format(thr=threshold * 100)
    else:
        reasons: list[str] = []
        if not metric_ok:
            reasons.append(
                f"metric gate FAILED: no +{threshold*100:.0f}-point lift in macro F1 or AUROC"
            )
        if not latency_cmp.no_regression:
            reasons.append("latency regression vs MVP")
        if not stability_cmp.no_regression:
            reasons.append("alert-stability regression vs MVP")
        decision = "KEEP_MVP"
        rationale = (
            "Keep MVP as production demo model. Gate outcome: "
            + "; ".join(reasons)
            + ". Advanced prototype artifacts are retained for future iteration."
        )

    return PromotionDecision(
        held_out_sessions=held_out_sessions or [],
        metric_deltas=deltas,
        latency=latency_cmp,
        stability=stability_cmp,
        metric_gate_passed=metric_ok,
        latency_gate_passed=latency_cmp.no_regression,
        stability_gate_passed=stability_cmp.no_regression,
        promoted=promoted,
        decision=decision,
        rationale=rationale,
        mvp_metrics=_extract_mvp_test_metrics(mvp_eval_summary),
        advanced_metrics=advanced_metrics.get("test", {}),
    )


# ===========================================================================
# Persistence + narrative
# ===========================================================================

def _decision_to_dict(decision: PromotionDecision) -> dict[str, Any]:
    return {
        "decision": decision.decision,
        "promoted": decision.promoted,
        "rationale": decision.rationale,
        "held_out_sessions": decision.held_out_sessions,
        "metric_gate_passed": decision.metric_gate_passed,
        "latency_gate_passed": decision.latency_gate_passed,
        "stability_gate_passed": decision.stability_gate_passed,
        "threshold_delta": PROMOTION_DELTA_THRESHOLD,
        "metric_deltas": [asdict(d) for d in decision.metric_deltas],
        "latency": asdict(decision.latency),
        "stability": asdict(decision.stability),
        "mvp_metrics": decision.mvp_metrics,
        "advanced_metrics": decision.advanced_metrics,
    }


def write_promotion_decision(
    decision: PromotionDecision,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Persist T19.4 decision note as JSON + Markdown narrative."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_decision_to_dict(decision), indent=2), encoding="utf-8")

    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_promotion_markdown(decision), encoding="utf-8")


def render_promotion_markdown(decision: PromotionDecision) -> str:
    """Render the T19.4 promotion decision as a human-readable Markdown note."""
    lines: list[str] = []
    lines.append("# Advanced vs MVP Promotion Decision (T19.4)")
    lines.append("")
    lines.append(f"- **Decision**: `{decision.decision}`")
    lines.append(f"- **Promoted**: `{decision.promoted}`")
    lines.append(f"- **Metric gate**: {'PASS' if decision.metric_gate_passed else 'FAIL'}")
    lines.append(f"- **Latency gate**: {'PASS' if decision.latency_gate_passed else 'FAIL'}")
    lines.append(f"- **Stability gate**: {'PASS' if decision.stability_gate_passed else 'FAIL'}")
    lines.append(f"- **Threshold**: +{PROMOTION_DELTA_THRESHOLD*100:.0f} points on macro F1 or AUROC (FR-026)")
    if decision.held_out_sessions:
        lines.append(f"- **Held-out sessions**: {', '.join(decision.held_out_sessions)}")
    lines.append("")

    lines.append("## Rationale")
    lines.append("")
    lines.append(decision.rationale)
    lines.append("")

    lines.append("## Held-out Metric Deltas (identical sessions)")
    lines.append("")
    lines.append("| Metric | MVP | Advanced | Delta | Meets +3pt |")
    lines.append("|--------|-----|----------|-------|------------|")
    for d in decision.metric_deltas:
        mvp_v = f"{d.mvp:.4f}" if d.mvp is not None else "—"
        adv_v = f"{d.advanced:.4f}" if d.advanced is not None else "—"
        delta_v = f"{d.delta:+.4f}" if d.delta is not None else "—"
        meets = "YES" if d.meets_threshold else "no"
        lines.append(f"| {d.name} | {mvp_v} | {adv_v} | {delta_v} | {meets} |")
    lines.append("")

    lines.append("## Latency Comparison (NFR-001)")
    lines.append("")
    lines.append("| Stat | MVP | Advanced |")
    lines.append("|------|-----|----------|")
    lines.append(
        f"| max (ms)  | {decision.latency.mvp_max_s*1000:.3f} | {decision.latency.advanced_max_s*1000:.3f} |"
    )
    lines.append(
        f"| p95 (ms)  | {decision.latency.mvp_p95_s*1000:.3f} | {decision.latency.advanced_p95_s*1000:.3f} |"
    )
    lines.append(
        f"| within budget | {decision.latency.within_budget_mvp} | {decision.latency.within_budget_advanced} |"
    )
    lines.append(f"\n> {decision.latency.notes}")
    lines.append("")

    lines.append("## Alert Stability Comparison (NFR-002)")
    lines.append("")
    lines.append("| Stat | MVP | Advanced |")
    lines.append("|------|-----|----------|")
    lines.append(
        f"| worst transitions/min | {decision.stability.mvp_transitions_per_minute:.3f} | "
        f"{decision.stability.advanced_transitions_per_minute:.3f} |"
    )
    lines.append(
        f"| worst max_consecutive_flips | {decision.stability.mvp_max_consecutive_flips} | "
        f"{decision.stability.advanced_max_consecutive_flips} |"
    )
    lines.append(f"\n> {decision.stability.notes}")
    lines.append("")

    return "\n".join(lines) + "\n"
