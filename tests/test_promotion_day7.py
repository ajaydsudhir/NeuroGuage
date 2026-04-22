"""Tests for Day 7 — T19 Advanced vs MVP promotion decision."""

from __future__ import annotations

import json
from pathlib import Path

from neuroguage.promotion import (
    PROMOTION_DELTA_THRESHOLD,
    compare_latency,
    compare_metrics,
    compare_stability,
    decide_promotion,
    render_promotion_markdown,
    write_promotion_decision,
)


MVP_EVAL = {
    "selected_variant": "eeg_optical",
    "variant_scores": {
        "eeg_optical": {
            "test_late_fusion": {"macro_f1": 0.40, "balanced_accuracy": 0.50, "auroc": 1.00},
        }
    },
}

MVP_LATENCY = {"max_s": 0.004, "p95_s": 0.002, "budget_s": 2.0, "within_budget": True}
MVP_STABILITY = [
    {"scenario": "noisy_risk", "transitions_per_minute": 1.0, "max_consecutive_flips": 1, "passed": True},
    {"scenario": "motion_artifact", "transitions_per_minute": 0.5, "max_consecutive_flips": 1, "passed": True},
]


def test_compare_metrics_detects_promotion_lift() -> None:
    adv = {"test": {"macro_f1": 0.46, "balanced_accuracy": 0.62, "auroc": 1.0}}
    deltas, passed, _ = compare_metrics(MVP_EVAL, adv)
    macro = next(d for d in deltas if d.name == "macro_f1")
    assert macro.delta is not None and macro.delta >= PROMOTION_DELTA_THRESHOLD
    assert passed is True


def test_compare_metrics_fails_on_tiny_lift() -> None:
    adv = {"test": {"macro_f1": 0.41, "balanced_accuracy": 0.52, "auroc": 1.0}}
    _, passed, _ = compare_metrics(MVP_EVAL, adv)
    assert passed is False


def test_compare_latency_flags_regression() -> None:
    adv_bench = {"max_s": 0.200, "p95_s": 0.150, "budget_s": 2.0}
    cmp = compare_latency(MVP_LATENCY, adv_bench)
    assert cmp.within_budget_advanced is True
    assert cmp.no_regression is False
    assert "advanced_max" in cmp.notes


def test_compare_latency_passes_when_close() -> None:
    adv_bench = {"max_s": 0.020, "p95_s": 0.010, "budget_s": 2.0}
    cmp = compare_latency(MVP_LATENCY, adv_bench)
    assert cmp.no_regression is True


def test_compare_stability_passes_and_fails() -> None:
    adv_ok = [
        {"scenario": "noisy", "transitions_per_minute": 1.2, "max_consecutive_flips": 2, "passed": True},
    ]
    cmp = compare_stability(MVP_STABILITY, adv_ok)
    assert cmp.no_regression is True

    adv_bad = [
        {"scenario": "noisy", "transitions_per_minute": 5.0, "max_consecutive_flips": 5, "passed": False},
    ]
    cmp_bad = compare_stability(MVP_STABILITY, adv_bad)
    assert cmp_bad.no_regression is False


def test_decide_promotion_all_green_promotes() -> None:
    adv_metrics = {"test": {"macro_f1": 0.50, "balanced_accuracy": 0.62, "auroc": 1.0}}
    adv_latency = {"max_s": 0.010, "p95_s": 0.008, "budget_s": 2.0}
    adv_stability = [
        {"scenario": "noisy_risk", "transitions_per_minute": 1.1, "max_consecutive_flips": 1, "passed": True},
    ]
    decision = decide_promotion(
        mvp_eval_summary=MVP_EVAL,
        advanced_metrics=adv_metrics,
        mvp_latency=MVP_LATENCY,
        advanced_latency=adv_latency,
        mvp_stability=MVP_STABILITY,
        advanced_stability=adv_stability,
        held_out_sessions=["s01_test"],
    )
    assert decision.promoted is True
    assert decision.decision == "PROMOTE_ADVANCED"
    assert decision.held_out_sessions == ["s01_test"]


def test_decide_promotion_keeps_mvp_on_metric_fail() -> None:
    adv_metrics = {"test": {"macro_f1": 0.41, "balanced_accuracy": 0.52, "auroc": 1.0}}
    adv_latency = {"max_s": 0.010, "p95_s": 0.008, "budget_s": 2.0}
    adv_stability = MVP_STABILITY
    decision = decide_promotion(
        mvp_eval_summary=MVP_EVAL,
        advanced_metrics=adv_metrics,
        mvp_latency=MVP_LATENCY,
        advanced_latency=adv_latency,
        mvp_stability=MVP_STABILITY,
        advanced_stability=adv_stability,
    )
    assert decision.promoted is False
    assert decision.decision == "KEEP_MVP"
    assert "metric gate" in decision.rationale.lower()


def test_render_and_write_promotion_decision(tmp_path: Path) -> None:
    adv_metrics = {"test": {"macro_f1": 0.50, "balanced_accuracy": 0.62, "auroc": 1.0}}
    adv_latency = {"max_s": 0.010, "p95_s": 0.008, "budget_s": 2.0}
    decision = decide_promotion(
        mvp_eval_summary=MVP_EVAL,
        advanced_metrics=adv_metrics,
        mvp_latency=MVP_LATENCY,
        advanced_latency=adv_latency,
        mvp_stability=MVP_STABILITY,
        advanced_stability=MVP_STABILITY,
    )
    md = render_promotion_markdown(decision)
    assert "Promotion Decision" in md
    json_path = tmp_path / "promo.json"
    md_path = tmp_path / "promo.md"
    write_promotion_decision(decision, json_path, md_path)
    data = json.loads(json_path.read_text())
    assert data["promoted"] is True
    assert md_path.read_text().startswith("# Advanced vs MVP Promotion Decision")
