"""Tests for Day 7 — T20 finalization utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuroguage.finalization import (
    RehearsalResult,
    build_final_demo_bundle,
    build_final_status_summary,
    render_final_status_markdown,
    resolve_stretch_status,
    run_final_rehearsal,
    write_final_status_summary,
    write_rehearsal_report,
)


MVP_EVAL = {
    "selected_variant": "eeg_optical",
    "variant_scores": {
        "eeg_optical": {
            "test_late_fusion": {"macro_f1": 0.4, "balanced_accuracy": 0.5, "auroc": 1.0},
        }
    },
}


def test_resolve_stretch_status_defers_when_no_go() -> None:
    status = resolve_stretch_status(
        day6_exit_gate={"phase_gate_decision": "NO-GO: Day 7 stays MVP-only"},
        promotion_decision=None,
    )
    assert status.deferred is True
    assert status.stretch_attempted is False
    assert "NO-GO" in status.reason or "Deferred" in status.reason or "deferred" in status.reason.lower()


def test_resolve_stretch_status_uses_promotion_decision() -> None:
    gate = {"phase_gate_decision": "GO for Day 7 stretch"}
    promo = {"decision": "PROMOTE_ADVANCED", "rationale": "Advanced exceeded threshold."}
    status = resolve_stretch_status(gate, promo)
    assert status.deferred is False
    assert status.decision == "PROMOTE_ADVANCED"
    assert status.stretch_attempted is True


def test_resolve_stretch_status_defers_without_promotion_when_go() -> None:
    gate = {"phase_gate_decision": "GO for Day 7 stretch"}
    status = resolve_stretch_status(gate, None)
    assert status.deferred is True
    assert status.stretch_attempted is False


def test_run_final_rehearsal_produces_within_budget_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    inference_path = tmp_path / "inference.json"
    guardrail_path = tmp_path / "guardrail.json"
    inference_path.write_text(json.dumps([{"latency_s": 0.002, "fault_flags": []} for _ in range(5)]))
    guardrail_path.write_text(json.dumps([]))

    latency_bench = {"within_budget": True, "max_s": 0.003}
    stability = [{"scenario": "noisy_risk", "passed": True}]

    result = run_final_rehearsal(
        demo_config_path=None,
        inference_diagnostics_path=inference_path,
        guardrail_log_path=guardrail_path,
        latency_benchmark=latency_bench,
        stability_reports=stability,
    )
    assert isinstance(result, RehearsalResult)
    assert result.within_budget is True
    assert result.failure_count == 0
    assert result.latency_within_budget is True
    assert result.stability_within_budget is True

    out_path = tmp_path / "rehearsal.json"
    write_rehearsal_report(result, out_path)
    payload = json.loads(out_path.read_text())
    assert payload["within_budget"] is True


def test_build_final_status_summary_fields() -> None:
    stretch = resolve_stretch_status(
        {"phase_gate_decision": "GO for Day 7 stretch"},
        {"decision": "KEEP_MVP", "rationale": "No lift.", "promoted": False},
    )
    rehearsal = RehearsalResult(
        within_budget=True,
        total_elapsed_s=420.0,
        stage_count=5,
        failure_count=0,
        latency_within_budget=True,
        stability_within_budget=True,
        timing_violations=[],
        report={},
    )
    acceptance = {
        "passed_count": 5,
        "total_count": 5,
        "criteria": [
            {"id": "AC-001", "description": "End-to-end", "passed": True},
        ],
    }
    summary = build_final_status_summary(
        rc_version="1.0.0-rc1",
        commit="abcdef",
        stretch=stretch,
        rehearsal=rehearsal,
        acceptance_results=acceptance,
        promotion_decision={"decision": "KEEP_MVP", "promoted": False},
        latency_benchmark={"mean_s": 0.002, "p95_s": 0.003, "max_s": 0.004, "within_budget": True, "budget_s": 2.0},
        stability_reports=[{"scenario": "noisy_risk", "passed": True, "transitions_per_minute": 1.1, "max_consecutive_flips": 1}],
        mvp_evaluation=MVP_EVAL,
        release_manifest_path=Path("/tmp/manifest.json"),
        release_dir=Path("/tmp/release"),
    )
    assert summary["rc_version"] == "1.0.0-rc1"
    assert summary["acceptance_ac_001_to_ac_005"]["all_passed"] is True
    assert summary["rehearsal"]["within_budget"] is True
    assert summary["stretch"]["deferred"] is False
    assert summary["stretch"]["promoted"] is False

    md = render_final_status_markdown(summary)
    assert "Final Status Summary" in md


def test_write_final_status_summary(tmp_path: Path) -> None:
    summary = {
        "rc_version": "1.0.0-rc1",
        "commit": "abc",
        "created_at": "2026-04-22T00:00:00Z",
        "release_dir": str(tmp_path),
        "release_manifest": str(tmp_path / "manifest.json"),
        "mvp": {"selected_variant": "eeg_optical", "test_metrics": {}},
        "acceptance_ac_001_to_ac_005": {"passed_count": 5, "total_count": 5, "all_passed": True, "criteria": []},
        "latency": {"mean_ms": 0.5, "p95_ms": 1.0, "max_ms": 2.0, "within_budget": True, "budget_s": 2.0},
        "stability": {"all_passed": True, "scenarios": []},
        "rehearsal": {"within_budget": True, "total_elapsed_s": 1.0, "total_elapsed_min": 0.02, "stage_count": 5, "failure_count": 0, "latency_within_budget": True, "stability_within_budget": True, "timing_violations": []},
        "stretch": {"attempted": True, "deferred": False, "decision": "KEEP_MVP", "promoted": False, "reason": "..."},
    }
    json_path = tmp_path / "final.json"
    md_path = tmp_path / "final.md"
    write_final_status_summary(summary, json_path, md_path)
    assert json.loads(json_path.read_text())["rc_version"] == "1.0.0-rc1"
    assert md_path.read_text().startswith("# NeuroGauge Final Status Summary")


def test_build_final_demo_bundle_includes_day7_stretch_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    day7_reports = tmp_path / "day7_reports"
    day7_reports.mkdir()
    advanced_model = tmp_path / "advanced_model"
    advanced_model.mkdir()
    (advanced_model / "bigru_attention.pt").write_bytes(b"weights")
    (advanced_model / "config.json").write_text("{}")

    for name in (
        "rehearsal_report.json",
        "final_status_summary.json",
        "final_status_summary.md",
        "day7_exit_gate.json",
        "advanced_training_metrics.json",
        "advanced_training_log.json",
        "advanced_predictions.csv",
        "advanced_latency_benchmark.json",
        "advanced_stability_report.json",
        "promotion_decision.json",
        "promotion_decision.md",
    ):
        (day7_reports / name).write_text("{}")

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "day7").mkdir()
    (tmp_path / "docs" / "day7" / "quick_start.md").write_text("# quick start")

    stretch = resolve_stretch_status(
        {"phase_gate_decision": "GO for Day 7 stretch"},
        {"decision": "KEEP_MVP", "rationale": "...", "promoted": False},
    )
    manifest = build_final_demo_bundle(
        release_dir=tmp_path / "rc",
        version="1.0.0-rc1",
        commit="abcdef",
        stretch=stretch,
        day7_report_dir=day7_reports,
        advanced_model_dir=advanced_model,
    )
    labels = {e["label"] for e in manifest.entries if e["included"]}
    assert "Day 7 final rehearsal report" in labels
    assert "Day 7 promotion decision (T19.4)" in labels
    assert "Day 7 advanced BiGRU-attention model artifacts" in labels
