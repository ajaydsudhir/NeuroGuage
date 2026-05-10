"""Tests for T16 MVP hardening module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuroguage.guardrails import GuardrailConfig, GuardrailState, update_guardrail
from neuroguage.lifecycle import (
    AcceptanceCriterionResult,
    benchmark_refresh_latency,
    prioritise_defects,
    render_acceptance_checklist,
    run_all_stability_scenarios,
    stability_scenario_motion_artifact,
    stability_scenario_noisy_risk,
    stability_scenario_sustained_fatigue,
    verify_ac001_end_to_end,
    verify_ac002_session_held_out,
    verify_ac003_latency,
    verify_ac004_quick_demo,
    verify_ac005_guardrail_rationale,
    write_ac_results,
    write_latency_benchmark,
    write_stability_report,
    write_triage_report,
)


# ---------------------------------------------------------------------------
# T16.1  Defect triage
# ---------------------------------------------------------------------------


class TestDefectTriage:
    def test_empty_reports_all_clear(self) -> None:
        triage = prioritise_defects([])
        assert triage["all_clear"] is True
        assert triage["total_defects"] == 0

    def test_groups_by_severity(self) -> None:
        reports = [
            {
                "run_id": "r1",
                "failures": [
                    {"severity": "high", "description": "H1", "stage": "baseline", "elapsed_s": 1.0},
                    {"severity": "medium", "description": "M1", "stage": "n_back", "elapsed_s": 2.0},
                    {"severity": "low", "description": "L1", "stage": "summary", "elapsed_s": 3.0},
                ],
            },
            {"run_id": "r2", "failures": [{"severity": "high", "description": "H2"}]},
        ]
        triage = prioritise_defects(reports)
        assert triage["total_defects"] == 4
        assert len(triage["high_priority"]) == 2
        assert len(triage["medium_priority"]) == 1
        assert len(triage["low_priority"]) == 1
        # Resolution order must start with high priority items
        assert triage["resolution_order"][0] in {"H1", "H2"}
        assert triage["resolution_order"][1] in {"H1", "H2"}

    def test_handles_missing_failures_key(self) -> None:
        triage = prioritise_defects([{"run_id": "x"}])
        assert triage["all_clear"] is True


# ---------------------------------------------------------------------------
# T16.2  Latency benchmark
# ---------------------------------------------------------------------------


class TestLatencyBenchmark:
    def test_benchmark_reports_within_budget(self) -> None:
        def infer(i: int) -> tuple[float, float]:
            return 0.3, 0.9

        def guard(r: float, c: float, ts: float) -> None:
            pass

        bench = benchmark_refresh_latency(infer, guard, sample_count=20, budget_s=2.0)
        assert bench.samples == 20
        assert bench.max_s < 1.0  # trivial fn must be well under 1s
        assert bench.within_budget is True
        assert "inference_mean" in bench.per_stage_s

    def test_budget_violation_detected(self) -> None:
        import time

        def slow_infer(i: int) -> tuple[float, float]:
            time.sleep(0.01)
            return 0.3, 0.9

        def guard(r: float, c: float, ts: float) -> None:
            pass

        bench = benchmark_refresh_latency(slow_infer, guard, sample_count=5, budget_s=0.001)
        assert bench.within_budget is False


# ---------------------------------------------------------------------------
# T16.3  Stability scenarios
# ---------------------------------------------------------------------------


class TestStability:
    def test_noisy_risk_suppressed_by_persistence(self) -> None:
        report = stability_scenario_noisy_risk()
        # Persistence logic must prevent rapid-flip bursts
        assert report.max_consecutive_flips <= 2
        assert report.passed is True

    def test_motion_artifact_blocks_red_escalation(self) -> None:
        report = stability_scenario_motion_artifact()
        assert report.red_fraction == 0.0
        assert report.passed is True

    def test_sustained_fatigue_still_escalates(self) -> None:
        report = stability_scenario_sustained_fatigue()
        # The ramp ends at 0.9 — engine must reach RED or BLACK territory
        assert report.red_fraction + (1.0 - report.green_fraction - report.yellow_fraction) > 0
        assert report.passed is True

    def test_run_all_scenarios(self) -> None:
        reports = run_all_stability_scenarios()
        assert len(reports) == 3
        assert all(r.passed for r in reports)


# ---------------------------------------------------------------------------
# T16.4  Acceptance verification
# ---------------------------------------------------------------------------


class TestAcceptanceVerification:
    def test_ac001_missing_evidence_fails(self, tmp_path: Path) -> None:
        result = verify_ac001_end_to_end(
            tmp_path / "inference.json",
            tmp_path / "guardrail.json",
            tmp_path / "snapshot.png",
        )
        assert result.passed is False
        assert result.id == "AC-001"

    def test_ac001_with_evidence_passes(self, tmp_path: Path) -> None:
        for name in ("inference.json", "guardrail.json", "snapshot.png"):
            (tmp_path / name).write_bytes(b"stub")
        result = verify_ac001_end_to_end(
            tmp_path / "inference.json",
            tmp_path / "guardrail.json",
            tmp_path / "snapshot.png",
        )
        assert result.passed is True

    def test_ac002_reads_metrics(self, tmp_path: Path) -> None:
        summary = tmp_path / "eval.json"
        ablation = tmp_path / "ablation.json"
        summary.write_text(json.dumps({
            "selected_variant": "eeg_optical",
            "variant_scores": {
                "eeg_optical": {
                    "test_late_fusion": {
                        "macro_f1": 0.4,
                        "balanced_accuracy": 0.5,
                        "auroc": 1.0,
                    }
                }
            },
        }))
        ablation.write_text("{}")
        result = verify_ac002_session_held_out(summary, ablation)
        assert result.passed is True
        assert "macro_f1" in result.notes

    def test_ac003_latency_pass(self) -> None:
        from neuroguage.lifecycle import LatencyBenchmark

        bench = LatencyBenchmark(
            samples=10, mean_s=0.01, median_s=0.01, p95_s=0.02,
            max_s=0.03, budget_s=2.0, within_budget=True,
        )
        result = verify_ac003_latency(bench, Path("nonexistent.json"))
        assert result.passed is True

    def test_ac004_quick_demo(self, tmp_path: Path) -> None:
        gate = tmp_path / "gate.json"
        summary = tmp_path / "summary.json"
        gate.write_text(json.dumps({"gate_passed": True}))
        summary.write_text(json.dumps({
            "total_dry_runs": 2,
            "timing_within_budget": True,
            "all_passed": True,
        }))
        result = verify_ac004_quick_demo(gate, summary)
        assert result.passed is True

    def test_ac005_no_non_green_transitions_is_valid(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps([]))
        result = verify_ac005_guardrail_rationale(log_path)
        assert result.passed is True

    def test_ac005_requires_rationale_on_non_green(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps([
            {"new_state": "YELLOW", "rationale": "Risk crossed threshold."},
            {"new_state": "RED", "rationale": ""},
        ]))
        result = verify_ac005_guardrail_rationale(log_path)
        assert result.passed is False

    def test_checklist_rendering(self) -> None:
        results = [
            AcceptanceCriterionResult(
                id="AC-001", description="End-to-end",
                passed=True, evidence_paths=["a.json"], notes="ok",
            ),
            AcceptanceCriterionResult(
                id="AC-002", description="Metrics",
                passed=False, evidence_paths=[], notes="missing",
            ),
        ]
        md = render_acceptance_checklist(results)
        assert "- [x] AC-001" in md
        assert "- [ ] AC-002" in md
        assert "1 / 2" in md


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


class TestArtifactPersistence:
    def test_write_triage(self, tmp_path: Path) -> None:
        path = write_triage_report({"total_defects": 0, "high_priority": [],
                                    "medium_priority": [], "low_priority": [],
                                    "resolution_order": [], "all_clear": True},
                                   tmp_path / "triage.json")
        assert path.exists()

    def test_write_latency_benchmark(self, tmp_path: Path) -> None:
        from neuroguage.lifecycle import LatencyBenchmark

        bench = LatencyBenchmark(samples=1, mean_s=0, median_s=0, p95_s=0,
                                 max_s=0, budget_s=2.0, within_budget=True)
        path = write_latency_benchmark(bench, tmp_path / "lat.json")
        data = json.loads(path.read_text())
        assert data["within_budget"] is True

    def test_write_stability_and_ac(self, tmp_path: Path) -> None:
        reports = run_all_stability_scenarios()
        write_stability_report(reports, tmp_path / "stab.json")

        results = [
            AcceptanceCriterionResult(id=f"AC-00{i+1}", description="d",
                                      passed=True, evidence_paths=["x"], notes="n")
            for i in range(5)
        ]
        jp, cp = write_ac_results(results, tmp_path / "ac.json", tmp_path / "ac.md")
        data = json.loads(jp.read_text())
        assert data["all_passed"] is True
        assert cp.read_text().startswith("# MVP Acceptance Checklist")
