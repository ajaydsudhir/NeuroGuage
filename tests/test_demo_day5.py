"""Tests for Day 5 — T15 Class Quick Demo Orchestration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from neuroguage.lifecycle import (
    DemoConfig,
    DemoOrchestrator,
    DemoStage,
    run_dry_demo,
    validate_demo_timing,
    write_demo_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_config() -> DemoConfig:
    return DemoConfig.default()


@pytest.fixture
def inference_diagnostics() -> list[dict]:
    return [
        {"timestamp_s": float(i), "risk_score": 0.3 + i * 0.02,
         "model_confidence": 0.9, "latency_s": 0.01, "fault_flags": []}
        for i in range(20)
    ]


@pytest.fixture
def guardrail_transitions() -> list[dict]:
    return [
        {"timestamp_s": 10.0, "previous_state": "GREEN", "new_state": "YELLOW",
         "risk_score": 0.45, "rationale": "Risk exceeded YELLOW threshold."},
    ]


# ---------------------------------------------------------------------------
# Tests — O-05  Demo envelope and timing budget
# ---------------------------------------------------------------------------


class TestDemoTiming:
    def test_default_config_has_five_stages(self, demo_config: DemoConfig) -> None:
        assert len(demo_config.stages) == 5

    def test_default_config_within_budget(self, demo_config: DemoConfig) -> None:
        report = validate_demo_timing(demo_config)
        assert report["within_budget"] is True

    def test_total_duration_matches_prd(self, demo_config: DemoConfig) -> None:
        total = sum(s.duration_s for s in demo_config.stages)
        assert 300 <= total <= 600  # 5–10 minutes

    def test_sequence_matches_prd(self, demo_config: DemoConfig) -> None:
        names = [s.name for s in demo_config.stages]
        expected = ["baseline", "sustained_attention", "n_back", "code_comprehension", "summary"]
        assert names == expected

    def test_violation_detected_for_overlong(self) -> None:
        config = DemoConfig(stages=[
            DemoStage("baseline", "Baseline", 700, "baseline_rest"),
        ], total_max_s=600)
        report = validate_demo_timing(config)
        assert report["within_budget"] is False
        assert len(report["violations"]) > 0

    def test_from_json_loads_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timing.json"
            path.write_text(json.dumps({
                "quick_demo": {
                    "stages": [
                        {"name": "baseline", "label": "Baseline", "duration_s": 60,
                         "task_type": "baseline_rest"},
                    ]
                },
                "timing_constraints": {"min_total_s": 50, "max_total_s": 100},
            }))
            config = DemoConfig.from_json(path)
            assert len(config.stages) == 1
            assert config.total_max_s == 100


# ---------------------------------------------------------------------------
# Tests — T15.1  Guided demo flow with stage timers
# ---------------------------------------------------------------------------


class TestDemoOrchestrator:
    def test_start_creates_state(self, demo_config: DemoConfig) -> None:
        orch = DemoOrchestrator(demo_config)
        state = orch.start("test_run")
        assert state.is_running is True
        assert state.run_id == "test_run"

    def test_advance_through_all_stages(self, demo_config: DemoConfig) -> None:
        orch = DemoOrchestrator(demo_config)
        orch.start("test_run")

        stage_names = []
        for _ in range(len(demo_config.stages)):
            stage = orch.advance_stage()
            assert stage is not None
            stage_names.append(stage.name)

        # One more advance should complete the demo
        result = orch.advance_stage()
        assert result is None
        assert orch.state.is_completed is True

        expected = ["baseline", "sustained_attention", "n_back", "code_comprehension", "summary"]
        assert stage_names == expected

    def test_current_stage_tracks_correctly(self, demo_config: DemoConfig) -> None:
        orch = DemoOrchestrator(demo_config)
        orch.start()
        assert orch.current_stage() is None  # Before first advance

        orch.advance_stage()
        assert orch.current_stage() is not None
        assert orch.current_stage().name == "baseline"

    def test_elapsed_time_increases(self, demo_config: DemoConfig) -> None:
        orch = DemoOrchestrator(demo_config)
        orch.start()
        assert orch.get_elapsed_s() >= 0.0


# ---------------------------------------------------------------------------
# Tests — T15.2  Validate sequence
# ---------------------------------------------------------------------------


class TestSequenceValidation:
    def test_correct_sequence_passes(self, demo_config: DemoConfig) -> None:
        report = validate_demo_timing(demo_config)
        assert report["within_budget"] is True
        # Sequence order is validated within the function
        assert len(report["violations"]) == 0

    def test_wrong_sequence_detected(self) -> None:
        config = DemoConfig(stages=[
            DemoStage("n_back", "2-Back", 120, "n_back"),
            DemoStage("baseline", "Baseline", 60, "baseline_rest"),
            DemoStage("sustained_attention", "Sustained", 120, "sustained_attention"),
            DemoStage("code_comprehension", "Code", 150, "code_comprehension"),
            DemoStage("summary", "Summary", 45, "summary"),
        ])
        report = validate_demo_timing(config)
        assert any("sequence" in v.lower() for v in report["violations"])

    def test_per_stage_tolerance(self, demo_config: DemoConfig) -> None:
        report = validate_demo_timing(demo_config)
        for stage in report["stages"]:
            assert stage["min_acceptable_s"] < stage["planned_s"]
            assert stage["max_acceptable_s"] > stage["planned_s"]


# ---------------------------------------------------------------------------
# Tests — T15.3  Summary panel
# ---------------------------------------------------------------------------


class TestSummaryPanel:
    def test_summary_contains_required_fields(
        self, demo_config: DemoConfig, inference_diagnostics: list[dict],
        guardrail_transitions: list[dict],
    ) -> None:
        orch = DemoOrchestrator(demo_config)
        orch.start("summary_test")

        # Run through all stages
        for _ in range(len(demo_config.stages)):
            orch.advance_stage()
        orch.advance_stage()

        summary = orch.build_summary(inference_diagnostics, guardrail_transitions)
        assert "run_id" in summary
        assert "total_elapsed_s" in summary
        assert "stages" in summary
        assert "risk_summary" in summary
        assert "transitions" in summary
        assert len(summary["stages"]) == 5

    def test_risk_summary_has_stats(
        self, demo_config: DemoConfig, inference_diagnostics: list[dict],
    ) -> None:
        orch = DemoOrchestrator(demo_config)
        orch.start()
        for _ in range(len(demo_config.stages)):
            orch.advance_stage()
        orch.advance_stage()

        summary = orch.build_summary(inference_diagnostics)
        rs = summary["risk_summary"]
        assert "risk_mean" in rs
        assert "risk_min" in rs
        assert "risk_max" in rs


# ---------------------------------------------------------------------------
# Tests — T15.4  Dry runs and failure logging
# ---------------------------------------------------------------------------


class TestDryRuns:
    def test_dry_run_completes(
        self, demo_config: DemoConfig, inference_diagnostics: list[dict],
        guardrail_transitions: list[dict],
    ) -> None:
        report = run_dry_demo(
            config=demo_config,
            inference_diagnostics=inference_diagnostics,
            guardrail_transitions=guardrail_transitions,
            run_id="test_dry_1",
        )
        assert report["dry_run"] is True
        assert "timing_validation" in report
        assert "stages" in report

    def test_dry_run_detects_latency_violation(self) -> None:
        diags = [{"latency_s": 3.0, "risk_score": 0.5, "model_confidence": 0.9, "fault_flags": []}]
        report = run_dry_demo(
            inference_diagnostics=diags,
            run_id="latency_test",
        )
        assert report["failure_count"] > 0
        assert any("latency" in f["description"].lower() for f in report["failures"])

    def test_dry_run_detects_fault_flags(self) -> None:
        diags = [{"latency_s": 0.01, "risk_score": 0.5, "model_confidence": 0.9,
                  "fault_flags": ["eeg_stale"]}]
        report = run_dry_demo(inference_diagnostics=diags, run_id="fault_test")
        assert report["failure_count"] > 0

    def test_failure_logging(self, demo_config: DemoConfig) -> None:
        orch = DemoOrchestrator(demo_config)
        orch.start("failure_test")
        orch.advance_stage()
        orch.log_failure("Test failure", severity="high")
        assert len(orch.state.failures) == 1
        assert orch.state.failures[0]["severity"] == "high"

    def test_write_demo_artifacts(
        self, demo_config: DemoConfig, inference_diagnostics: list[dict],
    ) -> None:
        reports = [
            run_dry_demo(demo_config, inference_diagnostics, run_id=f"test_run_{i}")
            for i in range(2)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_demo_artifacts(reports, tmp)
            assert "defect_log" in paths
            assert "summary" in paths
            assert paths["defect_log"].exists()
            assert paths["summary"].exists()

            summary = json.loads(paths["summary"].read_text())
            assert summary["total_dry_runs"] == 2
