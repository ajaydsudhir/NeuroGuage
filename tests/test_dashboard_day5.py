"""Tests for Day 5 — T14 Dashboard and Timeline Visualization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neuroguage.dashboard import (
    DashboardConfig,
    DashboardRenderer,
    DashboardSnapshot,
    build_snapshot_from_replay,
    write_dashboard_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_snapshot(n: int = 20) -> DashboardSnapshot:
    """Create a synthetic dashboard snapshot."""
    rng = np.random.default_rng(42)
    timestamps = list(np.arange(0, n * 2, 2, dtype=float))
    return DashboardSnapshot(
        timestamps=timestamps,
        eeg_theta=list(rng.uniform(0.2, 0.5, n)),
        eeg_alpha=list(rng.uniform(0.15, 0.4, n)),
        eeg_beta=list(rng.uniform(0.15, 0.35, n)),
        optical_trend=list(rng.uniform(-0.3, 0.5, n)),
        risk_scores=list(rng.uniform(0.1, 0.8, n)),
        alert_states=["GREEN"] * 10 + ["YELLOW"] * 5 + ["RED"] * 5,
        model_confidence=list(rng.uniform(0.5, 1.0, n)),
        transitions=[
            {
                "timestamp_s": 20.0,
                "previous_state": "GREEN",
                "new_state": "YELLOW",
                "risk_score": 0.45,
                "rationale": "Risk score (0.45) exceeded YELLOW threshold (0.40).",
            },
            {
                "timestamp_s": 30.0,
                "previous_state": "YELLOW",
                "new_state": "RED",
                "risk_score": 0.72,
                "rationale": "Risk score (0.72) exceeded RED threshold (0.65).",
            },
        ],
        eeg_health="OK",
        optical_health="OK",
        imu_health="WARN",
        model_health="OK",
        latency_s=0.015,
        current_state="RED",
        current_risk=0.72,
        current_confidence=0.85,
        elapsed_s=40.0,
        stage_name="Sustained Attention",
    )


@pytest.fixture
def snapshot() -> DashboardSnapshot:
    return _make_snapshot()


# ---------------------------------------------------------------------------
# Tests — T14.1  Real-time panels
# ---------------------------------------------------------------------------


class TestDashboardPanels:
    def test_renderer_creates_figure(self) -> None:
        renderer = DashboardRenderer()
        fig = renderer.initialise()
        assert isinstance(fig, plt.Figure)
        renderer.close()

    def test_renderer_has_all_axes(self) -> None:
        renderer = DashboardRenderer()
        renderer.initialise()
        assert "eeg" in renderer._axes
        assert "risk" in renderer._axes
        assert "optical" in renderer._axes
        assert "state" in renderer._axes
        assert "timeline" in renderer._axes
        renderer.close()

    def test_render_with_data(self, snapshot: DashboardSnapshot) -> None:
        renderer = DashboardRenderer()
        fig = renderer.render(snapshot)
        assert isinstance(fig, plt.Figure)
        renderer.close()

    def test_render_with_empty_snapshot(self) -> None:
        empty = DashboardSnapshot()
        renderer = DashboardRenderer()
        fig = renderer.render(empty)
        assert isinstance(fig, plt.Figure)
        renderer.close()


# ---------------------------------------------------------------------------
# Tests — T14.2  Timeline with transition markers
# ---------------------------------------------------------------------------


class TestTimeline:
    def test_transitions_in_snapshot(self, snapshot: DashboardSnapshot) -> None:
        assert len(snapshot.transitions) == 2
        assert snapshot.transitions[0]["new_state"] == "YELLOW"
        assert snapshot.transitions[1]["new_state"] == "RED"

    def test_timeline_renders_without_error(self, snapshot: DashboardSnapshot) -> None:
        renderer = DashboardRenderer()
        renderer.initialise()
        renderer._render_timeline_panel(snapshot)
        renderer.close()


# ---------------------------------------------------------------------------
# Tests — T14.3  Stream health indicators
# ---------------------------------------------------------------------------


class TestStreamHealth:
    def test_health_values_in_snapshot(self, snapshot: DashboardSnapshot) -> None:
        assert snapshot.eeg_health == "OK"
        assert snapshot.optical_health == "OK"
        assert snapshot.imu_health == "WARN"
        assert snapshot.model_health == "OK"

    def test_state_panel_renders(self, snapshot: DashboardSnapshot) -> None:
        renderer = DashboardRenderer()
        renderer.initialise()
        renderer._render_state_panel(snapshot)
        renderer.close()


# ---------------------------------------------------------------------------
# Tests — T14.4  Readability validation
# ---------------------------------------------------------------------------


class TestReadability:
    def test_save_snapshot_to_disk(self, snapshot: DashboardSnapshot) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            renderer = DashboardRenderer()
            path = renderer.save_snapshot(snapshot, Path(tmp) / "test.png")
            assert path.exists()
            assert path.stat().st_size > 0
            renderer.close()

    def test_meets_projection_minimum(self) -> None:
        config = DashboardConfig(figure_width=16.0, figure_height=9.0, dpi=100)
        assert config.figure_width * config.dpi >= 1280
        assert config.figure_height * config.dpi >= 720

    def test_dashboard_validation_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snap_path = Path(tmp) / "snap.png"
            val_path = Path(tmp) / "val.json"
            validation = write_dashboard_artifacts(snap_path, val_path)
            assert val_path.exists()
            assert validation["meets_projection_minimum"] is True
            assert validation["readability_pass"] is True


# ---------------------------------------------------------------------------
# Tests — Snapshot from replay
# ---------------------------------------------------------------------------


class TestSnapshotBuilder:
    def test_builds_from_replay_data(self) -> None:
        import pandas as pd

        diags = [
            {"timestamp_s": 0.0, "risk_score": 0.2, "model_confidence": 0.9, "latency_s": 0.01, "fault_flags": []},
            {"timestamp_s": 2.0, "risk_score": 0.5, "model_confidence": 0.8, "latency_s": 0.02, "fault_flags": []},
        ]
        transitions = [
            {"timestamp_s": 2.0, "previous_state": "GREEN", "new_state": "YELLOW",
             "risk_score": 0.5, "rationale": "test"},
        ]
        summary = {"final_state": "YELLOW"}
        frame = pd.DataFrame({
            "eeg_rel_theta": [0.3, 0.35],
            "eeg_rel_alpha": [0.25, 0.2],
            "eeg_rel_beta": [0.28, 0.3],
            "optical_norm_mean": [0.1, 0.15],
        })

        snap = build_snapshot_from_replay(diags, transitions, summary, frame)
        assert len(snap.timestamps) == 2
        assert snap.current_state == "YELLOW"
        assert len(snap.eeg_theta) == 2
