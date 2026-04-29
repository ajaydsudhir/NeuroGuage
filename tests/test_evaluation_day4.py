"""Tests for Day 4 — T11 MVP Evaluation and Ablation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuroguage.pipeline import (
    EvaluationError,
    apply_imu_quality_gate,
    build_ablation_feature_sets,
    build_variant_scaler_metadata,
    run_evaluation_pipeline,
    write_evaluation_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_labeled_frame(n_sessions: int = 5, rows_per_session: int = 8) -> pd.DataFrame:
    """Build a synthetic labeled feature matrix matching Day 3 output format."""
    rng = np.random.default_rng(42)
    records = []
    for s in range(n_sessions):
        session_id = f"s{s:02d}"
        for w in range(rows_per_session):
            record = {
                "session_id": session_id,
                "window_id": w,
                "start_time_s": float(w * 2),
                "end_time_s": float(w * 2 + 2),
                # EEG features
                "eeg_rel_theta": rng.uniform(0.15, 0.45),
                "eeg_rel_alpha": rng.uniform(0.15, 0.40),
                "eeg_rel_beta": rng.uniform(0.15, 0.35),
                "eeg_theta_alpha_ratio": rng.uniform(0.5, 2.5),
                "eeg_beta_alpha_ratio": rng.uniform(0.5, 2.0),
                "eeg_quality_coverage": rng.uniform(0.7, 1.0),
                "eeg_clipping_count": 0,
                "eeg_artifact_flag": int(rng.random() > 0.7),
                "eeg_quality_score": rng.uniform(0.5, 1.0),
                # Optical features
                "optical_norm_mean": rng.uniform(-0.5, 0.5),
                "optical_trend_slope": rng.uniform(-0.1, 0.2),
                "optical_mean_change": rng.uniform(-0.3, 0.5),
                "optical_coverage": rng.uniform(0.8, 1.0),
                # IMU features
                "imu_motion_score": rng.uniform(0.0, 0.6),
                "imu_coverage": rng.uniform(0.8, 1.0),
                # Quality
                "confidence_score": rng.uniform(0.5, 1.0),
                "downweight_factor": rng.uniform(0.5, 1.0),
                "alignment_gap_s": rng.uniform(0.0, 1.0),
            }
            # z-normalized columns
            for key in list(record.keys()):
                if key not in ("session_id", "window_id", "start_time_s", "end_time_s"):
                    record[f"z_{key}"] = (record[key] - 0.5) / 0.3
            # Label and split
            record["fatigue_label"] = int(rng.random() > 0.5)
            records.append(record)

    frame = pd.DataFrame(records)

    # Assign deterministic splits
    unique_sessions = sorted(frame["session_id"].unique())
    split_map = {}
    for i, sid in enumerate(unique_sessions):
        if i < 3:
            split_map[sid] = "train"
        elif i < 4:
            split_map[sid] = "val"
        else:
            split_map[sid] = "test"
    frame["split"] = frame["session_id"].map(split_map)
    return frame


@pytest.fixture
def labeled_frame() -> pd.DataFrame:
    return _make_labeled_frame()


@pytest.fixture
def scaler_metadata() -> dict:
    features = [
        "eeg_rel_theta", "eeg_rel_alpha", "eeg_rel_beta",
        "eeg_theta_alpha_ratio", "eeg_beta_alpha_ratio",
        "eeg_quality_coverage", "eeg_clipping_count", "eeg_artifact_flag",
        "eeg_quality_score", "optical_norm_mean", "optical_trend_slope",
        "optical_mean_change", "imu_motion_score", "optical_coverage",
        "imu_coverage", "confidence_score", "downweight_factor", "alignment_gap_s",
    ]
    return {
        "strategy": "train_only_zscore",
        "split_column": "split",
        "feature_columns": features,
        "statistics": {f: {"mean": 0.5, "std": 0.3} for f in features},
    }


# ---------------------------------------------------------------------------
# Tests — T11.1  Metrics on held-out sessions
# ---------------------------------------------------------------------------


class TestAblationFeatureSets:
    def test_builds_three_variants(self, labeled_frame: pd.DataFrame) -> None:
        sets = build_ablation_feature_sets(labeled_frame)
        assert "eeg_only" in sets
        assert "eeg_optical" in sets
        assert "eeg_optical_imu_gated" in sets

    def test_eeg_only_contains_eeg_columns(self, labeled_frame: pd.DataFrame) -> None:
        sets = build_ablation_feature_sets(labeled_frame)
        for col in sets["eeg_only"]:
            assert col.startswith("z_eeg_")

    def test_eeg_optical_contains_both(self, labeled_frame: pd.DataFrame) -> None:
        sets = build_ablation_feature_sets(labeled_frame)
        has_eeg = any(c.startswith("z_eeg_") for c in sets["eeg_optical"])
        has_optical = any(c.startswith("z_optical_") for c in sets["eeg_optical"])
        assert has_eeg and has_optical


class TestIMUQualityGate:
    def test_gate_applies_downweighting(self, labeled_frame: pd.DataFrame) -> None:
        feature_cols = [c for c in labeled_frame.columns if c.startswith("z_")]
        gated, count = apply_imu_quality_gate(labeled_frame, feature_cols)
        assert count >= 0
        assert len(gated) == len(labeled_frame)

    def test_gate_raises_on_missing_column(self, labeled_frame: pd.DataFrame) -> None:
        frame = labeled_frame.drop(columns=["imu_motion_score"])
        with pytest.raises(EvaluationError, match="motion score"):
            apply_imu_quality_gate(frame, ["z_eeg_rel_theta"])


# ---------------------------------------------------------------------------
# Tests — T11.2  Ablation runs
# ---------------------------------------------------------------------------


class TestEvaluationPipeline:
    def test_runs_all_ablations(self, labeled_frame: pd.DataFrame, scaler_metadata: dict) -> None:
        result = run_evaluation_pipeline(
            labeled_frame=labeled_frame,
            random_state=42,
            base_scaler_metadata=scaler_metadata,
        )
        assert len(result.ablation_runs) == 3
        assert result.selected_variant in result.ablation_runs

    def test_selects_variant_with_best_composite(self, labeled_frame: pd.DataFrame) -> None:
        result = run_evaluation_pipeline(labeled_frame=labeled_frame)
        selected = result.ablation_runs[result.selected_variant]
        for run in result.ablation_runs.values():
            assert selected.composite_score >= run.composite_score - 1e-9

    def test_produces_rationale(self, labeled_frame: pd.DataFrame) -> None:
        result = run_evaluation_pipeline(labeled_frame=labeled_frame)
        assert len(result.selection_rationale) > 0

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(EvaluationError, match="empty"):
            run_evaluation_pipeline(pd.DataFrame())


# ---------------------------------------------------------------------------
# Tests — T11.3  Model selection robustness
# ---------------------------------------------------------------------------


class TestRobustnessAndSelection:
    def test_composite_score_is_finite(self, labeled_frame: pd.DataFrame) -> None:
        result = run_evaluation_pipeline(labeled_frame=labeled_frame)
        for run in result.ablation_runs.values():
            assert np.isfinite(run.composite_score)

    def test_variant_scaler_metadata(self, scaler_metadata: dict) -> None:
        cols = ["z_eeg_rel_theta", "z_eeg_rel_alpha"]
        variant_meta = build_variant_scaler_metadata(scaler_metadata, cols)
        assert variant_meta is not None
        assert "eeg_rel_theta" in variant_meta["statistics"]


# ---------------------------------------------------------------------------
# Tests — T11.4  Artifact persistence
# ---------------------------------------------------------------------------


class TestEvaluationArtifacts:
    def test_writes_all_artifacts(self, labeled_frame: pd.DataFrame) -> None:
        result = run_evaluation_pipeline(labeled_frame=labeled_frame)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_evaluation_artifacts(
                result=result,
                model_output_root=tmp_path / "models",
                report_output_root=tmp_path / "reports",
                ablation_report_output_path=tmp_path / "reports" / "ablation.json",
                summary_output_path=tmp_path / "reports" / "summary.json",
                selection_output_path=tmp_path / "reports" / "selection.json",
                narrative_output_path=tmp_path / "reports" / "narrative.md",
            )
            assert (tmp_path / "reports" / "ablation.json").exists()
            assert (tmp_path / "reports" / "summary.json").exists()
            assert (tmp_path / "reports" / "selection.json").exists()
            assert (tmp_path / "reports" / "narrative.md").exists()

            # Check selection.json content
            selection = json.loads((tmp_path / "reports" / "selection.json").read_text())
            assert "selected_variant" in selection
            assert "selection_rationale" in selection
