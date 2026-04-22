"""Tests for Day 4 — T12 Real-Time Inference Service."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuroguage.inference import (
    InferenceConfig,
    InferenceError,
    InferenceState,
    InferenceUpdate,
    build_live_features,
    detect_faults,
    run_inference_cycle,
    run_replay_inference,
    write_inference_diagnostics,
    _apply_live_normalization,
    _default_eeg_features,
    _default_optical_features,
    _default_imu_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_eeg_buffer(n_rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "eeg_tp9": rng.uniform(-50, 50, n_rows),
        "eeg_af7": rng.uniform(-50, 50, n_rows),
        "eeg_af8": rng.uniform(-50, 50, n_rows),
        "eeg_tp10": rng.uniform(-50, 50, n_rows),
    })


def _make_optical_buffer(n_rows: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "optics_lo_nir": rng.uniform(0, 1, n_rows),
        "optics_ro_nir": rng.uniform(0, 1, n_rows),
    })


def _make_imu_buffer(n_rows: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "acc_x": rng.uniform(-1, 1, n_rows),
        "acc_y": rng.uniform(-1, 1, n_rows),
        "acc_z": rng.uniform(0.8, 1.2, n_rows),
        "gyro_x": rng.uniform(-10, 10, n_rows),
        "gyro_y": rng.uniform(-10, 10, n_rows),
        "gyro_z": rng.uniform(-10, 10, n_rows),
    })


def _make_labeled_frame(n_rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    z_cols = [
        "z_eeg_rel_theta", "z_eeg_rel_alpha", "z_eeg_rel_beta",
        "z_eeg_theta_alpha_ratio", "z_eeg_beta_alpha_ratio",
        "z_eeg_quality_coverage", "z_eeg_clipping_count", "z_eeg_artifact_flag",
        "z_eeg_quality_score", "z_optical_norm_mean", "z_optical_trend_slope",
        "z_optical_mean_change", "z_imu_motion_score", "z_optical_coverage",
        "z_imu_coverage", "z_confidence_score", "z_downweight_factor",
        "z_alignment_gap_s",
    ]
    data = {"start_time_s": np.arange(0, n_rows * 2, 2, dtype=float)}
    for col in z_cols:
        data[col] = rng.standard_normal(n_rows)
    return pd.DataFrame(data)


def _make_mock_models() -> dict:
    """Create mock model dict that mimics load_inference_models output."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    n_features = 18
    n_samples = 50

    X = rng.standard_normal((n_samples, n_features))
    y = (rng.random(n_samples) > 0.5).astype(int)

    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X, y)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    feature_columns = [
        "z_eeg_rel_theta", "z_eeg_rel_alpha", "z_eeg_rel_beta",
        "z_eeg_theta_alpha_ratio", "z_eeg_beta_alpha_ratio",
        "z_eeg_quality_coverage", "z_eeg_clipping_count", "z_eeg_artifact_flag",
        "z_eeg_quality_score", "z_optical_norm_mean", "z_optical_trend_slope",
        "z_optical_mean_change", "z_imu_motion_score", "z_optical_coverage",
        "z_imu_coverage", "z_confidence_score", "z_downweight_factor",
        "z_alignment_gap_s",
    ]

    return {
        "linear_model": lr,
        "tree_model": rf,
        "fusion_weights": {"linear": 0.6, "tree": 0.4},
        "feature_columns": feature_columns,
        "scaler_metadata": {
            "strategy": "train_only_zscore",
            "statistics": {col.removeprefix("z_"): {"mean": 0.5, "std": 0.3} for col in feature_columns},
        },
        "selected_variant": "test_variant",
    }


# ---------------------------------------------------------------------------
# Tests — T12.1  Live feature path
# ---------------------------------------------------------------------------


class TestLiveFeatures:
    def test_builds_all_feature_keys(self) -> None:
        features = build_live_features(
            eeg_buffer=_make_eeg_buffer(),
            optical_buffer=_make_optical_buffer(),
            imu_buffer=_make_imu_buffer(),
            timestamp_s=10.0,
            config=InferenceConfig(),
        )
        assert "eeg_rel_theta" in features
        assert "optical_norm_mean" in features
        assert "imu_motion_score" in features
        assert "confidence_score" in features

    def test_empty_buffers_produce_defaults(self) -> None:
        features = build_live_features(
            eeg_buffer=pd.DataFrame(),
            optical_buffer=pd.DataFrame(),
            imu_buffer=pd.DataFrame(),
            timestamp_s=0.0,
            config=InferenceConfig(),
        )
        assert features["eeg_artifact_flag"] == 1.0
        assert features["eeg_quality_score"] == 0.0

    def test_normalization_applies_zscore(self) -> None:
        raw = {"eeg_rel_theta": 0.5, "eeg_rel_alpha": 0.3}
        metadata = {"statistics": {"eeg_rel_theta": {"mean": 0.5, "std": 0.1}, "eeg_rel_alpha": {"mean": 0.3, "std": 0.1}}}
        normalized = _apply_live_normalization(raw, ["z_eeg_rel_theta", "z_eeg_rel_alpha"], metadata)
        assert abs(normalized["z_eeg_rel_theta"]) < 1e-9  # (0.5-0.5)/0.1 = 0
        assert abs(normalized["z_eeg_rel_alpha"]) < 1e-9


# ---------------------------------------------------------------------------
# Tests — T12.2  Risk score emission
# ---------------------------------------------------------------------------


class TestInferenceCycle:
    def test_produces_risk_score_in_range(self) -> None:
        models = _make_mock_models()
        config = InferenceConfig()
        state = InferenceState()

        update = run_inference_cycle(
            eeg_buffer=_make_eeg_buffer(),
            optical_buffer=_make_optical_buffer(),
            imu_buffer=_make_imu_buffer(),
            timestamp_s=1.0,
            models=models,
            config=config,
            state=state,
        )
        assert 0.0 <= update.risk_score <= 1.0

    def test_model_confidence_in_range(self) -> None:
        models = _make_mock_models()
        update = run_inference_cycle(
            eeg_buffer=_make_eeg_buffer(),
            optical_buffer=_make_optical_buffer(),
            imu_buffer=_make_imu_buffer(),
            timestamp_s=2.0,
            models=models,
            config=InferenceConfig(),
            state=InferenceState(),
        )
        assert 0.0 <= update.model_confidence <= 1.0


# ---------------------------------------------------------------------------
# Tests — T12.3  Fault handling
# ---------------------------------------------------------------------------


class TestFaultDetection:
    def test_detects_stale_eeg(self) -> None:
        state = InferenceState(last_eeg_update_s=0.0)
        faults = detect_faults(
            state=state,
            current_time_s=10.0,
            config=InferenceConfig(stale_modality_timeout_s=5.0),
            eeg_buffer_empty=False,
            optical_buffer_empty=False,
            imu_buffer_empty=False,
        )
        assert "eeg_stale" in faults

    def test_detects_empty_buffers(self) -> None:
        state = InferenceState(last_eeg_update_s=0.0, last_optical_update_s=0.0, last_imu_update_s=0.0)
        faults = detect_faults(
            state=state,
            current_time_s=0.5,
            config=InferenceConfig(),
            eeg_buffer_empty=True,
            optical_buffer_empty=True,
            imu_buffer_empty=True,
        )
        assert "eeg_buffer_empty" in faults
        assert "optical_buffer_empty" in faults
        assert "imu_buffer_empty" in faults

    def test_tracks_consecutive_faults(self) -> None:
        state = InferenceState()
        config = InferenceConfig(max_consecutive_faults=3, stale_modality_timeout_s=100.0)

        for _ in range(3):
            detect_faults(state, 1.0, config, eeg_buffer_empty=True,
                         optical_buffer_empty=False, imu_buffer_empty=False)

        assert state.consecutive_faults >= 3


# ---------------------------------------------------------------------------
# Tests — T12.4  Latency and diagnostics
# ---------------------------------------------------------------------------


class TestReplayInference:
    def test_replay_returns_updates(self) -> None:
        models = _make_mock_models()
        frame = _make_labeled_frame()
        updates = run_replay_inference(frame, models)
        assert len(updates) == len(frame)

    def test_replay_latencies_are_fast(self) -> None:
        models = _make_mock_models()
        frame = _make_labeled_frame(10)
        updates = run_replay_inference(frame, models)
        for u in updates:
            assert u.latency_s < 2.0  # NFR-001

    def test_write_diagnostics(self) -> None:
        models = _make_mock_models()
        frame = _make_labeled_frame(5)
        updates = run_replay_inference(frame, models)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diag.json"
            write_inference_diagnostics(updates, path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data) == 5
