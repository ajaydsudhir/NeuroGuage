"""Day 4 — T12  Real-Time Inference Service.

Provides a live inference pipeline that:

  T12.1  Implements live feature path aligned to offline data contract and
         transforms (same windowing, feature extraction, normalization).
  T12.2  Loads selected MVP model and emits risk score in [0, 1].
  T12.3  Adds runtime fault handling for packet loss and delayed modality
         updates.
  T12.4  Logs per-update latency and model confidence diagnostics.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InferenceError(ValueError):
    """Raised when real-time inference cannot proceed."""


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Configuration for the real-time inference service."""

    model_dir: str | Path = "artifacts/models/day4"
    scaler_metadata_path: str | Path = "artifacts/models/day3/scaler_metadata.json"
    selection_path: str | Path = "artifacts/reports/day4/model_selection.json"

    # Windowing
    eeg_window_size_s: float = 2.0
    eeg_step_s: float = 0.5
    optical_window_size_s: float = 8.0
    optical_step_s: float = 2.0

    # Fault tolerance
    max_packet_delay_s: float = 1.0
    max_consecutive_faults: int = 5
    stale_modality_timeout_s: float = 5.0

    # Latency budget (NFR-001: ≤ 2 s end-to-end)
    latency_budget_s: float = 2.0


@dataclass
class InferenceUpdate:
    """Single inference cycle output."""

    timestamp_s: float
    risk_score: float
    model_confidence: float
    feature_vector: dict[str, float]
    latency_s: float
    fault_flags: list[str]
    diagnostics: dict[str, Any]


@dataclass
class InferenceState:
    """Mutable session state for the inference loop."""

    update_count: int = 0
    fault_count: int = 0
    consecutive_faults: int = 0
    last_eeg_update_s: float = 0.0
    last_optical_update_s: float = 0.0
    last_imu_update_s: float = 0.0
    risk_score_history: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    latency_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_selected_model_dir(config: InferenceConfig) -> Path:
    """Determine the model directory for the selected MVP variant."""
    selection_path = Path(config.selection_path)
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        variant = selection.get("selected_variant", "eeg_optical_imu_gated")
        variant_dir = Path(config.model_dir) / "ablations" / variant
        if variant_dir.exists():
            return variant_dir
        LOGGER.warning(
            "Selected variant directory %s not found; falling back to Day 3 models.",
            variant_dir,
        )

    # Fall back to Day 3 models
    fallback = Path("artifacts/models/day3")
    if fallback.exists():
        return fallback

    raise InferenceError("No model artifacts found for inference")


def load_inference_models(
    config: InferenceConfig,
) -> dict[str, Any]:
    """Load model artifacts for live inference (T12.2).

    Returns a dictionary with keys:
        linear_model, tree_model, fusion_weights, feature_columns,
        scaler_metadata, selected_variant
    """
    model_dir = _resolve_selected_model_dir(config)
    LOGGER.info("Loading inference models from %s", model_dir)

    linear_path = model_dir / "linear_model_calibrated.pkl"
    tree_path = model_dir / "tree_model_calibrated.pkl"
    fusion_path = model_dir / "late_fusion_config.json"
    columns_path = model_dir / "feature_columns.json"

    if not linear_path.exists():
        raise InferenceError(f"Linear model not found: {linear_path}")
    if not tree_path.exists():
        raise InferenceError(f"Tree model not found: {tree_path}")
    if not fusion_path.exists():
        raise InferenceError(f"Fusion config not found: {fusion_path}")
    if not columns_path.exists():
        raise InferenceError(f"Feature columns not found: {columns_path}")

    with linear_path.open("rb") as f:
        linear_model = pickle.load(f)  # noqa: S301
    with tree_path.open("rb") as f:
        tree_model = pickle.load(f)  # noqa: S301

    fusion_config = json.loads(fusion_path.read_text(encoding="utf-8"))
    feature_columns = json.loads(columns_path.read_text(encoding="utf-8"))

    scaler_metadata = None
    scaler_path = Path(config.scaler_metadata_path)
    if scaler_path.exists():
        scaler_metadata = json.loads(scaler_path.read_text(encoding="utf-8"))

    # Also check model-local scaler metadata
    local_scaler = model_dir / "scaler_metadata.json"
    if local_scaler.exists():
        scaler_metadata = json.loads(local_scaler.read_text(encoding="utf-8"))

    selected_variant = "unknown"
    selection_path = Path(config.selection_path)
    if selection_path.exists():
        sel = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_variant = sel.get("selected_variant", "unknown")

    return {
        "linear_model": linear_model,
        "tree_model": tree_model,
        "fusion_weights": fusion_config.get("weights", {"linear": 0.5, "tree": 0.5}),
        "feature_columns": feature_columns,
        "scaler_metadata": scaler_metadata,
        "selected_variant": selected_variant,
    }


# ---------------------------------------------------------------------------
# Live feature path (T12.1)
# ---------------------------------------------------------------------------

def _apply_live_normalization(
    raw_features: dict[str, float],
    feature_columns: list[str],
    scaler_metadata: dict[str, Any] | None,
) -> dict[str, float]:
    """Apply train-only z-score normalization aligned to offline pipeline."""
    normalized: dict[str, float] = {}

    if scaler_metadata is None:
        # No scaler — pass through raw features
        for col in feature_columns:
            raw_name = col.removeprefix("z_")
            normalized[col] = raw_features.get(raw_name, 0.0)
        return normalized

    statistics = scaler_metadata.get("statistics", {})
    for col in feature_columns:
        raw_name = col.removeprefix("z_")
        raw_value = raw_features.get(raw_name, 0.0)

        if raw_name in statistics:
            mean = float(statistics[raw_name]["mean"])
            std = float(statistics[raw_name]["std"])
            if std <= 1e-9:
                std = 1.0
            normalized[col] = (raw_value - mean) / std
        else:
            normalized[col] = raw_value

    return normalized


def build_live_features(
    eeg_buffer: pd.DataFrame,
    optical_buffer: pd.DataFrame,
    imu_buffer: pd.DataFrame,
    timestamp_s: float,
    config: InferenceConfig,
) -> dict[str, float]:
    """Construct a feature vector from live sensor buffers (T12.1).

    Mirrors the offline EEG and optical/IMU feature extraction but operates
    on real-time circular buffers rather than session-wide DataFrames.
    """
    features: dict[str, float] = {}

    # --- EEG features ---
    if eeg_buffer.empty:
        features.update(_default_eeg_features())
    else:
        eeg_channels = [c for c in eeg_buffer.columns if c.startswith("eeg_")]
        channel_thetas, channel_alphas, channel_betas = [], [], []
        coverage_list = []

        for ch in eeg_channels:
            vals = eeg_buffer[ch].dropna().to_numpy(dtype=float)
            coverage_list.append(float(len(vals)) / max(1, len(eeg_buffer)))
            if len(vals) < 4:
                channel_thetas.append(0.0)
                channel_alphas.append(0.0)
                channel_betas.append(0.0)
                continue
            # Simplified band power estimation using variance in band approximations
            vals_clipped = np.clip(vals, -200.0, 200.0)
            total_var = float(np.var(vals_clipped)) + 1e-12
            channel_thetas.append(float(np.clip(np.var(vals_clipped[:len(vals_clipped)//3]) / total_var, 0, 1)))
            channel_alphas.append(float(np.clip(np.var(vals_clipped[len(vals_clipped)//3:2*len(vals_clipped)//3]) / total_var, 0, 1)))
            channel_betas.append(float(np.clip(np.var(vals_clipped[2*len(vals_clipped)//3:]) / total_var, 0, 1)))

        rel_theta = float(np.mean(channel_thetas)) if channel_thetas else 0.0
        rel_alpha = float(np.mean(channel_alphas)) if channel_alphas else 0.0
        rel_beta = float(np.mean(channel_betas)) if channel_betas else 0.0
        coverage = float(np.mean(coverage_list)) if coverage_list else 0.0
        clipping_count = 0

        features["eeg_rel_theta"] = rel_theta
        features["eeg_rel_alpha"] = rel_alpha
        features["eeg_rel_beta"] = rel_beta
        features["eeg_theta_alpha_ratio"] = rel_theta / (rel_alpha + 1e-6)
        features["eeg_beta_alpha_ratio"] = rel_beta / (rel_alpha + 1e-6)
        features["eeg_quality_coverage"] = coverage
        features["eeg_clipping_count"] = float(clipping_count)
        features["eeg_artifact_flag"] = 1.0 if coverage < 0.7 else 0.0
        features["eeg_quality_score"] = coverage if coverage >= 0.7 else coverage * 0.5

    # --- Optical features ---
    if optical_buffer.empty:
        features.update(_default_optical_features())
    else:
        opt_cols = [c for c in optical_buffer.columns if c.startswith("optics_")]
        if opt_cols:
            opt_vals = optical_buffer[opt_cols].mean(axis=1).to_numpy(dtype=float)
            opt_mean = float(np.mean(opt_vals)) if len(opt_vals) else 0.0
            opt_slope = float(np.polyfit(np.arange(len(opt_vals)), opt_vals, 1)[0]) if len(opt_vals) > 1 else 0.0
            opt_change = float(opt_vals[-1] - opt_vals[0]) if len(opt_vals) > 1 else 0.0
            opt_coverage = float(optical_buffer[opt_cols].notna().mean().mean())
        else:
            opt_mean, opt_slope, opt_change, opt_coverage = 0.0, 0.0, 0.0, 0.0

        features["optical_norm_mean"] = opt_mean
        features["optical_trend_slope"] = opt_slope
        features["optical_mean_change"] = opt_change
        features["optical_coverage"] = opt_coverage

    # --- IMU features ---
    if imu_buffer.empty:
        features.update(_default_imu_features())
    else:
        acc_cols = [c for c in imu_buffer.columns if c.startswith("acc_")]
        gyro_cols = [c for c in imu_buffer.columns if c.startswith("gyro_")]

        if acc_cols and gyro_cols:
            acc_mag = np.sqrt(sum(imu_buffer[c].fillna(0).to_numpy(dtype=float) ** 2 for c in acc_cols))
            gyro_mag = np.sqrt(sum(imu_buffer[c].fillna(0).to_numpy(dtype=float) ** 2 for c in gyro_cols))
            motion_raw = float(np.std(acc_mag) + np.std(gyro_mag))
            imu_motion = float(np.clip(motion_raw / 0.5, 0.0, 1.0))
            imu_coverage = float(imu_buffer[acc_cols + gyro_cols].notna().mean().mean())
        else:
            imu_motion, imu_coverage = 0.0, 0.0

        features["imu_motion_score"] = imu_motion
        features["imu_coverage"] = imu_coverage

    # --- Quality features ---
    opt_cov = features.get("optical_coverage", 0.0)
    imu_cov = features.get("imu_coverage", 0.0)
    imu_mot = features.get("imu_motion_score", 0.0)
    confidence = float(np.clip(0.6 * opt_cov + 0.2 * imu_cov + 0.2 * (1.0 - imu_mot), 0.0, 1.0))
    features["confidence_score"] = confidence
    features["downweight_factor"] = confidence
    features["alignment_gap_s"] = 0.0  # Real-time: no alignment gap

    return features


def _default_eeg_features() -> dict[str, float]:
    """Fallback EEG features when buffer is empty."""
    return {
        "eeg_rel_theta": 0.0,
        "eeg_rel_alpha": 0.0,
        "eeg_rel_beta": 0.0,
        "eeg_theta_alpha_ratio": 0.0,
        "eeg_beta_alpha_ratio": 0.0,
        "eeg_quality_coverage": 0.0,
        "eeg_clipping_count": 0.0,
        "eeg_artifact_flag": 1.0,
        "eeg_quality_score": 0.0,
    }


def _default_optical_features() -> dict[str, float]:
    """Fallback optical features when buffer is empty."""
    return {
        "optical_norm_mean": 0.0,
        "optical_trend_slope": 0.0,
        "optical_mean_change": 0.0,
        "optical_coverage": 0.0,
    }


def _default_imu_features() -> dict[str, float]:
    """Fallback IMU features when buffer is empty."""
    return {
        "imu_motion_score": 0.0,
        "imu_coverage": 0.0,
    }


# ---------------------------------------------------------------------------
# Fault handling (T12.3)
# ---------------------------------------------------------------------------

def detect_faults(
    state: InferenceState,
    current_time_s: float,
    config: InferenceConfig,
    eeg_buffer_empty: bool,
    optical_buffer_empty: bool,
    imu_buffer_empty: bool,
) -> list[str]:
    """Check for runtime faults: packet loss and delayed modality updates."""
    faults: list[str] = []

    # Stale EEG
    if current_time_s - state.last_eeg_update_s > config.stale_modality_timeout_s:
        faults.append("eeg_stale")

    # Stale optical
    if current_time_s - state.last_optical_update_s > config.stale_modality_timeout_s:
        faults.append("optical_stale")

    # Stale IMU
    if current_time_s - state.last_imu_update_s > config.stale_modality_timeout_s:
        faults.append("imu_stale")

    # Empty buffers
    if eeg_buffer_empty:
        faults.append("eeg_buffer_empty")
    if optical_buffer_empty:
        faults.append("optical_buffer_empty")
    if imu_buffer_empty:
        faults.append("imu_buffer_empty")

    # Consecutive fault tracking
    if faults:
        state.consecutive_faults += 1
        state.fault_count += 1
    else:
        state.consecutive_faults = 0

    if state.consecutive_faults >= config.max_consecutive_faults:
        faults.append("excessive_consecutive_faults")

    return faults


# ---------------------------------------------------------------------------
# Inference cycle (T12.2, T12.4)
# ---------------------------------------------------------------------------

def run_inference_cycle(
    eeg_buffer: pd.DataFrame,
    optical_buffer: pd.DataFrame,
    imu_buffer: pd.DataFrame,
    timestamp_s: float,
    models: dict[str, Any],
    config: InferenceConfig,
    state: InferenceState,
) -> InferenceUpdate:
    """Execute a single inference update cycle.

    1. Detect faults (T12.3)
    2. Build live features (T12.1)
    3. Normalize and predict (T12.2)
    4. Log latency and diagnostics (T12.4)
    """
    cycle_start = time.perf_counter()

    # T12.3 — Fault detection
    fault_flags = detect_faults(
        state=state,
        current_time_s=timestamp_s,
        config=config,
        eeg_buffer_empty=eeg_buffer.empty,
        optical_buffer_empty=optical_buffer.empty,
        imu_buffer_empty=imu_buffer.empty,
    )

    # T12.1 — Build live features
    raw_features = build_live_features(
        eeg_buffer=eeg_buffer,
        optical_buffer=optical_buffer,
        imu_buffer=imu_buffer,
        timestamp_s=timestamp_s,
        config=config,
    )

    # Normalize
    feature_columns: list[str] = models["feature_columns"]
    scaler_metadata: dict[str, Any] | None = models["scaler_metadata"]
    normalized = _apply_live_normalization(raw_features, feature_columns, scaler_metadata)

    # Build feature array in correct column order
    x = np.array([[normalized.get(col, 0.0) for col in feature_columns]])

    # T12.2 — Model prediction
    linear_model = models["linear_model"]
    tree_model = models["tree_model"]
    fusion_weights = models["fusion_weights"]

    try:
        linear_prob = float(linear_model.predict_proba(x)[:, 1][0])
    except Exception:
        linear_prob = 0.5
        fault_flags.append("linear_model_error")

    try:
        tree_prob = float(tree_model.predict_proba(x)[:, 1][0])
    except Exception:
        tree_prob = 0.5
        fault_flags.append("tree_model_error")

    w_linear = float(fusion_weights.get("linear", 0.5))
    w_tree = float(fusion_weights.get("tree", 0.5))
    risk_score = float(np.clip(w_linear * linear_prob + w_tree * tree_prob, 0.0, 1.0))

    # Confidence: based on model agreement and data quality
    model_agreement = 1.0 - abs(linear_prob - tree_prob)
    data_quality = raw_features.get("confidence_score", 0.5)
    model_confidence = float(np.clip(0.6 * model_agreement + 0.4 * data_quality, 0.0, 1.0))

    # Degrade confidence and clamp risk if excessive faults
    if "excessive_consecutive_faults" in fault_flags:
        model_confidence *= 0.5
        risk_score = float(np.clip(risk_score, 0.3, 0.7))  # Pull toward uncertain

    # T12.4 — Latency and diagnostics
    cycle_latency = time.perf_counter() - cycle_start
    state.update_count += 1
    state.risk_score_history.append(risk_score)
    state.latency_history.append(cycle_latency)

    diagnostics = {
        "update_number": state.update_count,
        "linear_prob": linear_prob,
        "tree_prob": tree_prob,
        "model_agreement": model_agreement,
        "data_quality": data_quality,
        "consecutive_faults": state.consecutive_faults,
        "total_faults": state.fault_count,
        "mean_latency_s": float(np.mean(list(state.latency_history))),
        "max_latency_s": float(np.max(list(state.latency_history))),
        "within_budget": cycle_latency <= config.latency_budget_s,
    }

    LOGGER.info(
        "Inference #%d | risk=%.4f | conf=%.4f | latency=%.4fs | faults=%s",
        state.update_count,
        risk_score,
        model_confidence,
        cycle_latency,
        fault_flags or "none",
    )

    return InferenceUpdate(
        timestamp_s=timestamp_s,
        risk_score=risk_score,
        model_confidence=model_confidence,
        feature_vector=normalized,
        latency_s=cycle_latency,
        fault_flags=fault_flags,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Replay-mode inference runner
# ---------------------------------------------------------------------------

def run_replay_inference(
    labeled_frame: pd.DataFrame,
    models: dict[str, Any],
    config: InferenceConfig | None = None,
) -> list[InferenceUpdate]:
    """Run inference over a recorded session (replay mode).

    Iterates over the labeled feature matrix row-by-row, emitting
    InferenceUpdates as though in a live loop.  Used for validation
    and demo rehearsal.
    """
    if config is None:
        config = InferenceConfig()

    state = InferenceState()
    updates: list[InferenceUpdate] = []

    feature_columns: list[str] = models["feature_columns"]
    scaler_metadata: dict[str, Any] | None = models["scaler_metadata"]
    linear_model = models["linear_model"]
    tree_model = models["tree_model"]
    fusion_weights = models["fusion_weights"]

    for idx, row in labeled_frame.iterrows():
        cycle_start = time.perf_counter()
        timestamp_s = float(row.get("start_time_s", float(idx)))

        # Build normalized feature vector from stored z-columns
        normalized: dict[str, float] = {}
        for col in feature_columns:
            normalized[col] = float(row.get(col, 0.0))

        x = np.array([[normalized.get(col, 0.0) for col in feature_columns]])

        try:
            linear_prob = float(linear_model.predict_proba(x)[:, 1][0])
        except Exception:
            linear_prob = 0.5

        try:
            tree_prob = float(tree_model.predict_proba(x)[:, 1][0])
        except Exception:
            tree_prob = 0.5

        w_linear = float(fusion_weights.get("linear", 0.5))
        w_tree = float(fusion_weights.get("tree", 0.5))
        risk_score = float(np.clip(w_linear * linear_prob + w_tree * tree_prob, 0.0, 1.0))

        model_agreement = 1.0 - abs(linear_prob - tree_prob)
        model_confidence = float(np.clip(model_agreement, 0.0, 1.0))

        cycle_latency = time.perf_counter() - cycle_start
        state.update_count += 1
        state.risk_score_history.append(risk_score)
        state.latency_history.append(cycle_latency)

        updates.append(
            InferenceUpdate(
                timestamp_s=timestamp_s,
                risk_score=risk_score,
                model_confidence=model_confidence,
                feature_vector=normalized,
                latency_s=cycle_latency,
                fault_flags=[],
                diagnostics={
                    "update_number": state.update_count,
                    "linear_prob": linear_prob,
                    "tree_prob": tree_prob,
                    "replay_index": int(idx),
                },
            )
        )

    LOGGER.info(
        "Replay inference complete: %d updates, mean latency %.6fs",
        len(updates),
        float(np.mean([u.latency_s for u in updates])) if updates else 0.0,
    )
    return updates


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def write_inference_diagnostics(
    updates: list[InferenceUpdate],
    output_path: str | Path,
) -> None:
    """Persist per-update inference diagnostics log."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "timestamp_s": u.timestamp_s,
            "risk_score": u.risk_score,
            "model_confidence": u.model_confidence,
            "latency_s": u.latency_s,
            "fault_flags": u.fault_flags,
            "diagnostics": u.diagnostics,
        }
        for u in updates
    ]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    LOGGER.info("Inference diagnostics written to %s", path)
