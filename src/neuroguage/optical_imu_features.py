from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class OpticalIMUFeatureError(ValueError):
    """Raised when optical/IMU quality feature extraction fails."""


@dataclass
class OpticalIMUFeatureResult:
    feature_frame: pd.DataFrame
    summary: dict[str, Any]


def _safe_interpolate(values: pd.Series) -> np.ndarray:
    if values.isna().all():
        return np.zeros(len(values), dtype=float)

    filled = values.interpolate(limit_direction="both").fillna(float(values.mean()))
    return filled.to_numpy(dtype=float)


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-9:
        return np.zeros_like(values)
    return (values - mean) / std


def _trend_slope(time_values: np.ndarray, signal_values: np.ndarray) -> float:
    if len(time_values) < 2:
        return 0.0

    x = time_values.astype(float)
    y = signal_values.astype(float)

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 1e-12:
        return 0.0

    return float(np.dot(x_centered, y_centered) / denominator)


def extract_optical_imu_window_features(
    synchronized_frame: pd.DataFrame,
    optical_windows: pd.DataFrame,
) -> OpticalIMUFeatureResult:
    if synchronized_frame.empty:
        raise OpticalIMUFeatureError("synchronized_frame is empty")

    if optical_windows.empty:
        raise OpticalIMUFeatureError("optical_windows is empty")

    optical_columns = [
        column
        for column in synchronized_frame.columns
        if column.startswith("optical_optics_")
    ]
    imu_columns = [
        "imu_acc_x",
        "imu_acc_y",
        "imu_acc_z",
        "imu_gyro_x",
        "imu_gyro_y",
        "imu_gyro_z",
    ]

    if not optical_columns:
        raise OpticalIMUFeatureError("No optical columns found in synchronized_frame")

    if not set(imu_columns).issubset(synchronized_frame.columns):
        raise OpticalIMUFeatureError("Missing one or more required IMU columns")

    rows: list[dict[str, Any]] = []

    for _, window in optical_windows.iterrows():
        start_idx = int(window["start_index"])
        end_idx = int(window["end_index"])

        segment = synchronized_frame.loc[start_idx:end_idx].copy()
        time_values = segment["time_s"].to_numpy(dtype=float)

        optical_arrays = [_safe_interpolate(segment[column]) for column in optical_columns]
        optical_matrix = np.vstack(optical_arrays) if optical_arrays else np.zeros((1, len(segment)))

        optical_norm = np.vstack([_zscore(row) for row in optical_matrix])
        optical_mean_signal = np.mean(optical_norm, axis=0)

        optical_norm_mean = float(np.mean(optical_mean_signal))
        optical_slope = _trend_slope(time_values=time_values, signal_values=optical_mean_signal)
        optical_mean_change = float(optical_mean_signal[-1] - optical_mean_signal[0]) if len(optical_mean_signal) else 0.0

        optical_coverage = float(segment[optical_columns].notna().mean().mean())

        acc_x = _safe_interpolate(segment["imu_acc_x"])
        acc_y = _safe_interpolate(segment["imu_acc_y"])
        acc_z = _safe_interpolate(segment["imu_acc_z"])
        gyro_x = _safe_interpolate(segment["imu_gyro_x"])
        gyro_y = _safe_interpolate(segment["imu_gyro_y"])
        gyro_z = _safe_interpolate(segment["imu_gyro_z"])

        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        motion_raw = float(np.std(acc_mag) + np.std(gyro_mag))
        imu_motion_score = float(np.clip(motion_raw / 0.5, 0.0, 1.0))

        imu_coverage = float(segment[imu_columns].notna().mean().mean())

        confidence_score = float(
            np.clip(
                (0.6 * optical_coverage) + (0.2 * imu_coverage) + (0.2 * (1.0 - imu_motion_score)),
                0.0,
                1.0,
            )
        )
        downweight_factor = confidence_score
        quality_flag = bool(confidence_score < 0.5)

        rows.append(
            {
                "window_id": int(window["window_id"]),
                "start_time_s": float(window["start_time_s"]),
                "end_time_s": float(window["end_time_s"]),
                "optical_norm_mean": optical_norm_mean,
                "optical_trend_slope": optical_slope,
                "optical_mean_change": optical_mean_change,
                "imu_motion_score": imu_motion_score,
                "optical_coverage": optical_coverage,
                "imu_coverage": imu_coverage,
                "confidence_score": confidence_score,
                "downweight_factor": downweight_factor,
                "quality_flag": quality_flag,
            }
        )

    feature_frame = pd.DataFrame(rows)

    summary = {
        "window_count": int(len(feature_frame)),
        "quality_flag_count": int(feature_frame["quality_flag"].sum()),
        "mean_confidence_score": float(feature_frame["confidence_score"].mean()),
    }

    return OpticalIMUFeatureResult(feature_frame=feature_frame, summary=summary)


def write_optical_imu_feature_artifacts(
    result: OpticalIMUFeatureResult,
    feature_output_path: str | Path,
    summary_output_path: str | Path,
) -> None:
    feature_path = Path(feature_output_path)
    summary_path = Path(summary_output_path)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    result.feature_frame.to_csv(feature_path, index=False)
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
