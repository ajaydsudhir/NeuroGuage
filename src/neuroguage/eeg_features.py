from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch


class EEGFeatureError(ValueError):
    """Raised when EEG feature extraction requirements are not satisfied."""


@dataclass
class EEGFeatureResult:
    feature_frame: pd.DataFrame
    summary: dict[str, Any]


def _estimate_sampling_rate(time_values: np.ndarray) -> float:
    if time_values.size < 2:
        return 1.0

    diffs = np.diff(np.sort(np.unique(time_values)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0

    return float(1.0 / np.median(diffs))


def _sanitize_signal(signal: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(signal, dtype=float)

    if np.all(np.isnan(values)):
        values = np.zeros_like(values)

    finite_median = float(np.nanmedian(values)) if np.any(~np.isnan(values)) else 0.0
    values = np.nan_to_num(values, nan=finite_median)

    clipped = np.clip(values, -200.0, 200.0)
    clipping_count = int(np.sum(np.abs(values) > 200.0))
    return clipped, clipping_count


def _bandpass_filter(signal: np.ndarray, fs_hz: float, low_hz: float = 1.0, high_hz: float = 40.0) -> np.ndarray:
    if signal.size < 8:
        return signal

    nyquist = fs_hz / 2.0
    if nyquist <= 0:
        return signal

    high = min(high_hz, nyquist * 0.95)
    low = max(0.1, min(low_hz, high * 0.5))

    if high <= low:
        return signal

    b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
    try:
        return filtfilt(b, a, signal)
    except ValueError:
        return signal


def _relative_band_powers(filtered_signal: np.ndarray, fs_hz: float) -> dict[str, float]:
    if filtered_signal.size < 4:
        return {
            "rel_theta": 0.0,
            "rel_alpha": 0.0,
            "rel_beta": 0.0,
        }

    freqs, power = welch(filtered_signal, fs=fs_hz, nperseg=min(128, filtered_signal.size))
    power = np.maximum(power, 0.0)

    total_mask = (freqs >= 1.0) & (freqs <= 40.0)
    total_power = float(power[total_mask].sum())
    if total_power <= 1e-12:
        return {
            "rel_theta": 0.0,
            "rel_alpha": 0.0,
            "rel_beta": 0.0,
        }

    def _band(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(power[mask].sum() / total_power)

    return {
        "rel_theta": max(0.0, min(1.0, _band(4.0, 8.0))),
        "rel_alpha": max(0.0, min(1.0, _band(8.0, 12.0))),
        "rel_beta": max(0.0, min(1.0, _band(12.0, 30.0))),
    }


def _extract_channel_features(
    frame: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    channel: str,
) -> dict[str, float | int | bool]:
    segment = frame.loc[start_idx:end_idx, ["time_s", channel]].copy()
    coverage = float(segment[channel].notna().mean())

    signal, clipping_count = _sanitize_signal(segment[channel].to_numpy(dtype=float))
    fs_hz = _estimate_sampling_rate(segment["time_s"].to_numpy(dtype=float))
    filtered = _bandpass_filter(signal=signal, fs_hz=fs_hz)

    bands = _relative_band_powers(filtered_signal=filtered, fs_hz=fs_hz)
    channel_std = float(np.std(filtered)) if filtered.size else 0.0

    artifact_flag = (coverage < 0.7) or (clipping_count > 0) or (channel_std > 75.0)

    return {
        "rel_theta": bands["rel_theta"],
        "rel_alpha": bands["rel_alpha"],
        "rel_beta": bands["rel_beta"],
        "coverage": coverage,
        "clipping_count": clipping_count,
        "artifact_flag": artifact_flag,
    }


def extract_eeg_window_features(
    synchronized_frame: pd.DataFrame,
    eeg_windows: pd.DataFrame,
    channel_columns: list[str] | None = None,
) -> EEGFeatureResult:
    if synchronized_frame.empty:
        raise EEGFeatureError("synchronized_frame is empty")

    if eeg_windows.empty:
        raise EEGFeatureError("eeg_windows is empty")

    if channel_columns is None:
        channel_columns = [
            column
            for column in synchronized_frame.columns
            if column.startswith("eeg_eeg_")
        ]

    if not channel_columns:
        raise EEGFeatureError("No EEG channel columns found in synchronized_frame")

    rows: list[dict[str, Any]] = []
    for _, window in eeg_windows.iterrows():
        start_idx = int(window["start_index"])
        end_idx = int(window["end_index"])

        per_channel = [
            _extract_channel_features(
                frame=synchronized_frame,
                start_idx=start_idx,
                end_idx=end_idx,
                channel=channel,
            )
            for channel in channel_columns
        ]

        rel_theta = float(np.mean([float(item["rel_theta"]) for item in per_channel]))
        rel_alpha = float(np.mean([float(item["rel_alpha"]) for item in per_channel]))
        rel_beta = float(np.mean([float(item["rel_beta"]) for item in per_channel]))

        theta_alpha_ratio = rel_theta / (rel_alpha + 1e-6)
        beta_alpha_ratio = rel_beta / (rel_alpha + 1e-6)

        quality_coverage = float(np.mean([float(item["coverage"]) for item in per_channel]))
        clipping_total = int(np.sum([int(item["clipping_count"]) for item in per_channel]))
        artifact_flag = bool(any(bool(item["artifact_flag"]) for item in per_channel))

        quality_score = quality_coverage if not artifact_flag else quality_coverage * 0.5

        rows.append(
            {
                "window_id": int(window["window_id"]),
                "start_time_s": float(window["start_time_s"]),
                "end_time_s": float(window["end_time_s"]),
                "eeg_rel_theta": rel_theta,
                "eeg_rel_alpha": rel_alpha,
                "eeg_rel_beta": rel_beta,
                "eeg_theta_alpha_ratio": float(theta_alpha_ratio),
                "eeg_beta_alpha_ratio": float(beta_alpha_ratio),
                "eeg_quality_coverage": quality_coverage,
                "eeg_clipping_count": clipping_total,
                "eeg_artifact_flag": artifact_flag,
                "eeg_quality_score": float(quality_score),
            }
        )

    feature_frame = pd.DataFrame(rows)

    summary = {
        "window_count": int(len(feature_frame)),
        "artifact_window_count": int(feature_frame["eeg_artifact_flag"].sum()),
        "mean_quality_score": float(feature_frame["eeg_quality_score"].mean()),
    }

    return EEGFeatureResult(feature_frame=feature_frame, summary=summary)


def write_eeg_feature_artifacts(
    result: EEGFeatureResult,
    feature_output_path: str | Path,
    summary_output_path: str | Path,
) -> None:
    feature_path = Path(feature_output_path)
    summary_path = Path(summary_output_path)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    result.feature_frame.to_csv(feature_path, index=False)
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
