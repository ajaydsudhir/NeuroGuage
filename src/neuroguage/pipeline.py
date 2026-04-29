"""Phase 1 + Phase 2 Offline Pipeline.

This module groups all non-live experiment pipeline stages:

- Phase 1: EEG-centric attention signal preparation
- Phase 2: Optical/IMU fatigue-related signal preparation
- Shared: schema validation, feature assembly, weak labels, train/eval utilities

It is intentionally consolidated so users can find the full offline path in
one file.
"""

from __future__ import annotations


from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CONTRACT_PATH = Path("configs/data_contract.json")


def load_data_contract(contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    path = Path(contract_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class IngestionError(ValueError):
    """Raised when ingestion or synchronization requirements are not met."""


@dataclass
class ModalityDiagnostics:
    modality: str
    row_count: int
    expected_hz: float
    start_offset_s: float
    end_offset_s: float
    dropout_count: int
    max_packet_delay_s: float
    drift_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "row_count": self.row_count,
            "expected_hz": self.expected_hz,
            "start_offset_s": self.start_offset_s,
            "end_offset_s": self.end_offset_s,
            "dropout_count": self.dropout_count,
            "max_packet_delay_s": self.max_packet_delay_s,
            "drift_s": self.drift_s,
        }


@dataclass
class SynchronizationDiagnostics:
    reference_timestamp_utc: str
    timeline_rows: int
    alignment_tolerance_s: float
    modalities: dict[str, ModalityDiagnostics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_timestamp_utc": self.reference_timestamp_utc,
            "timeline_rows": self.timeline_rows,
            "alignment_tolerance_s": self.alignment_tolerance_s,
            "modalities": {
                name: metrics.to_dict()
                for name, metrics in self.modalities.items()
            },
        }


@dataclass
class SynchronizationResult:
    synchronized_frame: pd.DataFrame
    diagnostics: SynchronizationDiagnostics
    validation_summary: dict[str, Any]


def _load_modality_frame(
    file_path: str | Path,
    modality_name: str,
    required_columns: list[str],
) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise IngestionError(f"{modality_name}: missing source file at {path}")

    frame = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise IngestionError(
            f"{modality_name}: missing required columns for ingestion: {missing_columns}"
        )

    frame = frame[required_columns].copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    if frame["timestamp_utc"].isna().any():
        raise IngestionError(f"{modality_name}: invalid timestamp_utc values")

    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    return frame


def ingest_modalities(
    eeg_path: str | Path,
    optical_path: str | Path,
    imu_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, pd.DataFrame]:
    contract = load_data_contract(contract_path)

    return {
        "eeg": _load_modality_frame(
            file_path=eeg_path,
            modality_name="eeg",
            required_columns=contract["modalities"]["eeg"]["required_columns"],
        ),
        "optical": _load_modality_frame(
            file_path=optical_path,
            modality_name="optical",
            required_columns=contract["modalities"]["optical"]["required_columns"],
        ),
        "imu": _load_modality_frame(
            file_path=imu_path,
            modality_name="imu",
            required_columns=contract["modalities"]["imu"]["required_columns"],
        ),
    }


def normalize_modality_timestamps(
    modalities: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    reference_timestamp = min(frame["timestamp_utc"].min() for frame in modalities.values())
    normalized: dict[str, pd.DataFrame] = {}

    for name, frame in modalities.items():
        updated = frame.copy()
        updated["time_s"] = (updated["timestamp_utc"] - reference_timestamp).dt.total_seconds()
        normalized[name] = updated

    return normalized, reference_timestamp


def _compute_modality_diagnostics(
    modality_name: str,
    frame: pd.DataFrame,
    expected_hz: float,
    global_start_s: float,
    global_end_s: float,
) -> ModalityDiagnostics:
    row_count = len(frame)
    if row_count == 0:
        raise IngestionError(f"{modality_name}: frame is empty")

    times = frame["time_s"].to_numpy(dtype=float)
    configured_dt = 1.0 / expected_hz if expected_hz > 0 else 0.0

    if row_count > 1:
        deltas = np.diff(times)
        observed_dt = float(np.median(deltas))
        # Use a conservative baseline interval so diagnostics remain meaningful
        # when replayed files are lower-rate than nominal device sampling.
        baseline_dt = max(configured_dt, observed_dt)

        dropout_count = int(np.sum(deltas > (baseline_dt * 1.5)))
        max_packet_delay_s = float(np.maximum(deltas - baseline_dt, 0.0).max())
        expected_duration = baseline_dt * (row_count - 1)
        observed_duration = float(times[-1] - times[0])
        drift_s = observed_duration - expected_duration
    else:
        dropout_count = 0
        max_packet_delay_s = 0.0
        drift_s = 0.0

    return ModalityDiagnostics(
        modality=modality_name,
        row_count=row_count,
        expected_hz=float(expected_hz),
        start_offset_s=float(times[0] - global_start_s),
        end_offset_s=float(global_end_s - times[-1]),
        dropout_count=dropout_count,
        max_packet_delay_s=max_packet_delay_s,
        drift_s=drift_s,
    )


def _build_synchronized_timeline(
    normalized_modalities: dict[str, pd.DataFrame],
    tolerance_s: float,
) -> pd.DataFrame:
    all_times = np.concatenate(
        [frame["time_s"].to_numpy(dtype=float) for frame in normalized_modalities.values()]
    )
    timeline = pd.DataFrame({"time_s": np.unique(np.round(all_times, 6))}).sort_values("time_s")

    synchronized = timeline.copy()
    for modality_name, frame in normalized_modalities.items():
        aligned = frame.copy()
        aligned["source_time_s"] = aligned["time_s"]

        rename_map: dict[str, str] = {}
        for column in aligned.columns:
            if column == "time_s":
                continue
            rename_map[column] = f"{modality_name}_{column}"

        aligned = aligned.rename(columns=rename_map).sort_values("time_s")

        synchronized = pd.merge_asof(
            synchronized,
            aligned,
            on="time_s",
            direction="nearest",
            tolerance=tolerance_s,
        )

    return synchronized


def _build_validation_summary(
    synchronized_frame: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    modality_missing_ratio: dict[str, float] = {}
    modality_match_ratio: dict[str, float] = {}

    for modality in ["eeg", "optical", "imu"]:
        signal_columns = [
            f"{modality}_{column}"
            for column in contract["modalities"][modality]["required_columns"]
            if column not in {"session_id", "timestamp_utc"}
        ]

        missing_ratio = float(synchronized_frame[signal_columns].isna().all(axis=1).mean())
        modality_missing_ratio[modality] = missing_ratio
        modality_match_ratio[modality] = 1.0 - missing_ratio

    return {
        "timeline_rows": int(len(synchronized_frame)),
        "modality_missing_ratio": modality_missing_ratio,
        "modality_match_ratio": modality_match_ratio,
    }


def synchronize_modalities(
    eeg_path: str | Path,
    optical_path: str | Path,
    imu_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    tolerance_s: float = 0.2,
) -> SynchronizationResult:
    contract = load_data_contract(contract_path)

    modalities = ingest_modalities(
        eeg_path=eeg_path,
        optical_path=optical_path,
        imu_path=imu_path,
        contract_path=contract_path,
    )

    normalized, reference_timestamp = normalize_modality_timestamps(modalities)
    global_start_s = min(frame["time_s"].min() for frame in normalized.values())
    global_end_s = max(frame["time_s"].max() for frame in normalized.values())

    diagnostics_by_modality: dict[str, ModalityDiagnostics] = {}
    for modality_name, frame in normalized.items():
        expected_hz = float(contract["modalities"][modality_name]["sampling_hz_assumption"])
        diagnostics_by_modality[modality_name] = _compute_modality_diagnostics(
            modality_name=modality_name,
            frame=frame,
            expected_hz=expected_hz,
            global_start_s=float(global_start_s),
            global_end_s=float(global_end_s),
        )

    synchronized = _build_synchronized_timeline(
        normalized_modalities=normalized,
        tolerance_s=tolerance_s,
    )
    synchronized.insert(
        0,
        "timestamp_utc",
        reference_timestamp + pd.to_timedelta(synchronized["time_s"], unit="s"),
    )

    diagnostics = SynchronizationDiagnostics(
        reference_timestamp_utc=reference_timestamp.isoformat(),
        timeline_rows=int(len(synchronized)),
        alignment_tolerance_s=tolerance_s,
        modalities=diagnostics_by_modality,
    )

    validation_summary = _build_validation_summary(
        synchronized_frame=synchronized,
        contract=contract,
    )

    return SynchronizationResult(
        synchronized_frame=synchronized,
        diagnostics=diagnostics,
        validation_summary=validation_summary,
    )


def write_synchronization_artifacts(
    result: SynchronizationResult,
    synchronized_output_path: str | Path,
    diagnostics_output_path: str | Path,
    validation_output_path: str | Path,
) -> None:
    synchronized_path = Path(synchronized_output_path)
    diagnostics_path = Path(diagnostics_output_path)
    validation_path = Path(validation_output_path)

    synchronized_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)

    export_frame = result.synchronized_frame.copy()
    export_frame["timestamp_utc"] = export_frame["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    export_frame.to_csv(synchronized_path, index=False)

    diagnostics_path.write_text(
        json.dumps(result.diagnostics.to_dict(), indent=2),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(result.validation_summary, indent=2),
        encoding="utf-8",
    )



from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class WindowingError(ValueError):
    """Raised when windowing requirements cannot be satisfied."""


@dataclass
class WindowingResult:
    eeg_windows: pd.DataFrame
    optical_windows: pd.DataFrame


def _estimate_baseline_dt(time_values: np.ndarray) -> float:
    if time_values.size < 2:
        return 0.0

    diffs = np.diff(np.sort(np.unique(time_values)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0

    return float(np.median(diffs))


def _build_windows_for_stream(
    frame: pd.DataFrame,
    window_type: str,
    window_size_s: float,
    step_s: float,
    time_col: str,
) -> pd.DataFrame:
    if frame.empty:
        raise WindowingError("Cannot build windows on empty synchronized frame")

    if window_size_s <= 0 or step_s <= 0:
        raise WindowingError("window_size_s and step_s must be positive")

    local_frame = frame.reset_index(drop=True).copy()
    if time_col not in local_frame.columns:
        raise WindowingError(f"Missing time column '{time_col}' in synchronized frame")

    local_frame[time_col] = local_frame[time_col].astype(float)
    session_start = float(local_frame[time_col].min())
    session_end = float(local_frame[time_col].max())

    baseline_dt = _estimate_baseline_dt(local_frame[time_col].to_numpy(dtype=float))

    starts = np.arange(session_start, session_end + step_s, step_s)

    rows: list[dict[str, Any]] = []
    window_id = 0
    for start in starts:
        end = float(start + window_size_s)

        mask = (local_frame[time_col] >= start) & (local_frame[time_col] < end)
        subset = local_frame.loc[mask]

        if subset.empty and end >= session_end:
            subset = local_frame.loc[local_frame[time_col] >= start]

        if subset.empty:
            continue

        subset_index = subset.index.to_list()
        subset_times = subset[time_col].to_numpy(dtype=float)

        if subset_times.size > 1 and baseline_dt > 0:
            window_diffs = np.diff(subset_times)
            missing_packet_count = int(np.sum(window_diffs > baseline_dt * 1.5))
        else:
            missing_packet_count = 0

        rows.append(
            {
                "window_id": window_id,
                "window_type": window_type,
                "start_time_s": float(start),
                "end_time_s": float(end),
                "center_time_s": float((start + end) / 2.0),
                "start_index": int(subset_index[0]),
                "end_index": int(subset_index[-1]),
                "absolute_timeline_start_idx": int(subset_index[0]),
                "absolute_timeline_end_idx": int(subset_index[-1]),
                "sample_count": int(len(subset)),
                "missing_packet_count": missing_packet_count,
            }
        )
        window_id += 1

    if not rows:
        raise WindowingError(f"No windows generated for {window_type}")

    window_frame = pd.DataFrame(rows)
    return window_frame


def build_windowing_layers(
    synchronized_frame: pd.DataFrame,
    time_col: str = "time_s",
    eeg_window_size_s: float = 2.0,
    eeg_step_s: float = 0.5,
    optical_window_size_s: float = 8.0,
    optical_step_s: float = 2.0,
) -> WindowingResult:
    eeg_windows = _build_windows_for_stream(
        frame=synchronized_frame,
        window_type="eeg_fast",
        window_size_s=eeg_window_size_s,
        step_s=eeg_step_s,
        time_col=time_col,
    )

    optical_windows = _build_windows_for_stream(
        frame=synchronized_frame,
        window_type="optical_slow",
        window_size_s=optical_window_size_s,
        step_s=optical_step_s,
        time_col=time_col,
    )

    return WindowingResult(
        eeg_windows=eeg_windows,
        optical_windows=optical_windows,
    )


def write_windowing_artifacts(
    result: WindowingResult,
    eeg_output_path: str | Path,
    optical_output_path: str | Path,
    summary_output_path: str | Path,
) -> None:
    eeg_path = Path(eeg_output_path)
    optical_path = Path(optical_output_path)
    summary_path = Path(summary_output_path)

    eeg_path.parent.mkdir(parents=True, exist_ok=True)
    optical_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    result.eeg_windows.to_csv(eeg_path, index=False)
    result.optical_windows.to_csv(optical_path, index=False)

    summary = {
        "eeg_windows": int(len(result.eeg_windows)),
        "optical_windows": int(len(result.optical_windows)),
        "eeg_missing_packet_max": int(result.eeg_windows["missing_packet_count"].max()),
        "optical_missing_packet_max": int(result.optical_windows["missing_packet_count"].max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")



from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch


# Phase 1 — EEG attention-oriented features


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


# Phase 2 — Optical/IMU fatigue-related features (fNIRS-style proxy)


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




from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FeatureBuilderError(ValueError):
    """Raised when feature builder requirements are not satisfied."""


@dataclass
class FeatureBuilderResult:
    unified_frame: pd.DataFrame
    normalized_frame: pd.DataFrame
    feature_columns: list[str]
    normalization_metadata: dict[str, Any]
    leakage_report: dict[str, Any]
    summary: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FeatureBuilderError(f"{frame_name} missing required columns: {missing}")


def _coerce_bool_columns(frame: pd.DataFrame) -> pd.DataFrame:
    updated = frame.copy()
    for column in updated.columns:
        if pd.api.types.is_bool_dtype(updated[column]):
            updated[column] = updated[column].astype(int)
    return updated


def build_unified_feature_matrix(
    eeg_feature_frame: pd.DataFrame,
    optical_imu_feature_frame: pd.DataFrame,
    session_column: str = "session_id",
    alignment_tolerance_s: float = 2.5,
    dataset_version: str = "day3_v1",
    feature_schema_version: str = "v1",
) -> pd.DataFrame:
    if eeg_feature_frame.empty:
        raise FeatureBuilderError("eeg_feature_frame is empty")

    if optical_imu_feature_frame.empty:
        raise FeatureBuilderError("optical_imu_feature_frame is empty")

    _require_columns(
        eeg_feature_frame,
        [session_column, "window_id", "start_time_s", "end_time_s"],
        "eeg_feature_frame",
    )
    _require_columns(
        optical_imu_feature_frame,
        [session_column, "window_id", "start_time_s", "end_time_s"],
        "optical_imu_feature_frame",
    )

    eeg_frame = _coerce_bool_columns(eeg_feature_frame)
    optical_frame = _coerce_bool_columns(optical_imu_feature_frame)

    eeg_sorted = eeg_frame.sort_values([session_column, "start_time_s", "window_id"]).reset_index(drop=True)
    optical_sorted = optical_frame.sort_values([session_column, "start_time_s", "window_id"]).reset_index(drop=True)

    optical_renamed = optical_sorted.rename(
        columns={
            "window_id": "optical_window_id",
            "start_time_s": "optical_start_time_s",
            "end_time_s": "optical_end_time_s",
        }
    )

    session_frames: list[pd.DataFrame] = []
    for session_id in eeg_sorted[session_column].astype(str).unique().tolist():
        eeg_session = (
            eeg_sorted.loc[eeg_sorted[session_column].astype(str) == session_id]
            .sort_values("start_time_s")
            .reset_index(drop=True)
        )
        optical_session = (
            optical_renamed.loc[optical_renamed[session_column].astype(str) == session_id]
            .sort_values("optical_start_time_s")
            .reset_index(drop=True)
        )

        if optical_session.empty:
            raise FeatureBuilderError(f"No optical/IMU windows found for session '{session_id}'")

        merged_session = pd.merge_asof(
            left=eeg_session,
            right=optical_session,
            left_on="start_time_s",
            right_on="optical_start_time_s",
            direction="nearest",
            tolerance=alignment_tolerance_s,
        )

        # Preserve the left-side session id as the canonical session key.
        if f"{session_column}_x" in merged_session.columns:
            merged_session[session_column] = merged_session[f"{session_column}_x"]
            merged_session = merged_session.drop(columns=[f"{session_column}_x"])
        if f"{session_column}_y" in merged_session.columns:
            merged_session = merged_session.drop(columns=[f"{session_column}_y"])

        session_frames.append(merged_session)

    unified = pd.concat(session_frames, ignore_index=True)

    if unified["optical_window_id"].isna().any():
        missing = int(unified["optical_window_id"].isna().sum())
        raise FeatureBuilderError(
            f"Unified join left {missing} EEG windows without optical/IMU context. "
            "Increase tolerance or validate optical window coverage."
        )

    unified["optical_window_id"] = unified["optical_window_id"].astype(int)
    unified["alignment_gap_s"] = (
        unified["start_time_s"].astype(float) - unified["optical_start_time_s"].astype(float)
    ).abs()
    unified["dataset_version"] = dataset_version
    unified["feature_schema_version"] = feature_schema_version
    unified["split_compatibility_marker"] = unified[session_column].astype(str)

    return unified


def build_session_split_map(
    session_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> dict[str, str]:
    unique_ids = sorted({str(session_id) for session_id in session_ids})
    if len(unique_ids) < 3:
        raise FeatureBuilderError("At least 3 sessions are required for train/val/test split isolation")

    rng = np.random.default_rng(seed)
    shuffled = np.array(unique_ids)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))

    if train_count + val_count >= total:
        val_count = 1
        train_count = total - 2

    train_sessions = shuffled[:train_count]
    val_sessions = shuffled[train_count : train_count + val_count]
    test_sessions = shuffled[train_count + val_count :]

    if len(test_sessions) == 0:
        raise FeatureBuilderError("Split policy produced empty test partition")

    split_map: dict[str, str] = {}
    for session_id in train_sessions:
        split_map[str(session_id)] = "train"
    for session_id in val_sessions:
        split_map[str(session_id)] = "val"
    for session_id in test_sessions:
        split_map[str(session_id)] = "test"

    return split_map


def apply_split_map(
    feature_frame: pd.DataFrame,
    split_map: dict[str, str],
    session_column: str = "session_id",
) -> pd.DataFrame:
    if feature_frame.empty:
        raise FeatureBuilderError("feature_frame is empty")

    updated = feature_frame.copy()
    updated["split"] = updated[session_column].astype(str).map(split_map)
    if updated["split"].isna().any():
        missing_sessions = sorted(
            updated.loc[updated["split"].isna(), session_column].astype(str).unique().tolist()
        )
        raise FeatureBuilderError(f"Missing split assignment for sessions: {missing_sessions}")

    return updated


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    include_prefixes = (
        "eeg_",
        "optical_",
        "imu_",
    )
    include_explicit = {
        "confidence_score",
        "downweight_factor",
        "alignment_gap_s",
    }
    excluded = {
        "window_id",
        "optical_window_id",
        "start_time_s",
        "end_time_s",
        "optical_start_time_s",
        "optical_end_time_s",
        "split",
        "fatigue_label",
        "weak_label",
        "weak_label_score",
        "weak_label_score_smoothed",
    }

    feature_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if column in include_explicit:
            feature_columns.append(column)
            continue
        if column.startswith(include_prefixes) and pd.api.types.is_numeric_dtype(frame[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise FeatureBuilderError("No numeric feature columns inferred for normalization")

    return feature_columns


def fit_train_only_normalization(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    split_column: str = "split",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(feature_frame, [split_column], "feature_frame")

    train_mask = feature_frame[split_column] == "train"
    if not bool(train_mask.any()):
        raise FeatureBuilderError("No train rows available for train-only normalization")

    normalized = feature_frame.copy()

    means = feature_frame.loc[train_mask, feature_columns].mean()
    stds = feature_frame.loc[train_mask, feature_columns].std(ddof=0)
    stds = stds.replace(0.0, 1.0).fillna(1.0)

    for column in feature_columns:
        normalized[f"z_{column}"] = (feature_frame[column] - float(means[column])) / float(stds[column])

    metadata = {
        "strategy": "train_only_zscore",
        "split_column": split_column,
        "feature_columns": feature_columns,
        "statistics": {
            column: {
                "mean": float(means[column]),
                "std": float(stds[column]),
            }
            for column in feature_columns
        },
    }

    return normalized, metadata


def build_leakage_report(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str] | None = None,
    session_column: str = "session_id",
    split_column: str = "split",
) -> dict[str, Any]:
    target_candidates = target_columns or ["fatigue_label", "weak_label", "weak_label_score"]
    contaminated_features = sorted(set(feature_columns).intersection(target_candidates))

    overlap_sessions: dict[str, list[str]] = {}
    grouped = frame.groupby(session_column)[split_column].nunique()
    violating_sessions = grouped[grouped > 1].index.tolist()
    for session_id in violating_sessions:
        splits = sorted(frame.loc[frame[session_column] == session_id, split_column].unique().tolist())
        overlap_sessions[str(session_id)] = [str(split) for split in splits]

    target_pass = len(contaminated_features) == 0
    overlap_pass = len(overlap_sessions) == 0

    return {
        "target_contamination": {
            "pass": target_pass,
            "contaminated_feature_columns": contaminated_features,
        },
        "session_overlap": {
            "pass": overlap_pass,
            "violating_sessions": overlap_sessions,
        },
        "overall_pass": bool(target_pass and overlap_pass),
    }


def run_feature_builder_pipeline(
    eeg_feature_frame: pd.DataFrame,
    optical_imu_feature_frame: pd.DataFrame,
    session_column: str = "session_id",
    split_seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    alignment_tolerance_s: float = 2.5,
    dataset_version: str = "day3_v1",
    feature_schema_version: str = "v1",
) -> FeatureBuilderResult:
    unified = build_unified_feature_matrix(
        eeg_feature_frame=eeg_feature_frame,
        optical_imu_feature_frame=optical_imu_feature_frame,
        session_column=session_column,
        alignment_tolerance_s=alignment_tolerance_s,
        dataset_version=dataset_version,
        feature_schema_version=feature_schema_version,
    )

    split_map = build_session_split_map(
        session_ids=unified[session_column].astype(str).tolist(),
        seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    unified_with_split = apply_split_map(unified, split_map=split_map, session_column=session_column)

    feature_columns = infer_feature_columns(unified_with_split)
    normalized, normalization_metadata = fit_train_only_normalization(
        feature_frame=unified_with_split,
        feature_columns=feature_columns,
        split_column="split",
    )
    leakage_report = build_leakage_report(
        frame=normalized,
        feature_columns=feature_columns,
        session_column=session_column,
        split_column="split",
    )

    summary = {
        "dataset_version": dataset_version,
        "feature_schema_version": feature_schema_version,
        "rows": int(len(normalized)),
        "sessions": int(normalized[session_column].nunique()),
        "split_counts": {
            split: int(count)
            for split, count in normalized["split"].value_counts(dropna=False).to_dict().items()
        },
        "feature_column_count": int(len(feature_columns)),
        "leakage_pass": bool(leakage_report["overall_pass"]),
    }

    return FeatureBuilderResult(
        unified_frame=unified_with_split,
        normalized_frame=normalized,
        feature_columns=feature_columns,
        normalization_metadata=normalization_metadata,
        leakage_report=leakage_report,
        summary=summary,
    )


def write_feature_builder_artifacts(
    result: FeatureBuilderResult,
    unified_output_path: str | Path,
    normalized_output_path: str | Path,
    summary_output_path: str | Path,
    normalization_output_path: str | Path,
    leakage_output_path: str | Path,
) -> None:
    unified_path = Path(unified_output_path)
    normalized_path = Path(normalized_output_path)
    summary_path = Path(summary_output_path)
    normalization_path = Path(normalization_output_path)
    leakage_path = Path(leakage_output_path)

    unified_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_path.parent.mkdir(parents=True, exist_ok=True)

    result.unified_frame.to_csv(unified_path, index=False)
    result.normalized_frame.to_csv(normalized_path, index=False)
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    normalization_path.write_text(json.dumps(result.normalization_metadata, indent=2), encoding="utf-8")
    leakage_path.write_text(json.dumps(result.leakage_report, indent=2), encoding="utf-8")



from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LabelingError(ValueError):
    """Raised when labeling or split-generation requirements are not satisfied."""


@dataclass
class LabelingResult:
    labeled_frame: pd.DataFrame
    split_manifest: dict[str, Any]
    summary: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise LabelingError(f"{frame_name} missing required columns: {missing}")


def _safe_minmax(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if maximum - minimum <= 1e-12:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - minimum) / (maximum - minimum)


def generate_weak_labels(
    feature_frame: pd.DataFrame,
    annotation_frame: pd.DataFrame,
    session_column: str = "session_id",
    window_column: str = "window_id",
    task_phase_column: str = "task_phase",
    subjective_column: str = "subjective_fatigue",
    performance_column: str = "performance_drift",
    enable_smoothing: bool = True,
    smoothing_window: int = 3,
    label_threshold: float = 0.55,
) -> pd.DataFrame:
    if feature_frame.empty:
        raise LabelingError("feature_frame is empty")
    if annotation_frame.empty:
        raise LabelingError("annotation_frame is empty")

    _require_columns(feature_frame, [session_column, window_column], "feature_frame")
    _require_columns(
        annotation_frame,
        [
            session_column,
            window_column,
            task_phase_column,
            subjective_column,
            performance_column,
        ],
        "annotation_frame",
    )

    merged = feature_frame.merge(
        annotation_frame,
        on=[session_column, window_column],
        how="left",
        validate="one_to_one",
    )

    if merged[[task_phase_column, subjective_column, performance_column]].isna().any().any():
        raise LabelingError("Annotations are missing for one or more feature rows")

    phase_score_map = {
        "baseline": 0.10,
        "sustained_attention": 0.45,
        "n_back": 0.75,
        "code_comprehension": 0.85,
    }
    merged["task_phase_score"] = (
        merged[task_phase_column].astype(str).str.lower().map(phase_score_map).fillna(0.5)
    )

    merged["subjective_norm"] = merged.groupby(session_column)[subjective_column].transform(_safe_minmax)
    merged["performance_norm"] = merged.groupby(session_column)[performance_column].transform(_safe_minmax)

    merged["weak_label_score"] = (
        0.4 * merged["task_phase_score"]
        + 0.35 * merged["subjective_norm"]
        + 0.25 * merged["performance_norm"]
    ).clip(0.0, 1.0)

    if enable_smoothing:
        if smoothing_window < 1:
            raise LabelingError("smoothing_window must be >= 1")
        merged = merged.sort_values([session_column, window_column]).reset_index(drop=True)
        merged["weak_label_score_smoothed"] = merged.groupby(session_column)["weak_label_score"].transform(
            lambda series: series.rolling(window=smoothing_window, min_periods=1, center=True).mean()
        )
    else:
        merged["weak_label_score_smoothed"] = merged["weak_label_score"]

    merged["fatigue_label"] = (merged["weak_label_score_smoothed"] >= label_threshold).astype(int)
    return merged


def build_session_split_manifest(
    session_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> dict[str, Any]:
    unique_ids = sorted({str(session_id) for session_id in session_ids})
    if len(unique_ids) < 3:
        raise LabelingError("At least 3 unique sessions are required for train/val/test isolation")

    rng = np.random.default_rng(seed)
    shuffled = np.array(unique_ids)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))
    if train_count + val_count >= total:
        val_count = 1
        train_count = total - 2

    train_sessions = [str(session_id) for session_id in shuffled[:train_count]]
    val_sessions = [str(session_id) for session_id in shuffled[train_count : train_count + val_count]]
    test_sessions = [str(session_id) for session_id in shuffled[train_count + val_count :]]

    if not test_sessions:
        raise LabelingError("Split policy produced empty test partition")

    split_map: dict[str, str] = {}
    for session_id in train_sessions:
        split_map[session_id] = "train"
    for session_id in val_sessions:
        split_map[session_id] = "val"
    for session_id in test_sessions:
        split_map[session_id] = "test"

    return {
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": max(0.0, 1.0 - train_ratio - val_ratio),
        },
        "splits": {
            "train": train_sessions,
            "val": val_sessions,
            "test": test_sessions,
        },
        "session_split_map": split_map,
    }


def apply_split_manifest(
    frame: pd.DataFrame,
    split_manifest: dict[str, Any],
    session_column: str = "session_id",
) -> pd.DataFrame:
    if "session_split_map" not in split_manifest:
        raise LabelingError("split_manifest missing session_split_map")

    split_map = {
        str(session_id): str(split)
        for session_id, split in split_manifest["session_split_map"].items()
    }

    updated = frame.copy()
    updated["split"] = updated[session_column].astype(str).map(split_map)
    if updated["split"].isna().any():
        missing = sorted(updated.loc[updated["split"].isna(), session_column].astype(str).unique().tolist())
        raise LabelingError(f"Missing split assignment for sessions: {missing}")

    return updated


def validate_split_isolation(
    frame: pd.DataFrame,
    session_column: str = "session_id",
    split_column: str = "split",
) -> dict[str, Any]:
    session_split_counts = frame.groupby(session_column)[split_column].nunique()
    violating_sessions = sorted(
        [str(session_id) for session_id, count in session_split_counts.items() if int(count) > 1]
    )

    split_counts = {
        str(split): int(count)
        for split, count in frame[split_column].value_counts(dropna=False).to_dict().items()
    }

    required_splits_present = all(split in split_counts for split in ["train", "val", "test"])
    non_empty_required_splits = all(split_counts.get(split, 0) > 0 for split in ["train", "val", "test"])

    return {
        "session_overlap_pass": len(violating_sessions) == 0,
        "required_splits_present": required_splits_present,
        "required_splits_non_empty": non_empty_required_splits,
        "violating_sessions": violating_sessions,
        "split_counts": split_counts,
        "overall_pass": bool(
            len(violating_sessions) == 0 and required_splits_present and non_empty_required_splits
        ),
    }


def run_labeling_pipeline(
    feature_frame: pd.DataFrame,
    annotation_frame: pd.DataFrame,
    session_column: str = "session_id",
    window_column: str = "window_id",
    split_seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    enable_smoothing: bool = True,
    smoothing_window: int = 3,
) -> LabelingResult:
    labeled = generate_weak_labels(
        feature_frame=feature_frame,
        annotation_frame=annotation_frame,
        session_column=session_column,
        window_column=window_column,
        enable_smoothing=enable_smoothing,
        smoothing_window=smoothing_window,
    )

    split_manifest = build_session_split_manifest(
        session_ids=labeled[session_column].astype(str).tolist(),
        seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    labeled = apply_split_manifest(
        frame=labeled,
        split_manifest=split_manifest,
        session_column=session_column,
    )

    split_validation = validate_split_isolation(
        frame=labeled,
        session_column=session_column,
        split_column="split",
    )
    if not split_validation["overall_pass"]:
        raise LabelingError(f"Split isolation validation failed: {split_validation}")

    summary = {
        "rows": int(len(labeled)),
        "sessions": int(labeled[session_column].nunique()),
        "label_distribution": {
            str(label): int(count)
            for label, count in labeled["fatigue_label"].value_counts(dropna=False).to_dict().items()
        },
        "weak_label_score_mean": float(labeled["weak_label_score_smoothed"].mean()),
        "split_validation": split_validation,
        "smoothing": {
            "enabled": bool(enable_smoothing),
            "window": int(smoothing_window),
        },
    }

    return LabelingResult(
        labeled_frame=labeled,
        split_manifest=split_manifest,
        summary=summary,
    )


def write_labeling_artifacts(
    result: LabelingResult,
    labeled_output_path: str | Path,
    split_manifest_output_path: str | Path,
    split_table_output_path: str | Path,
    summary_output_path: str | Path,
) -> None:
    labeled_path = Path(labeled_output_path)
    manifest_path = Path(split_manifest_output_path)
    split_table_path = Path(split_table_output_path)
    summary_path = Path(summary_output_path)

    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_table_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    result.labeled_frame.to_csv(labeled_path, index=False)
    manifest_path.write_text(json.dumps(result.split_manifest, indent=2), encoding="utf-8")

    split_rows = [
        {"session_id": session_id, "split": split}
        for session_id, split in result.split_manifest["session_split_map"].items()
    ]
    split_table = pd.DataFrame(split_rows).sort_values("session_id").reset_index(drop=True)
    split_table.to_csv(split_table_path, index=False)

    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")



from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


class TrainingError(ValueError):
    """Raised when MVP model training cannot proceed."""


@dataclass
class ModelSelectionResult:
    name: str
    estimator: Any
    parameters: dict[str, Any]
    validation_metrics: dict[str, Any]


@dataclass
class TrainingResult:
    linear_result: ModelSelectionResult
    tree_result: ModelSelectionResult
    fusion_weights: dict[str, float]
    metrics: dict[str, Any]
    feature_columns: list[str]
    scaler_metadata: dict[str, Any] | None


def _safe_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_pred = (probabilities >= 0.5).astype(int)

    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        metrics["auroc"] = None
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, probabilities))

    return metrics


def _selection_score(metrics: dict[str, Any]) -> float:
    auroc = metrics.get("auroc")
    auroc_score = float(auroc) if auroc is not None else 0.0
    return (0.7 * float(metrics["macro_f1"])) + (0.2 * float(metrics["balanced_accuracy"])) + (0.1 * auroc_score)


def _calibrate_estimator(base_estimator: Any, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    class_counts = np.bincount(y_train)
    min_count = int(class_counts.min()) if class_counts.size > 1 else 0

    if min_count < 2:
        return base_estimator

    cv = min(3, min_count)
    calibrator = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=cv)
    calibrator.fit(x_train, y_train)
    return calibrator


def _fit_linear_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> ModelSelectionResult:
    candidates: list[ModelSelectionResult] = []

    for c_value in [0.1, 1.0, 10.0]:
        estimator = LogisticRegression(
            C=c_value,
            max_iter=600,
            class_weight="balanced",
            random_state=random_state,
        )
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_val)[:, 1]
        metrics = _safe_metrics(y_val, probabilities)
        candidates.append(
            ModelSelectionResult(
                name="logistic_regression",
                estimator=estimator,
                parameters={"C": c_value},
                validation_metrics=metrics,
            )
        )

    lda_configs = [
        {"solver": "svd", "shrinkage": None},
        {"solver": "lsqr", "shrinkage": "auto"},
        {"solver": "lsqr", "shrinkage": 0.1},
        {"solver": "lsqr", "shrinkage": 0.3},
    ]

    for config in lda_configs:
        estimator = LinearDiscriminantAnalysis(**config)
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_val)[:, 1]
        metrics = _safe_metrics(y_val, probabilities)
        candidates.append(
            ModelSelectionResult(
                name="lda",
                estimator=estimator,
                parameters=config,
                validation_metrics=metrics,
            )
        )

    best = max(candidates, key=lambda candidate: _selection_score(candidate.validation_metrics))
    return best


def _fit_tree_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> ModelSelectionResult:
    candidates: list[ModelSelectionResult] = []

    tree_configs = [
        {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 160, "max_depth": 6, "min_samples_leaf": 1},
        {"n_estimators": 220, "max_depth": 8, "min_samples_leaf": 2},
    ]

    for config in tree_configs:
        estimator = RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
            **config,
        )
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_val)[:, 1]
        metrics = _safe_metrics(y_val, probabilities)
        candidates.append(
            ModelSelectionResult(
                name="random_forest",
                estimator=estimator,
                parameters=config,
                validation_metrics=metrics,
            )
        )

    best = max(candidates, key=lambda candidate: _selection_score(candidate.validation_metrics))
    return best


def _select_feature_columns(frame: pd.DataFrame) -> list[str]:
    z_columns = [column for column in frame.columns if column.startswith("z_")]
    if not z_columns:
        raise TrainingError("No normalized feature columns found. Expected columns prefixed with 'z_'.")
    return sorted(z_columns)


def run_training_pipeline(
    labeled_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    split_column: str = "split",
    label_column: str = "fatigue_label",
    random_state: int = 42,
    scaler_metadata: dict[str, Any] | None = None,
) -> TrainingResult:
    if labeled_frame.empty:
        raise TrainingError("labeled_frame is empty")

    required_columns = [split_column, label_column]
    missing = [column for column in required_columns if column not in labeled_frame.columns]
    if missing:
        raise TrainingError(f"labeled_frame missing required columns: {missing}")

    frame = labeled_frame.copy()
    frame[label_column] = frame[label_column].astype(int)

    if feature_columns is None:
        feature_columns = _select_feature_columns(frame)

    train_frame = frame.loc[frame[split_column] == "train"].copy()
    val_frame = frame.loc[frame[split_column] == "val"].copy()
    test_frame = frame.loc[frame[split_column] == "test"].copy()

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise TrainingError("Train/val/test splits must all be non-empty")

    if train_frame[label_column].nunique() < 2:
        raise TrainingError("Training split must include at least two classes")

    x_train = train_frame[feature_columns].to_numpy(dtype=float)
    y_train = train_frame[label_column].to_numpy(dtype=int)
    x_val = val_frame[feature_columns].to_numpy(dtype=float)
    y_val = val_frame[label_column].to_numpy(dtype=int)
    x_test = test_frame[feature_columns].to_numpy(dtype=float)
    y_test = test_frame[label_column].to_numpy(dtype=int)

    linear_result = _fit_linear_candidates(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=random_state,
    )
    tree_result = _fit_tree_candidates(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=random_state,
    )

    calibrated_linear = _calibrate_estimator(linear_result.estimator, x_train, y_train)
    calibrated_tree = _calibrate_estimator(tree_result.estimator, x_train, y_train)

    linear_val_prob = calibrated_linear.predict_proba(x_val)[:, 1]
    tree_val_prob = calibrated_tree.predict_proba(x_val)[:, 1]

    linear_val_metrics = _safe_metrics(y_val, linear_val_prob)
    tree_val_metrics = _safe_metrics(y_val, tree_val_prob)

    linear_weight = max(0.05, _selection_score(linear_val_metrics))
    tree_weight = max(0.05, _selection_score(tree_val_metrics))
    weight_sum = linear_weight + tree_weight
    fusion_weights = {
        "linear": float(linear_weight / weight_sum),
        "tree": float(tree_weight / weight_sum),
    }

    fused_val_prob = (fusion_weights["linear"] * linear_val_prob) + (fusion_weights["tree"] * tree_val_prob)
    fused_test_prob = (fusion_weights["linear"] * calibrated_linear.predict_proba(x_test)[:, 1]) + (
        fusion_weights["tree"] * calibrated_tree.predict_proba(x_test)[:, 1]
    )

    metrics = {
        "validation": {
            "linear": linear_val_metrics,
            "tree": tree_val_metrics,
            "late_fusion": _safe_metrics(y_val, fused_val_prob),
        },
        "test": {
            "linear": _safe_metrics(y_test, calibrated_linear.predict_proba(x_test)[:, 1]),
            "tree": _safe_metrics(y_test, calibrated_tree.predict_proba(x_test)[:, 1]),
            "late_fusion": _safe_metrics(y_test, fused_test_prob),
        },
        "selected_models": {
            "linear": {
                "name": linear_result.name,
                "parameters": linear_result.parameters,
            },
            "tree": {
                "name": tree_result.name,
                "parameters": tree_result.parameters,
            },
        },
    }

    linear_result.estimator = calibrated_linear
    tree_result.estimator = calibrated_tree

    return TrainingResult(
        linear_result=linear_result,
        tree_result=tree_result,
        fusion_weights=fusion_weights,
        metrics=metrics,
        feature_columns=feature_columns,
        scaler_metadata=scaler_metadata,
    )


def write_training_artifacts(
    result: TrainingResult,
    model_output_dir: str | Path,
    metrics_output_path: str | Path,
    manifest_output_path: str | Path,
) -> None:
    output_dir = Path(model_output_dir)
    metrics_path = Path(metrics_output_path)
    manifest_path = Path(manifest_output_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    linear_model_path = output_dir / "linear_model_calibrated.pkl"
    tree_model_path = output_dir / "tree_model_calibrated.pkl"
    fusion_config_path = output_dir / "late_fusion_config.json"
    feature_columns_path = output_dir / "feature_columns.json"
    scaler_metadata_path = output_dir / "scaler_metadata.json"

    with linear_model_path.open("wb") as linear_handle:
        pickle.dump(result.linear_result.estimator, linear_handle)
    with tree_model_path.open("wb") as tree_handle:
        pickle.dump(result.tree_result.estimator, tree_handle)

    fusion_config = {
        "weights": result.fusion_weights,
        "linear_model": {
            "name": result.linear_result.name,
            "parameters": result.linear_result.parameters,
        },
        "tree_model": {
            "name": result.tree_result.name,
            "parameters": result.tree_result.parameters,
        },
    }
    fusion_config_path.write_text(json.dumps(fusion_config, indent=2), encoding="utf-8")
    feature_columns_path.write_text(json.dumps(result.feature_columns, indent=2), encoding="utf-8")

    if result.scaler_metadata is not None:
        scaler_metadata_path.write_text(json.dumps(result.scaler_metadata, indent=2), encoding="utf-8")

    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    manifest = {
        "linear_model_path": str(linear_model_path),
        "tree_model_path": str(tree_model_path),
        "fusion_config_path": str(fusion_config_path),
        "feature_columns_path": str(feature_columns_path),
        "metrics_path": str(metrics_path),
    }
    if result.scaler_metadata is not None:
        manifest["scaler_metadata_path"] = str(scaler_metadata_path)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")



from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd



class EvaluationError(ValueError):
    """Raised when MVP evaluation or ablation analysis cannot proceed."""


@dataclass
class AblationRun:
    name: str
    feature_columns: list[str]
    imu_gating_applied: bool
    gated_row_count: int
    training_result: TrainingResult
    robustness: dict[str, Any]
    composite_score: float


@dataclass
class EvaluationResult:
    ablation_runs: dict[str, AblationRun]
    selected_variant: str
    selection_rationale: str
    summary: dict[str, Any]


def _metric_or_zero(metric_value: float | None) -> float:
    return float(metric_value) if metric_value is not None else 0.0


def _select_z_feature_columns(frame: pd.DataFrame) -> list[str]:
    z_columns = sorted([column for column in frame.columns if column.startswith("z_")])
    if not z_columns:
        raise EvaluationError("No normalized feature columns found. Expected columns prefixed with 'z_'.")
    return z_columns


def build_ablation_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    z_columns = _select_z_feature_columns(frame)

    eeg_columns = sorted([column for column in z_columns if column.startswith("z_eeg_")])
    optical_columns = sorted([column for column in z_columns if column.startswith("z_optical_")])
    imu_columns = sorted([column for column in z_columns if column.startswith("z_imu_")])
    quality_columns = sorted(
        [
            column
            for column in z_columns
            if column in {"z_confidence_score", "z_downweight_factor", "z_alignment_gap_s"}
        ]
    )

    if not eeg_columns:
        raise EvaluationError("EEG normalized features were not found for EEG-only ablation")
    if not optical_columns:
        raise EvaluationError("Optical normalized features were not found for EEG+optical ablation")

    feature_sets = {
        "eeg_only": eeg_columns,
        "eeg_optical": sorted(eeg_columns + optical_columns),
        "eeg_optical_imu_gated": sorted(eeg_columns + optical_columns + imu_columns + quality_columns),
    }

    if not feature_sets["eeg_optical_imu_gated"]:
        raise EvaluationError("IMU-gated ablation feature set is empty")

    return feature_sets


def build_variant_scaler_metadata(
    base_scaler_metadata: dict[str, Any] | None,
    feature_columns: list[str],
) -> dict[str, Any] | None:
    if base_scaler_metadata is None:
        return None

    statistics = base_scaler_metadata.get("statistics")
    if not isinstance(statistics, dict):
        return None

    raw_feature_names = [column.removeprefix("z_") for column in feature_columns]
    variant_statistics = {
        feature_name: statistics[feature_name]
        for feature_name in raw_feature_names
        if feature_name in statistics
    }

    if not variant_statistics:
        return None

    return {
        "strategy": str(base_scaler_metadata.get("strategy", "train_only_zscore")),
        "split_column": str(base_scaler_metadata.get("split_column", "split")),
        "feature_columns": raw_feature_names,
        "statistics": variant_statistics,
    }


def apply_imu_quality_gate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    motion_score_column: str = "imu_motion_score",
    downweight_column: str = "downweight_factor",
    high_motion_threshold: float = 0.35,
) -> tuple[pd.DataFrame, int]:
    if motion_score_column not in frame.columns:
        raise EvaluationError(f"Missing required IMU motion score column: {motion_score_column}")
    if downweight_column not in frame.columns:
        raise EvaluationError(f"Missing required downweight column: {downweight_column}")

    gated = frame.copy()
    high_motion_mask = gated[motion_score_column].astype(float) >= float(high_motion_threshold)
    downweights = gated[downweight_column].astype(float).clip(0.2, 1.0)

    gated.loc[high_motion_mask, feature_columns] = gated.loc[high_motion_mask, feature_columns].mul(
        downweights.loc[high_motion_mask],
        axis=0,
    )

    return gated, int(high_motion_mask.sum())


def _build_robustness_summary(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    macro_gap = abs(float(validation_metrics["macro_f1"]) - float(test_metrics["macro_f1"]))
    balanced_gap = abs(
        float(validation_metrics["balanced_accuracy"]) - float(test_metrics["balanced_accuracy"])
    )
    auroc_gap = abs(_metric_or_zero(validation_metrics.get("auroc")) - _metric_or_zero(test_metrics.get("auroc")))

    consistency_penalty = macro_gap + (0.5 * balanced_gap) + (0.25 * auroc_gap)
    consistency_score = max(0.0, 1.0 - consistency_penalty)

    return {
        "macro_f1_gap": float(macro_gap),
        "balanced_accuracy_gap": float(balanced_gap),
        "auroc_gap": float(auroc_gap),
        "consistency_penalty": float(consistency_penalty),
        "consistency_score": float(consistency_score),
    }


def _build_composite_score(test_metrics: dict[str, Any], robustness: dict[str, Any]) -> float:
    perf_score = (
        0.5 * float(test_metrics["macro_f1"])
        + 0.3 * float(test_metrics["balanced_accuracy"])
        + 0.2 * _metric_or_zero(test_metrics.get("auroc"))
    )
    return float(perf_score - (0.2 * float(robustness["consistency_penalty"])))


def run_evaluation_pipeline(
    labeled_frame: pd.DataFrame,
    random_state: int = 42,
    base_scaler_metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    if labeled_frame.empty:
        raise EvaluationError("labeled_frame is empty")
    if "split" not in labeled_frame.columns:
        raise EvaluationError("labeled_frame is missing required 'split' column")
    if "fatigue_label" not in labeled_frame.columns:
        raise EvaluationError("labeled_frame is missing required 'fatigue_label' column")

    feature_sets = build_ablation_feature_sets(labeled_frame)

    ablation_runs: dict[str, AblationRun] = {}

    for variant_name, feature_columns in feature_sets.items():
        variant_frame = labeled_frame.copy()
        gated_row_count = 0

        if variant_name == "eeg_optical_imu_gated":
            variant_frame, gated_row_count = apply_imu_quality_gate(
                frame=variant_frame,
                feature_columns=feature_columns,
            )

        scaler_metadata = build_variant_scaler_metadata(
            base_scaler_metadata=base_scaler_metadata,
            feature_columns=feature_columns,
        )

        training_result = run_training_pipeline(
            labeled_frame=variant_frame,
            feature_columns=feature_columns,
            random_state=random_state,
            scaler_metadata=scaler_metadata,
        )

        validation_metrics = training_result.metrics["validation"]["late_fusion"]
        test_metrics = training_result.metrics["test"]["late_fusion"]
        robustness = _build_robustness_summary(validation_metrics=validation_metrics, test_metrics=test_metrics)
        composite_score = _build_composite_score(test_metrics=test_metrics, robustness=robustness)

        ablation_runs[variant_name] = AblationRun(
            name=variant_name,
            feature_columns=feature_columns,
            imu_gating_applied=variant_name == "eeg_optical_imu_gated",
            gated_row_count=gated_row_count,
            training_result=training_result,
            robustness=robustness,
            composite_score=composite_score,
        )

    selected_variant = max(
        ablation_runs,
        key=lambda variant: (
            ablation_runs[variant].composite_score,
            float(ablation_runs[variant].training_result.metrics["test"]["late_fusion"]["macro_f1"]),
        ),
    )
    selected_run = ablation_runs[selected_variant]
    selected_test_metrics = selected_run.training_result.metrics["test"]["late_fusion"]

    selection_rationale = (
        "Selected variant '{variant}' because it achieved the best composite score ({score:.4f}) "
        "while maintaining consistency (penalty={penalty:.4f}) and robust held-out performance "
        "(macro_f1={macro:.4f}, balanced_accuracy={balanced:.4f}, auroc={auroc})."
    ).format(
        variant=selected_variant,
        score=selected_run.composite_score,
        penalty=float(selected_run.robustness["consistency_penalty"]),
        macro=float(selected_test_metrics["macro_f1"]),
        balanced=float(selected_test_metrics["balanced_accuracy"]),
        auroc=selected_test_metrics.get("auroc"),
    )

    summary = {
        "ablation_count": int(len(ablation_runs)),
        "selected_variant": selected_variant,
        "selection_rationale": selection_rationale,
        "variant_scores": {
            name: {
                "composite_score": float(run.composite_score),
                "test_late_fusion": run.training_result.metrics["test"]["late_fusion"],
                "consistency_penalty": float(run.robustness["consistency_penalty"]),
            }
            for name, run in ablation_runs.items()
        },
    }

    return EvaluationResult(
        ablation_runs=ablation_runs,
        selected_variant=selected_variant,
        selection_rationale=selection_rationale,
        summary=summary,
    )


def write_evaluation_artifacts(
    result: EvaluationResult,
    model_output_root: str | Path,
    report_output_root: str | Path,
    ablation_report_output_path: str | Path,
    summary_output_path: str | Path,
    selection_output_path: str | Path,
    narrative_output_path: str | Path,
) -> None:
    model_root = Path(model_output_root)
    report_root = Path(report_output_root)
    ablation_report_path = Path(ablation_report_output_path)
    summary_path = Path(summary_output_path)
    selection_path = Path(selection_output_path)
    narrative_path = Path(narrative_output_path)

    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    ablation_report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    narrative_path.parent.mkdir(parents=True, exist_ok=True)

    ablation_payload: dict[str, Any] = {
        "ablations": {},
    }

    for variant_name, ablation_run in result.ablation_runs.items():
        variant_model_dir = model_root / "ablations" / variant_name
        variant_report_dir = report_root / "ablations"
        variant_report_dir.mkdir(parents=True, exist_ok=True)

        variant_metrics_path = variant_report_dir / f"{variant_name}_training_metrics.json"
        variant_manifest_path = variant_report_dir / f"{variant_name}_training_manifest.json"

        write_training_artifacts(
            result=ablation_run.training_result,
            model_output_dir=variant_model_dir,
            metrics_output_path=variant_metrics_path,
            manifest_output_path=variant_manifest_path,
        )

        ablation_payload["ablations"][variant_name] = {
            "feature_columns": ablation_run.feature_columns,
            "feature_column_count": int(len(ablation_run.feature_columns)),
            "imu_gating_applied": bool(ablation_run.imu_gating_applied),
            "gated_row_count": int(ablation_run.gated_row_count),
            "robustness": ablation_run.robustness,
            "composite_score": float(ablation_run.composite_score),
            "metrics": ablation_run.training_result.metrics,
            "training_manifest_path": str(variant_manifest_path),
        }

    selected_variant_payload = ablation_payload["ablations"][result.selected_variant]

    selection_payload = {
        "selected_variant": result.selected_variant,
        "selected_training_manifest_path": selected_variant_payload["training_manifest_path"],
        "selected_models": selected_variant_payload["metrics"]["selected_models"],
        "selected_test_metrics": selected_variant_payload["metrics"]["test"]["late_fusion"],
        "selected_composite_score": selected_variant_payload["composite_score"],
        "selection_rationale": result.selection_rationale,
    }

    ranked_variants = sorted(
        result.ablation_runs.values(),
        key=lambda run: run.composite_score,
        reverse=True,
    )

    narrative_lines = [
        "# MVP Evaluation Summary",
        "",
        "## Production Selection",
        "",
        f"Selected variant: {result.selected_variant}",
        f"Reason: {result.selection_rationale}",
        "",
        "## Variant Ranking (Best to Worst)",
        "",
    ]
    for rank, variant_run in enumerate(ranked_variants, start=1):
        metrics = variant_run.training_result.metrics["test"]["late_fusion"]
        narrative_lines.append(
            (
                f"{rank}. {variant_run.name}: composite={variant_run.composite_score:.4f}, "
                f"macro_f1={float(metrics['macro_f1']):.4f}, "
                f"balanced_accuracy={float(metrics['balanced_accuracy']):.4f}, "
                f"auroc={metrics.get('auroc')}"
            )
        )

    ablation_report_path.write_text(json.dumps(ablation_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    selection_path.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    narrative_path.write_text("\n".join(narrative_lines) + "\n", encoding="utf-8")



import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import DEFAULT_CONTRACT_PATH, load_data_contract


class SchemaValidationError(ValueError):
    """Raised when session data does not satisfy the session data contract."""


@dataclass
class ValidationSummary:
    eeg_rows: int
    optical_rows: int
    imu_rows: int
    metadata_rows: int
    max_missing_ratio: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eeg_rows": self.eeg_rows,
            "optical_rows": self.optical_rows,
            "imu_rows": self.imu_rows,
            "metadata_rows": self.metadata_rows,
            "max_missing_ratio": self.max_missing_ratio,
        }


def _fail_or_collect(message: str, errors: list[str], fail_fast: bool) -> None:
    if fail_fast:
        raise SchemaValidationError(message)
    errors.append(message)


def _validate_required_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    modality_name: str,
    errors: list[str],
    fail_fast: bool,
) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        _fail_or_collect(
            f"{modality_name}: missing required columns: {missing_columns}",
            errors,
            fail_fast,
        )


def _validate_timestamps(
    frame: pd.DataFrame,
    timestamp_column: str,
    modality_name: str,
    errors: list[str],
    fail_fast: bool,
) -> None:
    if timestamp_column not in frame.columns:
        return

    if frame[timestamp_column].isna().any():
        _fail_or_collect(
            f"{modality_name}: null values found in {timestamp_column}",
            errors,
            fail_fast,
        )
        return

    if frame[timestamp_column].duplicated().any():
        _fail_or_collect(
            f"{modality_name}: duplicate values found in {timestamp_column}",
            errors,
            fail_fast,
        )

    if not frame[timestamp_column].is_monotonic_increasing:
        _fail_or_collect(
            f"{modality_name}: {timestamp_column} must be monotonic increasing",
            errors,
            fail_fast,
        )


def _validate_missing_ratio(
    frame: pd.DataFrame,
    signal_columns: list[str],
    threshold: float,
    modality_name: str,
    errors: list[str],
    fail_fast: bool,
) -> float:
    if not signal_columns:
        return 0.0

    max_ratio = float(frame[signal_columns].isna().mean().max())
    if max_ratio > threshold:
        _fail_or_collect(
            (
                f"{modality_name}: max missing ratio {max_ratio:.4f} exceeds "
                f"threshold {threshold:.4f}"
            ),
            errors,
            fail_fast,
        )

    return max_ratio


def _validate_modality_file(
    modality_name: str,
    file_path: str | Path,
    modality_contract: dict[str, Any],
    errors: list[str],
    fail_fast: bool,
) -> tuple[pd.DataFrame, float]:
    path = Path(file_path)
    if not path.exists():
        _fail_or_collect(f"{modality_name}: file not found: {path}", errors, fail_fast)
        return pd.DataFrame(), 0.0

    frame = pd.read_csv(path)

    required_columns = modality_contract["required_columns"]
    _validate_required_columns(frame, required_columns, modality_name, errors, fail_fast)

    timestamp_column = modality_contract["timestamp_column"]
    _validate_timestamps(frame, timestamp_column, modality_name, errors, fail_fast)

    signal_columns = [
        column
        for column in required_columns
        if column not in {"session_id", timestamp_column}
    ]
    max_ratio = _validate_missing_ratio(
        frame=frame,
        signal_columns=signal_columns,
        threshold=float(modality_contract["max_missing_ratio"]),
        modality_name=modality_name,
        errors=errors,
        fail_fast=fail_fast,
    )

    return frame, max_ratio


def _validate_metadata(
    metadata_frame: pd.DataFrame,
    contract: dict[str, Any],
    errors: list[str],
    fail_fast: bool,
) -> None:
    metadata_contract = contract["metadata"]
    required_columns = metadata_contract["required_fields"]

    _validate_required_columns(
        frame=metadata_frame,
        required_columns=required_columns,
        modality_name="metadata",
        errors=errors,
        fail_fast=fail_fast,
    )

    if metadata_frame.empty:
        _fail_or_collect("metadata: file must contain at least one row", errors, fail_fast)
        return

    allowed_task_types = set(metadata_contract["allowed_task_types"])
    task_value = str(metadata_frame.iloc[0]["task_type"])
    if task_value not in allowed_task_types:
        _fail_or_collect(
            f"metadata: unsupported task_type '{task_value}'",
            errors,
            fail_fast,
        )

    start_ts = pd.to_datetime(metadata_frame.iloc[0]["start_timestamp_utc"], utc=True, errors="coerce")
    end_ts = pd.to_datetime(metadata_frame.iloc[0]["end_timestamp_utc"], utc=True, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        _fail_or_collect(
            "metadata: start_timestamp_utc and end_timestamp_utc must be valid UTC timestamps",
            errors,
            fail_fast,
        )
    elif start_ts >= end_ts:
        _fail_or_collect(
            "metadata: start_timestamp_utc must be earlier than end_timestamp_utc",
            errors,
            fail_fast,
        )

    fatigue_score = float(metadata_frame.iloc[0]["subjective_fatigue_score"])
    min_score = float(metadata_contract["fatigue_score_range"][0])
    max_score = float(metadata_contract["fatigue_score_range"][1])
    if not (min_score <= fatigue_score <= max_score):
        _fail_or_collect(
            (
                f"metadata: subjective_fatigue_score {fatigue_score} is out of "
                f"range [{min_score}, {max_score}]"
            ),
            errors,
            fail_fast,
        )


def validate_session_files(
    eeg_path: str | Path,
    optical_path: str | Path,
    imu_path: str | Path,
    metadata_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    fail_fast: bool = True,
) -> ValidationSummary:
    contract = load_data_contract(contract_path)
    errors: list[str] = []

    eeg_frame, eeg_missing = _validate_modality_file(
        modality_name="eeg",
        file_path=eeg_path,
        modality_contract=contract["modalities"]["eeg"],
        errors=errors,
        fail_fast=fail_fast,
    )
    optical_frame, optical_missing = _validate_modality_file(
        modality_name="optical",
        file_path=optical_path,
        modality_contract=contract["modalities"]["optical"],
        errors=errors,
        fail_fast=fail_fast,
    )
    imu_frame, imu_missing = _validate_modality_file(
        modality_name="imu",
        file_path=imu_path,
        modality_contract=contract["modalities"]["imu"],
        errors=errors,
        fail_fast=fail_fast,
    )

    metadata_frame = pd.read_csv(metadata_path)
    _validate_metadata(metadata_frame, contract, errors, fail_fast)

    if errors:
        raise SchemaValidationError("; ".join(errors))

    return ValidationSummary(
        eeg_rows=len(eeg_frame),
        optical_rows=len(optical_frame),
        imu_rows=len(imu_frame),
        metadata_rows=len(metadata_frame),
        max_missing_ratio={
            "eeg": eeg_missing,
            "optical": optical_missing,
            "imu": imu_missing,
        },
    )


def write_validation_summary(summary: ValidationSummary, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2)
    return output

