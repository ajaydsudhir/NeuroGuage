from __future__ import annotations

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
