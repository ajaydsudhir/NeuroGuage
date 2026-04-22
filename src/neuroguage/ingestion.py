from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_contract import DEFAULT_CONTRACT_PATH, load_data_contract


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
