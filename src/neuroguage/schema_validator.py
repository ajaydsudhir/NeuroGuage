from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data_contract import DEFAULT_CONTRACT_PATH, load_data_contract


class SchemaValidationError(ValueError):
    """Raised when session data does not satisfy the Day 1 data contract."""


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
