from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FeatureBuilderError(ValueError):
    """Raised when Day 3 unified feature builder requirements are not satisfied."""


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
