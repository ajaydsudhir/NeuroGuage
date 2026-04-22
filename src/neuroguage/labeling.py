from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LabelingError(ValueError):
    """Raised when Day 3 labeling or split-generation requirements are not satisfied."""


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
