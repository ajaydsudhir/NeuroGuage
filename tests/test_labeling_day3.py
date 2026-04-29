from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuroguage.pipeline import run_feature_builder_pipeline
from neuroguage.pipeline import run_labeling_pipeline


FIXTURES = Path("tests/fixtures/day3")


def _normalized_feature_frame() -> pd.DataFrame:
    eeg = pd.read_csv(FIXTURES / "eeg_window_features_sessions.csv")
    optical_imu = pd.read_csv(FIXTURES / "optical_imu_window_features_sessions.csv")
    result = run_feature_builder_pipeline(
        eeg_feature_frame=eeg,
        optical_imu_feature_frame=optical_imu,
        split_seed=42,
    )
    return result.normalized_frame


def _annotations() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "session_annotations.csv")


def test_weak_labels_are_generated_and_bounded() -> None:
    result = run_labeling_pipeline(
        feature_frame=_normalized_feature_frame(),
        annotation_frame=_annotations(),
        split_seed=42,
    )

    frame = result.labeled_frame
    assert "weak_label_score" in frame.columns
    assert "fatigue_label" in frame.columns
    assert (frame["weak_label_score"] >= 0.0).all()
    assert (frame["weak_label_score"] <= 1.0).all()
    assert set(frame["fatigue_label"].unique().tolist()) <= {0, 1}


def test_label_smoothing_is_optional_and_changes_scores() -> None:
    smoothed = run_labeling_pipeline(
        feature_frame=_normalized_feature_frame(),
        annotation_frame=_annotations(),
        split_seed=42,
        enable_smoothing=True,
        smoothing_window=3,
    )
    raw = run_labeling_pipeline(
        feature_frame=_normalized_feature_frame(),
        annotation_frame=_annotations(),
        split_seed=42,
        enable_smoothing=False,
    )

    merged = smoothed.labeled_frame.merge(
        raw.labeled_frame[["session_id", "window_id", "weak_label_score_smoothed"]],
        on=["session_id", "window_id"],
        suffixes=("_smoothed", "_raw"),
    )

    assert (merged["weak_label_score_smoothed_smoothed"] != merged["weak_label_score_smoothed_raw"]).any()


def test_split_manifest_is_deterministic_and_isolated() -> None:
    first = run_labeling_pipeline(
        feature_frame=_normalized_feature_frame(),
        annotation_frame=_annotations(),
        split_seed=17,
    )
    second = run_labeling_pipeline(
        feature_frame=_normalized_feature_frame(),
        annotation_frame=_annotations(),
        split_seed=17,
    )

    assert first.split_manifest["session_split_map"] == second.split_manifest["session_split_map"]
    assert first.summary["split_validation"]["overall_pass"] is True
