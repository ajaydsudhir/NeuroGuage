from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuroguage.pipeline import build_leakage_report, run_feature_builder_pipeline


FIXTURES = Path("tests/fixtures/day3")


def _load_feature_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    eeg = pd.read_csv(FIXTURES / "eeg_window_features_sessions.csv")
    optical_imu = pd.read_csv(FIXTURES / "optical_imu_window_features_sessions.csv")
    return eeg, optical_imu


def test_unified_feature_builder_joins_all_eeg_windows() -> None:
    eeg, optical_imu = _load_feature_sources()
    result = run_feature_builder_pipeline(
        eeg_feature_frame=eeg,
        optical_imu_feature_frame=optical_imu,
        split_seed=42,
    )

    assert len(result.unified_frame) == len(eeg)
    assert "split_compatibility_marker" in result.unified_frame.columns
    assert result.summary["sessions"] == 4


def test_train_only_normalization_centers_train_partition() -> None:
    eeg, optical_imu = _load_feature_sources()
    result = run_feature_builder_pipeline(
        eeg_feature_frame=eeg,
        optical_imu_feature_frame=optical_imu,
        split_seed=42,
    )

    train = result.normalized_frame.loc[result.normalized_frame["split"] == "train"]
    z_columns = [column for column in train.columns if column.startswith("z_")]

    assert z_columns
    for column in z_columns:
        assert abs(float(train[column].mean())) < 1e-8


def test_leakage_report_catches_session_overlap() -> None:
    eeg, optical_imu = _load_feature_sources()
    result = run_feature_builder_pipeline(
        eeg_feature_frame=eeg,
        optical_imu_feature_frame=optical_imu,
        split_seed=42,
    )

    manipulated = result.normalized_frame.copy()
    session_id = str(manipulated["session_id"].iloc[0])
    session_rows = manipulated.index[manipulated["session_id"] == session_id].tolist()
    original_split = str(manipulated.loc[session_rows[0], "split"])
    manipulated.loc[session_rows[0], "split"] = "test" if original_split != "test" else "train"

    report = build_leakage_report(
        frame=manipulated,
        feature_columns=result.feature_columns,
    )

    assert report["session_overlap"]["pass"] is False
    assert report["overall_pass"] is False
