from __future__ import annotations

from pathlib import Path

from neuroguage.ingestion import synchronize_modalities
from neuroguage.optical_imu_features import extract_optical_imu_window_features
from neuroguage.windowing import build_windowing_layers


FIXTURES = Path("tests/fixtures/day2")


def _synchronized_and_optical_windows():
    sync_result = synchronize_modalities(
        eeg_path=FIXTURES / "eeg_raw.csv",
        optical_path=FIXTURES / "optical_raw.csv",
        imu_path=FIXTURES / "imu_raw.csv",
        contract_path="configs/data_contract.json",
        tolerance_s=0.25,
    )
    windows = build_windowing_layers(sync_result.synchronized_frame)
    return sync_result.synchronized_frame, windows.optical_windows


def test_optical_imu_feature_rows_match_optical_windows() -> None:
    synchronized, optical_windows = _synchronized_and_optical_windows()
    result = extract_optical_imu_window_features(
        synchronized_frame=synchronized,
        optical_windows=optical_windows,
    )

    assert len(result.feature_frame) == len(optical_windows)


def test_optical_imu_expected_columns_exist() -> None:
    synchronized, optical_windows = _synchronized_and_optical_windows()
    result = extract_optical_imu_window_features(
        synchronized_frame=synchronized,
        optical_windows=optical_windows,
    )

    for column in [
        "optical_norm_mean",
        "optical_trend_slope",
        "optical_mean_change",
        "imu_motion_score",
        "confidence_score",
        "downweight_factor",
        "quality_flag",
    ]:
        assert column in result.feature_frame.columns


def test_confidence_and_motion_scores_are_bounded() -> None:
    synchronized, optical_windows = _synchronized_and_optical_windows()
    result = extract_optical_imu_window_features(
        synchronized_frame=synchronized,
        optical_windows=optical_windows,
    )

    assert (result.feature_frame["imu_motion_score"] >= 0.0).all()
    assert (result.feature_frame["imu_motion_score"] <= 1.0).all()
    assert (result.feature_frame["confidence_score"] >= 0.0).all()
    assert (result.feature_frame["confidence_score"] <= 1.0).all()


def test_downweight_factor_matches_confidence_score() -> None:
    synchronized, optical_windows = _synchronized_and_optical_windows()
    result = extract_optical_imu_window_features(
        synchronized_frame=synchronized,
        optical_windows=optical_windows,
    )

    assert (result.feature_frame["downweight_factor"] == result.feature_frame["confidence_score"]).all()
