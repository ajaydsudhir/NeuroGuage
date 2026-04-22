from __future__ import annotations

from pathlib import Path

from neuroguage.eeg_features import extract_eeg_window_features
from neuroguage.ingestion import synchronize_modalities
from neuroguage.windowing import build_windowing_layers


FIXTURES = Path("tests/fixtures/day2")


def _synchronized_and_windows():
    sync_result = synchronize_modalities(
        eeg_path=FIXTURES / "eeg_raw.csv",
        optical_path=FIXTURES / "optical_raw.csv",
        imu_path=FIXTURES / "imu_raw.csv",
        contract_path="configs/data_contract.json",
        tolerance_s=0.25,
    )
    windows = build_windowing_layers(sync_result.synchronized_frame)
    return sync_result.synchronized_frame, windows.eeg_windows


def test_eeg_feature_extraction_runs_for_each_window() -> None:
    synchronized, eeg_windows = _synchronized_and_windows()
    result = extract_eeg_window_features(
        synchronized_frame=synchronized,
        eeg_windows=eeg_windows,
    )

    assert len(result.feature_frame) == len(eeg_windows)


def test_eeg_feature_columns_exist_and_are_finite() -> None:
    synchronized, eeg_windows = _synchronized_and_windows()
    result = extract_eeg_window_features(
        synchronized_frame=synchronized,
        eeg_windows=eeg_windows,
    )

    for column in [
        "eeg_rel_theta",
        "eeg_rel_alpha",
        "eeg_rel_beta",
        "eeg_theta_alpha_ratio",
        "eeg_beta_alpha_ratio",
    ]:
        assert column in result.feature_frame.columns
        assert result.feature_frame[column].notna().all()


def test_eeg_relative_band_features_are_bounded() -> None:
    synchronized, eeg_windows = _synchronized_and_windows()
    result = extract_eeg_window_features(
        synchronized_frame=synchronized,
        eeg_windows=eeg_windows,
    )

    assert (result.feature_frame["eeg_rel_theta"] >= 0.0).all()
    assert (result.feature_frame["eeg_rel_alpha"] >= 0.0).all()
    assert (result.feature_frame["eeg_rel_beta"] >= 0.0).all()


def test_eeg_quality_flags_and_scores_are_present() -> None:
    synchronized, eeg_windows = _synchronized_and_windows()
    result = extract_eeg_window_features(
        synchronized_frame=synchronized,
        eeg_windows=eeg_windows,
    )

    assert "eeg_artifact_flag" in result.feature_frame.columns
    assert "eeg_quality_score" in result.feature_frame.columns
    assert (result.feature_frame["eeg_quality_score"] >= 0.0).all()
