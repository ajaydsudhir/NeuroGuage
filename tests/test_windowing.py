from __future__ import annotations

from pathlib import Path

from neuroguage.pipeline import synchronize_modalities
from neuroguage.pipeline import build_windowing_layers


FIXTURES = Path("tests/fixtures/day2")


def _synchronized_frame():
    result = synchronize_modalities(
        eeg_path=FIXTURES / "eeg_raw.csv",
        optical_path=FIXTURES / "optical_raw.csv",
        imu_path=FIXTURES / "imu_raw.csv",
        contract_path="configs/data_contract.json",
        tolerance_s=0.25,
    )
    return result.synchronized_frame


def test_windowing_produces_fast_and_slow_layers() -> None:
    frame = _synchronized_frame()
    result = build_windowing_layers(frame)

    assert len(result.eeg_windows) > len(result.optical_windows)
    assert "absolute_timeline_start_idx" in result.eeg_windows.columns
    assert "absolute_timeline_end_idx" in result.optical_windows.columns


def test_windowing_preserves_session_start_boundary() -> None:
    frame = _synchronized_frame()
    result = build_windowing_layers(frame)

    first_window = result.eeg_windows.iloc[0]
    assert int(first_window["start_index"]) == 0


def test_windowing_covers_session_end_boundary() -> None:
    frame = _synchronized_frame()
    result = build_windowing_layers(frame)

    final_index = len(frame) - 1
    assert final_index in set(result.eeg_windows["end_index"].astype(int).tolist())


def test_windowing_detects_missing_packets_in_some_windows() -> None:
    frame = _synchronized_frame()
    result = build_windowing_layers(frame)

    assert int(result.eeg_windows["missing_packet_count"].max()) >= 1
