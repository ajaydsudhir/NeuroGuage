from __future__ import annotations

from pathlib import Path

from neuroguage.ingestion import (
    ingest_modalities,
    normalize_modality_timestamps,
    synchronize_modalities,
)


FIXTURES = Path("tests/fixtures/day2")


def test_ingest_modalities_loads_all_sources() -> None:
    frames = ingest_modalities(
        eeg_path=FIXTURES / "eeg_raw.csv",
        optical_path=FIXTURES / "optical_raw.csv",
        imu_path=FIXTURES / "imu_raw.csv",
    )

    assert set(frames.keys()) == {"eeg", "optical", "imu"}
    assert len(frames["eeg"]) == 5
    assert len(frames["optical"]) == 5
    assert len(frames["imu"]) == 5


def test_normalize_modality_timestamps_uses_common_reference() -> None:
    frames = ingest_modalities(
        eeg_path=FIXTURES / "eeg_raw.csv",
        optical_path=FIXTURES / "optical_raw.csv",
        imu_path=FIXTURES / "imu_raw.csv",
    )
    normalized, reference = normalize_modality_timestamps(frames)

    assert reference.isoformat() == "2026-04-22T09:00:00+00:00"
    assert normalized["eeg"]["time_s"].iloc[0] == 0.0
    assert round(float(normalized["optical"]["time_s"].iloc[0]), 3) == 0.2
    assert round(float(normalized["imu"]["time_s"].iloc[0]), 3) == 0.05


def test_synchronize_modalities_emits_dropout_diagnostics() -> None:
    result = synchronize_modalities(
        eeg_path=FIXTURES / "eeg_raw.csv",
        optical_path=FIXTURES / "optical_raw.csv",
        imu_path=FIXTURES / "imu_raw.csv",
        contract_path="configs/data_contract.json",
        tolerance_s=0.25,
    )

    eeg_diag = result.diagnostics.modalities["eeg"]
    assert eeg_diag.dropout_count >= 1
    assert result.validation_summary["timeline_rows"] > 0
    assert "modality_match_ratio" in result.validation_summary


def test_synchronized_output_contains_prefixed_modalities() -> None:
    result = synchronize_modalities(
        eeg_path=FIXTURES / "eeg_raw.csv",
        optical_path=FIXTURES / "optical_raw.csv",
        imu_path=FIXTURES / "imu_raw.csv",
        contract_path="configs/data_contract.json",
        tolerance_s=0.25,
    )

    frame = result.synchronized_frame
    assert "eeg_eeg_tp9" in frame.columns
    assert "optical_optics_lo_nir" in frame.columns
    assert "imu_acc_x" in frame.columns
    assert "timestamp_utc" in frame.columns
