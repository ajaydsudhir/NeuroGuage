from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neuroguage.pipeline import SchemaValidationError, validate_session_files


FIXTURES = Path("tests/fixtures/day1")


def test_validate_session_files_accepts_valid_inputs() -> None:
    summary = validate_session_files(
        eeg_path=FIXTURES / "eeg_valid.csv",
        optical_path=FIXTURES / "optical_valid.csv",
        imu_path=FIXTURES / "imu_valid.csv",
        metadata_path=FIXTURES / "metadata_valid.csv",
        contract_path="configs/data_contract.json",
        fail_fast=True,
    )

    assert summary.eeg_rows == 3
    assert summary.optical_rows == 3
    assert summary.imu_rows == 3
    assert summary.metadata_rows == 1


def test_validate_session_files_rejects_malformed_inputs(tmp_path: Path) -> None:
    bad_eeg_path = tmp_path / "eeg_bad.csv"
    eeg_frame = pd.read_csv(FIXTURES / "eeg_valid.csv")
    eeg_frame = eeg_frame.drop(columns=["eeg_tp10"])
    eeg_frame.to_csv(bad_eeg_path, index=False)

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_session_files(
            eeg_path=bad_eeg_path,
            optical_path=FIXTURES / "optical_valid.csv",
            imu_path=FIXTURES / "imu_valid.csv",
            metadata_path=FIXTURES / "metadata_valid.csv",
            contract_path="configs/data_contract.json",
            fail_fast=True,
        )

    assert "missing required columns" in str(exc_info.value)
