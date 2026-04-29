"""Tests for :mod:`neuroguage.muse_stream`.

Covers:
* Synthetic source produces contract-shaped frames at the right cadence.
* Raw-recording replay source wires OpenMuse.decode_rawdata into contract CSVs.
* ``build_data_source`` routes correctly and validates inputs.
* ``decode_raw_session_to_contract`` writes eeg/optical/imu CSVs + metadata.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroguage.muse_stream import (  # noqa: E402
    EEG_OUTPUT_COLUMNS,
    IMU_OUTPUT_COLUMNS,
    LSLMuseSource,
    OPTICAL_OUTPUT_COLUMNS,
    RawRecordingReplaySource,
    SyntheticMuseSource,
    build_data_source,
    decode_raw_session_to_contract,
)


class TestSyntheticMuseSource:
    def test_columns_match_contract(self) -> None:
        source = SyntheticMuseSource(real_time_factor=50.0)
        source.start()
        try:
            time.sleep(0.1)
            frames = source.read_frames()
        finally:
            source.stop()

        assert list(frames.eeg.columns) == EEG_OUTPUT_COLUMNS
        assert list(frames.optical.columns) == OPTICAL_OUTPUT_COLUMNS
        assert list(frames.imu.columns) == IMU_OUTPUT_COLUMNS

    def test_produces_samples_at_roughly_native_rate(self) -> None:
        source = SyntheticMuseSource(real_time_factor=1.0)
        source.start()
        try:
            time.sleep(0.5)
            frames = source.read_frames()
        finally:
            source.stop()

        # EEG at 256Hz over ~0.5s => at least ~80 samples, far less than max 512.
        assert 60 <= len(frames.eeg) <= 350
        assert 10 <= len(frames.optical) <= 90
        assert 10 <= len(frames.imu) <= 80
        assert frames.eeg["session_id"].nunique() == 1

    def test_health_reports_started_state(self) -> None:
        source = SyntheticMuseSource()
        source.start()
        try:
            assert source.health()["started"] is True
            assert source.health()["source"] == "synthetic"
        finally:
            source.stop()
        assert source.health()["started"] is False


class TestBuildDataSource:
    def test_routes_to_synthetic(self) -> None:
        source = build_data_source(source_kind="synthetic", fatigue_trend=0.7)
        assert isinstance(source, SyntheticMuseSource)
        assert source.fatigue_trend == pytest.approx(0.7)

    def test_muse_requires_address(self) -> None:
        with pytest.raises(ValueError):
            build_data_source(source_kind="muse")

    def test_replay_requires_raw_file(self) -> None:
        with pytest.raises(ValueError):
            build_data_source(source_kind="replay")

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_data_source(source_kind="bogus")

    def test_muse_source_requires_full_link_by_default(self) -> None:
        source = build_data_source(
            source_kind="muse",
            device_address="00:11:22:33:44:55",
        )
        assert isinstance(source, LSLMuseSource)
        assert source.require_full_link is True


def _fake_decoded_session(n_seconds: float = 2.0) -> dict[str, pd.DataFrame]:
    eeg_n = int(n_seconds * 256)
    opt_n = int(n_seconds * 64)
    imu_n = int(n_seconds * 52)

    def _tdf(n: int, freq: float, channels: list[str], base: float = 0.0) -> pd.DataFrame:
        frame = pd.DataFrame({ch: base + 0.01 * ch.count("A") for ch in channels}, index=range(n))
        frame["time"] = [i / freq for i in range(n)]
        return frame

    return {
        "EEG": _tdf(eeg_n, 256.0, ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10",
                                   "AUX_1", "AUX_2", "AUX_3", "AUX_4"]),
        "OPTICS": _tdf(opt_n, 64.0, list({
            "OPTICS_LO_NIR", "OPTICS_RO_NIR", "OPTICS_LO_IR", "OPTICS_RO_IR",
            "OPTICS_LO_RED", "OPTICS_RO_RED", "OPTICS_LO_AMB", "OPTICS_RO_AMB",
            "OPTICS_LI_NIR", "OPTICS_RI_NIR", "OPTICS_LI_IR", "OPTICS_RI_IR",
            "OPTICS_LI_RED", "OPTICS_RI_RED", "OPTICS_LI_AMB", "OPTICS_RI_AMB",
        })),
        "ACCGYRO": _tdf(imu_n, 52.0, ["ACC_X", "ACC_Y", "ACC_Z",
                                      "GYRO_X", "GYRO_Y", "GYRO_Z"]),
    }


@pytest.fixture
def fake_raw_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Write a dummy raw file. Contents don't matter because decode_rawdata is patched.
    raw = tmp_path / "session.txt"
    raw.write_text("fake\n", encoding="utf-8")

    import OpenMuse
    monkeypatch.setattr(OpenMuse, "decode_rawdata", lambda _messages: _fake_decoded_session())
    return raw


class TestRawRecordingReplaySource:
    def test_decodes_all_three_modalities(self, fake_raw_file: Path) -> None:
        source = RawRecordingReplaySource(raw_file=fake_raw_file, real_time_factor=100.0)
        source.start()
        try:
            time.sleep(0.1)
            frames = source.read_frames()
        finally:
            source.stop()

        assert list(frames.eeg.columns) == EEG_OUTPUT_COLUMNS
        assert list(frames.optical.columns) == OPTICAL_OUTPUT_COLUMNS
        assert list(frames.imu.columns) == IMU_OUTPUT_COLUMNS
        assert len(frames.eeg) > 0
        assert len(frames.optical) > 0
        assert len(frames.imu) > 0

    def test_finished_flag_after_full_playback(self, fake_raw_file: Path) -> None:
        source = RawRecordingReplaySource(raw_file=fake_raw_file, real_time_factor=1000.0)
        source.start()
        try:
            time.sleep(0.3)
            # Drain in a loop until no more data.
            for _ in range(5):
                source.read_frames()
                time.sleep(0.05)
            assert source.finished is True
        finally:
            source.stop()


class TestLSLMuseSource:
    def test_resolve_streams_raises_when_required_modalities_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_lsl = types.SimpleNamespace(
            StreamInlet=object,
            resolve_streams=lambda timeout=0.0: [],
        )
        monkeypatch.setitem(sys.modules, "mne_lsl.lsl", fake_lsl)
        source = LSLMuseSource(
            device_address="00:11:22:33:44:55",
            manage_bridge=False,
            resolve_timeout_s=0.0,
            resolve_poll_interval_s=0.0,
            require_full_link=True,
        )
        with pytest.raises(RuntimeError, match="Muse link not ready"):
            source._resolve_streams()


class TestDecodeRawSessionToContract:
    def test_writes_contract_csvs_and_metadata(self, tmp_path: Path, fake_raw_file: Path) -> None:
        out_dir = tmp_path / "decoded"
        paths = decode_raw_session_to_contract(
            raw_file=fake_raw_file,
            session_id="sub01_test",
            output_dir=out_dir,
            task_type="baseline_rest",
            subjective_fatigue_score=3.5,
        )

        assert paths["eeg"].exists()
        assert paths["optical"].exists()
        assert paths["imu"].exists()
        assert paths["metadata"].exists()

        eeg = pd.read_csv(paths["eeg"])
        assert set(["session_id", "timestamp_utc", "eeg_tp9", "eeg_af7",
                    "eeg_af8", "eeg_tp10", "eeg_aux"]).issubset(eeg.columns)
        assert (eeg["session_id"] == "sub01_test").all()

        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assert metadata["task_type"] == "baseline_rest"
        assert metadata["subjective_fatigue_score"] == pytest.approx(3.5)
        assert metadata["session_id"] == "sub01_test"
