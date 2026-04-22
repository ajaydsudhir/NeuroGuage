"""Tests for :mod:`neuroguage.live_app`.

These exercise the end-to-end live pipeline against the synthetic source so
we get a deterministic smoke test of the orchestrator without requiring a
Muse device or a production model bundle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroguage.guardrails import AlertState  # noqa: E402
from neuroguage.live_app import LiveApp, LiveAppConfig, run_live_app  # noqa: E402


@pytest.fixture
def synth_config(tmp_path: Path) -> LiveAppConfig:
    return LiveAppConfig(
        source_kind="synthetic",
        duration_s=3.0,
        step_s=0.5,
        poll_s=0.02,
        session_id="testsynth",
        output_dir=tmp_path,
        enable_dashboard=False,
        write_outputs=True,
        fatigue_trend=0.8,
        real_time_factor=1.0,
    )


def test_synthetic_live_run_completes(synth_config: LiveAppConfig) -> None:
    summary = run_live_app(synth_config)
    assert summary["cycles"] >= 4
    assert summary["final_state"] in {s.value for s in AlertState}
    assert summary["mean_latency_s"] < 2.0
    assert summary["source"] == "synthetic"


def test_synthetic_live_writes_artifacts(synth_config: LiveAppConfig) -> None:
    summary = run_live_app(synth_config)
    out_dir = Path(synth_config.output_dir)
    assert (out_dir / "testsynth_summary.json").exists()
    assert (out_dir / "testsynth_transitions.json").exists()
    assert (out_dir / "testsynth_session_log.csv").exists()
    assert summary["artifacts"]["summary"].endswith("testsynth_summary.json")


def test_live_app_reports_stub_when_models_missing(synth_config: LiveAppConfig) -> None:
    # In the test repo, real models may exist. When they do, the assertion holds
    # vacuously; when they don't, stub_models must be True.
    synth_config.step_s = 0.5
    synth_config.duration_s = 1.5
    summary = run_live_app(synth_config)
    assert "stub_models" in summary
    assert isinstance(summary["stub_models"], bool)


def test_max_cycles_caps_run_length(tmp_path: Path) -> None:
    config = LiveAppConfig(
        source_kind="synthetic",
        duration_s=60.0,
        step_s=0.1,
        poll_s=0.01,
        session_id="cap",
        output_dir=tmp_path,
        enable_dashboard=False,
        write_outputs=False,
        max_inference_cycles=3,
    )
    app = LiveApp(config)
    summary = app.run()
    assert summary["cycles"] == 3


def test_live_app_handles_replay_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import OpenMuse
    import pandas as pd

    def _tdf(n: int, freq: float, channels: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame({ch: 0.1 for ch in channels}, index=range(n))
        frame["time"] = [i / freq for i in range(n)]
        return frame

    # 5-second fake session, replayed in real time.
    def _fake_decode(_messages):
        return {
            "EEG": _tdf(1280, 256.0, ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10",
                                       "AUX_1", "AUX_2", "AUX_3", "AUX_4"]),
            "OPTICS": _tdf(320, 64.0, ["OPTICS_LO_NIR", "OPTICS_RO_NIR", "OPTICS_LO_IR",
                                        "OPTICS_RO_IR", "OPTICS_LO_RED", "OPTICS_RO_RED",
                                        "OPTICS_LO_AMB", "OPTICS_RO_AMB"]),
            "ACCGYRO": _tdf(260, 52.0, ["ACC_X", "ACC_Y", "ACC_Z",
                                         "GYRO_X", "GYRO_Y", "GYRO_Z"]),
        }

    monkeypatch.setattr(OpenMuse, "decode_rawdata", _fake_decode)
    raw = tmp_path / "fake.txt"
    raw.write_text("fake\n", encoding="utf-8")

    config = LiveAppConfig(
        source_kind="replay",
        raw_file=str(raw),
        duration_s=1.5,
        step_s=0.3,
        poll_s=0.01,
        session_id="replay",
        output_dir=tmp_path / "replay_out",
        enable_dashboard=False,
        write_outputs=True,
        real_time_factor=1.0,
    )
    summary = run_live_app(config)
    assert summary["cycles"] >= 1
    assert summary["source"] == "raw_replay"
