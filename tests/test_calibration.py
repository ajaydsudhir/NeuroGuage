"""Tests for the personal-baseline calibrator (Fix B)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroguage.calibration import (  # noqa: E402
    BaselineStats,
    PersonalBaselineCalibrator,
)


def _drive(
    calibrator: PersonalBaselineCalibrator,
    raw_values: list[float],
    *,
    dt: float = 0.5,
    t0: float = 0.0,
):
    results = []
    for i, v in enumerate(raw_values):
        results.append(calibrator.update(v, t0 + i * dt))
    return results


def test_calibrator_passthrough_when_disabled() -> None:
    cal = PersonalBaselineCalibrator(baseline_duration_s=0.0)
    out = cal.update(0.42, timestamp_s=10.0)
    assert out.normalised_risk == pytest.approx(0.42)
    assert out.phase == "passthrough"
    assert not cal.enabled
    assert not cal.baseline_ready


def test_calibrator_collects_baseline_within_window() -> None:
    cal = PersonalBaselineCalibrator(
        baseline_duration_s=2.0, stdev_floor=0.04, sigma_span=4.0, minimum_samples=3
    )
    raws = [0.45, 0.48, 0.50, 0.47, 0.49]
    results = _drive(cal, raws, dt=0.5)

    assert all(r.phase == "baseline" for r in results[:4])
    assert results[-1].phase in {"baseline"}
    assert all(r.normalised_risk == pytest.approx(raw) for r, raw in zip(results, raws))
    follow_up = cal.update(0.50, timestamp_s=10.0)
    assert follow_up.phase == "active"


def test_calibrator_normalises_post_baseline() -> None:
    cal = PersonalBaselineCalibrator(
        baseline_duration_s=1.0, stdev_floor=0.04, sigma_span=4.0, minimum_samples=3
    )
    _drive(cal, [0.40, 0.44, 0.48, 0.42, 0.46], dt=0.3)

    assert cal.baseline_ready
    baseline = cal.baseline
    assert baseline is not None

    centered = cal.update(baseline.mean, timestamp_s=10.0)
    assert centered.normalised_risk == pytest.approx(0.5, abs=1e-9)

    elevated = cal.update(baseline.mean + 4 * max(baseline.stdev, baseline.floor_used), timestamp_s=11.0)
    assert elevated.normalised_risk == pytest.approx(1.0, abs=1e-9)

    relaxed = cal.update(baseline.mean - 4 * max(baseline.stdev, baseline.floor_used), timestamp_s=12.0)
    assert relaxed.normalised_risk == pytest.approx(0.0, abs=1e-9)


def test_stdev_floor_prevents_oversensitivity() -> None:
    """Even if the raw stdev is tiny, normalisation should remain stable."""
    cal = PersonalBaselineCalibrator(
        baseline_duration_s=1.0, stdev_floor=0.04, sigma_span=4.0, minimum_samples=3
    )
    _drive(cal, [0.5] * 5, dt=0.3)
    assert cal.baseline_ready
    out = cal.update(0.55, timestamp_s=10.0)
    assert 0.55 < out.normalised_risk < 0.7
    assert out.normalised_risk < 1.0


def test_baseline_save_load_roundtrip(tmp_path: Path) -> None:
    cal = PersonalBaselineCalibrator(baseline_duration_s=1.0, minimum_samples=3)
    _drive(cal, [0.40, 0.44, 0.48, 0.42, 0.46], dt=0.3)
    target = tmp_path / "baseline.json"
    cal.save(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "mean" in payload and "stdev" in payload

    fresh = PersonalBaselineCalibrator(baseline_duration_s=10.0, minimum_samples=3)
    fresh.adopt_baseline(BaselineStats.from_dict(payload))
    assert fresh.baseline_ready
    out = fresh.update(payload["mean"], timestamp_s=0.0)
    assert out.phase == "active"
    assert out.normalised_risk == pytest.approx(0.5, abs=1e-9)


def test_calibrator_state_is_serialisable() -> None:
    cal = PersonalBaselineCalibrator(baseline_duration_s=1.0, minimum_samples=3)
    _drive(cal, [0.40, 0.42, 0.44, 0.46, 0.48], dt=0.3)
    info = cal.state_for_log()
    assert info["enabled"] is True
    assert info["baseline_ready"] is True
    assert info["baseline"]["samples"] == 5

    cal_off = PersonalBaselineCalibrator(baseline_duration_s=0.0)
    info_off = cal_off.state_for_log()
    assert info_off["enabled"] is False
    assert info_off["baseline"] is None
