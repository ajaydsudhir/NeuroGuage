"""Personal-baseline calibration (Fix B).

The Day 4 fusion checkpoint has a compressed dynamic range and a per-subject
operating point that is far from the textbook ``[0, 1]`` distribution the
guardrail thresholds were originally chosen for.  This module addresses the
problem by collecting a short *baseline* segment at the start of a session,
estimating ``(mean, stdev)`` of the model's risk output for **this user, this
session, this electrode placement**, and then expressing every subsequent
risk score as a personalised, normalised value:

.. math::

    z = (raw - \\mu_{baseline}) / \\sigma_{baseline}
    normalised = clip(0.5 + z / (2 \\cdot k), 0, 1)

with ``k = 4`` by default so that ``baseline_mean -> 0.5`` (centre),
``baseline_mean + 2\\sigma -> 0.75`` (clear elevation),
``baseline_mean + 4\\sigma -> 1.0`` (saturation).

A floor on ``\\sigma_{baseline}`` prevents single-sample noise from collapsing
the divisor and producing oscillatory output.

The class is intentionally pure-Python and side-effect-free outside of its
internal state; persistence (``save``/``load``) is a thin JSON layer so a
baseline collected during one session can be re-used for the next without
recollecting (useful for in-class demos: collect once, demo many times).
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineStats:
    """Persisted personal baseline (saved alongside session artifacts)."""

    mean: float
    stdev: float
    samples: int
    duration_s: float
    floor_used: float
    raw_min: float
    raw_max: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "stdev": self.stdev,
            "samples": self.samples,
            "duration_s": self.duration_s,
            "floor_used": self.floor_used,
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineStats":
        return cls(
            mean=float(payload["mean"]),
            stdev=float(payload["stdev"]),
            samples=int(payload["samples"]),
            duration_s=float(payload["duration_s"]),
            floor_used=float(payload.get("floor_used", 0.04)),
            raw_min=float(payload.get("raw_min", payload["mean"])),
            raw_max=float(payload.get("raw_max", payload["mean"])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class CalibrationResult:
    """Output of one calibrator update."""

    raw_risk: float
    normalised_risk: float
    z_score: float
    phase: str          # "baseline" | "active" | "passthrough"


class PersonalBaselineCalibrator:
    """Collect a baseline window then z-score subsequent risk scores.

    Parameters
    ----------
    baseline_duration_s
        Length of the baseline-collection window in seconds.  ``0`` (or
        negative) disables calibration and the class becomes a no-op
        passthrough that returns ``raw_risk`` unchanged.
    stdev_floor
        Minimum stdev used in the divisor.  Prevents an unrealistically tight
        baseline (e.g. synthetic source with ``stdev = 0.014``) from making
        every tiny noise jiggle saturate the normalised output.  ``0.04`` is
        a sensible default that corresponds to ~10 % of the typical raw
        risk-score range and roughly matches measurement noise.
    sigma_span
        How many standard deviations map to the upper edge of the normalised
        output (default ``4`` ⇒ ``baseline_mean + 4 sigma -> 1.0``).
    minimum_samples
        Minimum number of risk samples required before the baseline is
        accepted.  Below this we keep collecting even past the time budget.
    """

    def __init__(
        self,
        baseline_duration_s: float = 20.0,
        stdev_floor: float = 0.04,
        sigma_span: float = 4.0,
        minimum_samples: int = 5,
    ) -> None:
        self.baseline_duration_s = float(baseline_duration_s)
        self.stdev_floor = float(stdev_floor)
        self.sigma_span = float(sigma_span)
        self.minimum_samples = int(minimum_samples)

        self._raw_samples: list[float] = []
        self._first_timestamp_s: Optional[float] = None
        self._baseline: Optional[BaselineStats] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.baseline_duration_s > 0.0

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> Optional[BaselineStats]:
        return self._baseline

    @property
    def samples_collected(self) -> int:
        return len(self._raw_samples)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, raw_risk: float, timestamp_s: float) -> CalibrationResult:
        """Feed a new risk sample and return the normalised value.

        While in the baseline window the calibrator returns
        ``raw_risk`` unchanged so the dashboard shows the un-personalised
        signal during the "calibrating" period.  Once the baseline is
        ready, every subsequent call returns the z-scored / clipped value.
        """
        raw = float(raw_risk)

        if not self.enabled:
            return CalibrationResult(
                raw_risk=raw, normalised_risk=raw, z_score=0.0, phase="passthrough"
            )

        if self._first_timestamp_s is None:
            self._first_timestamp_s = float(timestamp_s)

        if not self.baseline_ready:
            self._raw_samples.append(raw)
            elapsed = float(timestamp_s) - self._first_timestamp_s
            time_done = elapsed >= self.baseline_duration_s
            samples_done = len(self._raw_samples) >= self.minimum_samples
            if time_done and samples_done:
                self._finalise_baseline(elapsed)
            return CalibrationResult(
                raw_risk=raw,
                normalised_risk=raw,
                z_score=0.0,
                phase="baseline",
            )

        # Active phase: z-score against personal baseline.
        baseline = self._baseline
        assert baseline is not None
        sigma = max(baseline.stdev, baseline.floor_used)
        z = (raw - baseline.mean) / sigma
        normalised = 0.5 + z / (2.0 * self.sigma_span)
        normalised = max(0.0, min(1.0, normalised))
        return CalibrationResult(
            raw_risk=raw,
            normalised_risk=normalised,
            z_score=z,
            phase="active",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _finalise_baseline(self, duration_s: float) -> None:
        samples = self._raw_samples
        if len(samples) < 2:
            stdev = self.stdev_floor
        else:
            stdev = float(statistics.pstdev(samples))
        baseline = BaselineStats(
            mean=float(statistics.fmean(samples)),
            stdev=stdev,
            samples=len(samples),
            duration_s=duration_s,
            floor_used=self.stdev_floor,
            raw_min=float(min(samples)),
            raw_max=float(max(samples)),
        )
        self._baseline = baseline
        LOGGER.info(
            "Personal baseline established | samples=%d | duration=%.1fs | "
            "raw mean=%.3f stdev=%.3f (floor=%.3f) | range=[%.3f, %.3f] | "
            "baseline -> normalised 0.5; +%.0f sigma -> normalised 1.0",
            baseline.samples, baseline.duration_s,
            baseline.mean, baseline.stdev, baseline.floor_used,
            baseline.raw_min, baseline.raw_max,
            self.sigma_span,
        )

    # ------------------------------------------------------------------
    # Manual baseline plumbing
    # ------------------------------------------------------------------

    def adopt_baseline(self, baseline: BaselineStats) -> None:
        """Skip the baseline-collection phase and use a previously saved one."""
        if not self.enabled:
            LOGGER.warning("Calibrator is disabled; adopt_baseline() is a no-op.")
            return
        if not math.isfinite(baseline.mean) or not math.isfinite(baseline.stdev):
            raise ValueError(f"Adopted baseline contains non-finite values: {baseline}")
        self._baseline = baseline
        LOGGER.info(
            "Adopted pre-existing personal baseline | mean=%.3f stdev=%.3f "
            "(samples=%d, duration=%.1fs)",
            baseline.mean, baseline.stdev, baseline.samples, baseline.duration_s,
        )

    def save(self, path: str | Path) -> Path:
        if self._baseline is None:
            raise RuntimeError("No baseline to save (calibration phase not yet complete).")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._baseline.to_dict(), indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load_baseline(cls, path: str | Path) -> BaselineStats:
        return BaselineStats.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    def state_for_log(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "baseline_duration_s": self.baseline_duration_s,
            "stdev_floor": self.stdev_floor,
            "sigma_span": self.sigma_span,
            "baseline_ready": self.baseline_ready,
            "baseline": self._baseline.to_dict() if self._baseline else None,
            "samples_collected": self.samples_collected,
        }


__all__ = [
    "BaselineStats",
    "CalibrationResult",
    "PersonalBaselineCalibrator",
]
