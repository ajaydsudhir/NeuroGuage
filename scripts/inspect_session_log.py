"""Inspect the empirical risk distribution from a live session log.

Handles both legacy logs (only ``risk_score``) and Fix B logs that include
``raw_risk``, ``normalised_risk``, ``z_score`` and ``calibration_phase``.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


def _quantiles(values: list[float], qs: tuple[float, ...]) -> dict[str, float]:
    s = sorted(values)
    n = len(s)
    return {f"p{int(q * 100)}": s[max(0, min(n - 1, int(n * q)))] for q in qs}


def _summarise(values: list[float], label: str) -> None:
    n = len(values)
    if n == 0:
        print(f"{label}: <empty>")
        return
    quants = _quantiles(values, (0.10, 0.25, 0.50, 0.75, 0.90))
    print(
        f"{label}  n={n}  mean={statistics.fmean(values):.3f}  "
        f"stdev={statistics.pstdev(values):.3f}  min={min(values):.3f}  max={max(values):.3f}  "
        + "  ".join(f"{k}={v:.3f}" for k, v in quants.items())
    )


def _band_occupancy(values: list[float], y_th: float, r_th: float, b_th: float) -> None:
    n = len(values)
    if n == 0:
        return
    bands = [
        (f"GREEN  (<{y_th:.2f})", lambda r: r < y_th),
        (f"YELLOW ({y_th:.2f}-{r_th:.2f})", lambda r: y_th <= r < r_th),
        (f"RED    ({r_th:.2f}-{b_th:.2f})", lambda r: r_th <= r < b_th),
        (f"BLACK  (>={b_th:.2f})", lambda r: r >= b_th),
    ]
    for name, pred in bands:
        c = sum(1 for v in values if pred(v))
        print(f"  {name:24s} {c:>5d}/{n} = {c / n * 100:5.1f}%")


def main(path: str, *, y_th: float = 0.40, r_th: float = 0.65, b_th: float = 0.85) -> None:
    p = Path(path)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        print(f"{p}: empty session log")
        return

    risks = [float(r["risk_score"]) for r in rows]
    states = [r["state"] for r in rows]

    has_calibration = bool(rows[0].get("normalised_risk"))
    raw = [float(r["raw_risk"]) for r in rows] if has_calibration else risks
    normalised = (
        [float(r["normalised_risk"]) for r in rows] if has_calibration else None
    )
    phases = (
        [r.get("calibration_phase", "") for r in rows] if has_calibration else None
    )

    print(f"=== {p.name} ===")
    print(f"cycles: {len(rows)}  |  calibration={'ON' if has_calibration else 'OFF'}")
    print()

    _summarise(raw, "raw_risk     ")
    if normalised is not None:
        active_norm = [n for n, ph in zip(normalised, phases or []) if ph == "active"]
        _summarise(normalised, "normalised   ")
        if active_norm:
            _summarise(active_norm, "norm (active)")
    print(f"time-in-state (cycles): {dict(Counter(states))}")

    if phases:
        print(f"calibration phases:     {dict(Counter(phases))}")

    print()
    print(f"Raw band occupancy (default thresholds y={y_th}, r={r_th}, b={b_th}):")
    _band_occupancy(raw, y_th, r_th, b_th)

    if normalised is not None:
        print()
        print("Normalised band occupancy (personalized thresholds y=0.55, r=0.70, b=0.85):")
        _band_occupancy(normalised, 0.55, 0.70, 0.85)


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0] if args else "artifacts/reports/live/classroom_demo_20260424_2359_session_log.csv"
    main(path)
