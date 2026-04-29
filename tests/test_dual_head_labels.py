"""Tests for :mod:`neuroguage.dual_head_labels`."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroguage.dual_head import (  # noqa: E402
    ATTENTION_LABEL_COLUMN,
    DualHeadLabelConfig,
    DualHeadLabelError,
    METABOLIC_LABEL_COLUMN,
    MISMATCH_LABEL_COLUMN,
    derive_dual_head_labels,
    summarise_dual_head_labels,
)


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Engaged + healthy
            {
                "session_id": "s01", "task_phase": "n_back",
                "subjective_fatigue": 2.0, "performance_drift": 0.05,
                "optical_coverage": 0.95, "imu_motion_score": 0.10,
            },
            # Engaged + degraded hemodynamics  -> mismatch
            {
                "session_id": "s01", "task_phase": "code_comprehension",
                "subjective_fatigue": 8.0, "performance_drift": 0.10,
                "optical_coverage": 0.55, "imu_motion_score": 0.65,
            },
            # Resting baseline -> not attentive (drift is highest in session)
            {
                "session_id": "s01", "task_phase": "baseline",
                "subjective_fatigue": 1.0, "performance_drift": 0.50,
                "optical_coverage": 0.95, "imu_motion_score": 0.05,
            },
        ]
    )


def test_derive_dual_head_labels_adds_required_columns() -> None:
    out = derive_dual_head_labels(_toy_frame())
    assert ATTENTION_LABEL_COLUMN in out.columns
    assert METABOLIC_LABEL_COLUMN in out.columns
    assert MISMATCH_LABEL_COLUMN in out.columns
    assert "attention_score" in out.columns
    assert "metabolic_score" in out.columns


def test_mismatch_label_fires_for_engaged_but_unhealthy_row() -> None:
    out = derive_dual_head_labels(_toy_frame())
    # Row 1 is the engaged-but-degraded case
    assert out.loc[1, ATTENTION_LABEL_COLUMN] == 1
    assert out.loc[1, METABOLIC_LABEL_COLUMN] == 0
    assert out.loc[1, MISMATCH_LABEL_COLUMN] == 1


def test_baseline_row_is_not_attentive() -> None:
    out = derive_dual_head_labels(_toy_frame())
    assert out.loc[2, ATTENTION_LABEL_COLUMN] == 0
    assert out.loc[2, MISMATCH_LABEL_COLUMN] == 0


def test_thresholds_are_configurable() -> None:
    aggressive = DualHeadLabelConfig(attention_threshold=0.05)
    out = derive_dual_head_labels(_toy_frame(), config=aggressive)
    # Lowering the bar should turn at least one resting row attentive.
    assert out[ATTENTION_LABEL_COLUMN].sum() >= 2


def test_missing_required_column_raises() -> None:
    bad = _toy_frame().drop(columns=["optical_coverage"])
    with pytest.raises(DualHeadLabelError):
        derive_dual_head_labels(bad)


def test_summarise_returns_rate_keys() -> None:
    out = derive_dual_head_labels(_toy_frame())
    summary = summarise_dual_head_labels(out)
    assert summary["rows"] == len(out)
    assert 0.0 <= summary["attention_positive_rate"] <= 1.0
    assert 0.0 <= summary["metabolic_positive_rate"] <= 1.0
    assert 0.0 <= summary["mismatch_positive_rate"] <= 1.0
