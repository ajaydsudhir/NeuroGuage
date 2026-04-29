"""Tests for the symbolic MetabolicMismatchRule."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroguage.guardrails import (  # noqa: E402
    GuardrailConfig,
    MismatchState,
    evaluate_metabolic_mismatch,
    write_mismatch_artifacts,
)


def test_rule_does_not_fire_below_persistence() -> None:
    state = MismatchState()
    config = GuardrailConfig(mismatch_persistence=3)
    for ts in range(2):
        evaluate_metabolic_mismatch(0.9, 0.1, float(ts), state, config)
    assert state.active is False
    assert state.pending_count == 2


def test_rule_fires_after_persistence_window() -> None:
    state = MismatchState()
    config = GuardrailConfig(mismatch_persistence=3)
    activated_event = None
    for ts in range(3):
        event = evaluate_metabolic_mismatch(0.9, 0.1, float(ts), state, config)
        if event is not None:
            activated_event = event
    assert state.active is True
    assert activated_event is not None
    assert activated_event.active is True
    assert activated_event.persistence_count >= 3


def test_rule_clears_when_metabolic_recovers() -> None:
    state = MismatchState()
    config = GuardrailConfig(mismatch_persistence=2)
    for ts in range(3):
        evaluate_metabolic_mismatch(0.9, 0.1, float(ts), state, config)
    assert state.active is True

    cleared_event = evaluate_metabolic_mismatch(0.9, 0.8, 99.0, state, config)
    assert cleared_event is not None
    assert cleared_event.active is False
    assert state.active is False


def test_rule_does_not_fire_when_attention_too_low() -> None:
    state = MismatchState()
    config = GuardrailConfig(mismatch_persistence=2)
    for ts in range(5):
        evaluate_metabolic_mismatch(0.3, 0.1, float(ts), state, config)
    assert state.active is False


def test_rule_does_not_fire_when_risk_is_below_threshold() -> None:
    state = MismatchState()
    config = GuardrailConfig(mismatch_persistence=2, mismatch_min_risk=0.6)
    for ts in range(5):
        evaluate_metabolic_mismatch(0.9, 0.1, float(ts), state, risk_score=0.45, config=config)
    assert state.active is False


def test_artifacts_written(tmp_path: Path) -> None:
    state = MismatchState()
    config = GuardrailConfig(mismatch_persistence=2)
    for ts in range(2):
        evaluate_metabolic_mismatch(0.9, 0.1, float(ts), state, config)
    output = tmp_path / "mismatch.json"
    write_mismatch_artifacts(state, output)
    assert output.exists()
    assert output.read_text().strip().startswith("[")


def test_invalid_mismatch_thresholds_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        GuardrailConfig.from_dict({
            "mismatch_attention_min": 1.20,
        })
