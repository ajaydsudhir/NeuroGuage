"""Tests for Day 4 — T13 Symbolic Guardrail Engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from neuroguage.guardrails import (
    AlertState,
    GuardrailConfig,
    GuardrailState,
    run_guardrail_sequence,
    update_guardrail,
    write_guardrail_artifacts,
)


# ---------------------------------------------------------------------------
# Tests — T13.1  GREEN, YELLOW, RED, BLACK transition logic
# ---------------------------------------------------------------------------


class TestTransitionLogic:
    def test_starts_green(self) -> None:
        state = GuardrailState()
        assert state.current_state == AlertState.GREEN

    def test_escalates_to_yellow(self) -> None:
        config = GuardrailConfig(escalation_persistence=2)
        state = GuardrailState()

        # Feed scores above yellow threshold persistently
        for _ in range(5):
            update_guardrail(0.50, 1.0, float(_), state, config)

        assert state.current_state == AlertState.YELLOW

    def test_escalates_to_red(self) -> None:
        config = GuardrailConfig(escalation_persistence=2)
        state = GuardrailState()

        # First escalate to YELLOW
        for i in range(5):
            update_guardrail(0.50, 1.0, float(i), state, config)

        # Then escalate to RED
        for i in range(5, 15):
            update_guardrail(0.75, 1.0, float(i), state, config)

        assert state.current_state == AlertState.RED

    def test_escalates_to_black_when_enabled(self) -> None:
        config = GuardrailConfig(
            escalation_persistence=2,
            enable_black=True,
        )
        state = GuardrailState()

        # Escalate through all levels
        for i in range(5):
            update_guardrail(0.50, 1.0, float(i), state, config)
        for i in range(5, 10):
            update_guardrail(0.75, 1.0, float(i), state, config)
        for i in range(10, 20):
            update_guardrail(0.90, 1.0, float(i), state, config)

        assert state.current_state == AlertState.BLACK

    def test_no_black_when_disabled(self) -> None:
        config = GuardrailConfig(
            escalation_persistence=2,
            enable_black=False,
        )
        state = GuardrailState()

        for i in range(20):
            update_guardrail(0.95, 1.0, float(i), state, config)

        assert state.current_state == AlertState.RED  # Not BLACK

    def test_severity_ordering(self) -> None:
        assert AlertState.GREEN.severity() < AlertState.YELLOW.severity()
        assert AlertState.YELLOW.severity() < AlertState.RED.severity()
        assert AlertState.RED.severity() < AlertState.BLACK.severity()


# ---------------------------------------------------------------------------
# Tests — T13.2  Persistence and trend checks
# ---------------------------------------------------------------------------


class TestPersistenceAndOscillation:
    def test_no_immediate_escalation(self) -> None:
        config = GuardrailConfig(escalation_persistence=3)
        state = GuardrailState()

        # One high score should NOT immediately escalate
        update_guardrail(0.70, 1.0, 0.0, state, config)
        assert state.current_state == AlertState.GREEN

    def test_requires_persistence_count(self) -> None:
        config = GuardrailConfig(escalation_persistence=4)
        state = GuardrailState()

        # 3 high scores: not enough
        for i in range(3):
            update_guardrail(0.50, 1.0, float(i), state, config)
        assert state.current_state == AlertState.GREEN

        # 4th triggers escalation
        update_guardrail(0.50, 1.0, 3.0, state, config)
        assert state.current_state == AlertState.YELLOW

    def test_intermittent_scores_reset_pending(self) -> None:
        config = GuardrailConfig(escalation_persistence=3)
        state = GuardrailState()

        # 2 high, then 1 low, then 2 high
        update_guardrail(0.50, 1.0, 0.0, state, config)
        update_guardrail(0.50, 1.0, 1.0, state, config)
        update_guardrail(0.10, 1.0, 2.0, state, config)  # resets
        update_guardrail(0.50, 1.0, 3.0, state, config)
        update_guardrail(0.50, 1.0, 4.0, state, config)

        # Should NOT have escalated because the low score reset the counter
        # The total is only 2 consecutive at this point
        # Actually the 3rd and 4th are consecutive, need one more
        assert state.current_state == AlertState.GREEN

    def test_de_escalation_requires_more_persistence(self) -> None:
        config = GuardrailConfig(
            escalation_persistence=2,
            de_escalation_persistence=4,
        )
        state = GuardrailState()

        # Escalate to YELLOW
        for i in range(5):
            update_guardrail(0.50, 1.0, float(i), state, config)
        assert state.current_state == AlertState.YELLOW

        # 3 low scores: not enough to de-escalate
        for i in range(5, 8):
            update_guardrail(0.10, 1.0, float(i), state, config)
        assert state.current_state == AlertState.YELLOW

        # 4th low score triggers de-escalation
        update_guardrail(0.10, 1.0, 8.0, state, config)
        assert state.current_state == AlertState.GREEN

    def test_blocks_red_with_low_confidence(self) -> None:
        config = GuardrailConfig(
            escalation_persistence=2,
            min_confidence_for_red=0.5,
        )
        state = GuardrailState()

        # Try to escalate to RED with low confidence
        for i in range(10):
            update_guardrail(0.80, 0.2, float(i), state, config)

        # Should stay at YELLOW (confidence too low for RED)
        assert state.current_state == AlertState.YELLOW


# ---------------------------------------------------------------------------
# Tests — T13.3  Rationale text
# ---------------------------------------------------------------------------


class TestRationale:
    def test_rationale_on_escalation(self) -> None:
        config = GuardrailConfig(escalation_persistence=2)
        state = GuardrailState()

        for i in range(5):
            update_guardrail(0.50, 1.0, float(i), state, config)

        assert len(state.transitions) >= 1
        transition = state.transitions[0]
        assert len(transition.rationale) > 0
        assert "YELLOW" in transition.rationale

    def test_rationale_on_de_escalation(self) -> None:
        config = GuardrailConfig(escalation_persistence=2, de_escalation_persistence=3)
        state = GuardrailState()

        # Escalate
        for i in range(5):
            update_guardrail(0.50, 1.0, float(i), state, config)

        # De-escalate
        for i in range(5, 15):
            update_guardrail(0.10, 1.0, float(i), state, config)

        # Find de-escalation transition
        de_escalation = [t for t in state.transitions if t.new_state == "GREEN"]
        assert len(de_escalation) >= 1
        assert "recovery" in de_escalation[0].rationale.lower() or "de-escalat" in de_escalation[0].rationale.lower()

    def test_no_rationale_for_green(self) -> None:
        state = GuardrailState()
        config = GuardrailConfig()
        update_guardrail(0.10, 1.0, 0.0, state, config)
        # No transitions should have been logged
        assert len(state.transitions) == 0


# ---------------------------------------------------------------------------
# Tests — T13.4  Timestamped transition logs
# ---------------------------------------------------------------------------


class TestTransitionLogs:
    def test_logs_contain_timestamps(self) -> None:
        config = GuardrailConfig(escalation_persistence=2)
        state = GuardrailState()

        for i in range(5):
            update_guardrail(0.50, 1.0, float(i) * 2.0, state, config)

        for t in state.transitions:
            assert t.timestamp_s >= 0.0
            assert t.previous_state in {"GREEN", "YELLOW", "RED", "BLACK"}
            assert t.new_state in {"GREEN", "YELLOW", "RED", "BLACK"}

    def test_write_artifacts_creates_files(self) -> None:
        config = GuardrailConfig(escalation_persistence=2)
        state = GuardrailState()

        for i in range(5):
            update_guardrail(0.50, 1.0, float(i), state, config)

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "transitions.json"
            sum_path = Path(tmp) / "summary.json"
            write_guardrail_artifacts(state, log_path, sum_path)

            assert log_path.exists()
            assert sum_path.exists()

            transitions = json.loads(log_path.read_text())
            assert isinstance(transitions, list)
            assert len(transitions) >= 1

            summary = json.loads(sum_path.read_text())
            assert "total_transitions" in summary
            assert "final_state" in summary


# ---------------------------------------------------------------------------
# Tests — Batch sequence processing
# ---------------------------------------------------------------------------


class TestGuardrailSequence:
    def test_processes_sequence(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.45, 0.45, 0.45, 0.45, 0.50, 0.70, 0.70, 0.70, 0.70]
        state = run_guardrail_sequence(scores)
        assert state.update_count == len(scores)

    def test_stable_low_scores_stay_green(self) -> None:
        scores = [0.1] * 20
        state = run_guardrail_sequence(scores)
        assert state.current_state == AlertState.GREEN
        assert len(state.transitions) == 0

    def test_escalating_sequence_produces_transitions(self) -> None:
        # Gradually escalating risk
        scores = [0.1] * 5 + [0.5] * 10 + [0.8] * 10
        state = run_guardrail_sequence(scores, config=GuardrailConfig(escalation_persistence=3))
        assert len(state.transitions) >= 1
        assert state.current_state.severity() >= AlertState.YELLOW.severity()
