"""Day 4 — T13  Symbolic Guardrail Engine.

Maps continuous fatigue risk scores to discrete alert states with:

  T13.1  GREEN, YELLOW, RED, and optional BLACK transition logic.
  T13.2  Persistence and trend checks to suppress oscillation.
  T13.3  Rationale text for each non-GREEN transition.
  T13.4  Timestamped transition logs for replay and post-hoc review.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert States
# ---------------------------------------------------------------------------

class AlertState(str, Enum):
    """Symbolic fatigue risk states."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    BLACK = "BLACK"

    def severity(self) -> int:
        """Return numeric severity for ordering comparisons."""
        return _SEVERITY_MAP[self]


_SEVERITY_MAP: dict[AlertState, int] = {
    AlertState.GREEN: 0,
    AlertState.YELLOW: 1,
    AlertState.RED: 2,
    AlertState.BLACK: 3,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GuardrailConfig:
    """Configurable thresholds and persistence parameters."""

    # Transition thresholds (risk_score ∈ [0, 1])
    yellow_threshold: float = 0.40
    red_threshold: float = 0.65
    black_threshold: float = 0.85

    # De-escalation thresholds (hysteresis band)
    green_recovery_threshold: float = 0.30
    yellow_recovery_threshold: float = 0.55
    red_recovery_threshold: float = 0.75

    # Persistence: minimum consecutive windows at a level before transition
    escalation_persistence: int = 3
    de_escalation_persistence: int = 5

    # Trend check: if risk has been consistently rising over this many windows,
    # allow faster escalation.
    trend_window: int = 5
    trend_rising_threshold: float = 0.05   # minimum slope to count as "rising"

    # Enable BLACK state (optional per PRD)
    enable_black: bool = True

    # Minimum confidence before we allow RED or BLACK escalation
    min_confidence_for_red: float = 0.3


# ---------------------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------------------

@dataclass
class GuardrailTransition:
    """Timestamped record of a state change (T13.4)."""

    timestamp_s: float
    previous_state: str
    new_state: str
    risk_score: float
    model_confidence: float
    rationale: str
    persistence_count: int
    trend_direction: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine state
# ---------------------------------------------------------------------------

@dataclass
class GuardrailState:
    """Mutable state for the guardrail engine."""

    current_state: AlertState = AlertState.GREEN
    state_duration: int = 0            # windows in current state
    pending_state: AlertState | None = None
    pending_count: int = 0             # consecutive windows pending transition
    risk_history: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    transitions: list[GuardrailTransition] = field(default_factory=list)
    update_count: int = 0


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def _score_to_raw_state(score: float, config: GuardrailConfig) -> AlertState:
    """Map a raw risk score to a state ignoring persistence / hysteresis."""
    if config.enable_black and score >= config.black_threshold:
        return AlertState.BLACK
    if score >= config.red_threshold:
        return AlertState.RED
    if score >= config.yellow_threshold:
        return AlertState.YELLOW
    return AlertState.GREEN


def _compute_trend(risk_history: deque[float], window: int) -> tuple[str, float]:
    """Compute the recent trend direction and slope."""
    if len(risk_history) < 2:
        return "stable", 0.0

    recent = list(risk_history)[-window:]
    if len(recent) < 2:
        return "stable", 0.0

    import numpy as np  # local import to keep module-level dependency light
    x = np.arange(len(recent), dtype=float)
    y = np.array(recent, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])

    if slope > 0.01:
        return "rising", slope
    if slope < -0.01:
        return "falling", slope
    return "stable", slope


def _build_rationale(
    new_state: AlertState,
    risk_score: float,
    model_confidence: float,
    trend_direction: str,
    trend_slope: float,
    persistence_count: int,
    config: GuardrailConfig,
) -> str:
    """Generate human-readable rationale for a non-GREEN transition (T13.3)."""
    if new_state == AlertState.GREEN:
        return ""

    parts: list[str] = []

    if new_state == AlertState.YELLOW:
        parts.append(
            f"Risk score ({risk_score:.2f}) exceeded YELLOW threshold "
            f"({config.yellow_threshold:.2f})."
        )
    elif new_state == AlertState.RED:
        parts.append(
            f"Risk score ({risk_score:.2f}) exceeded RED threshold "
            f"({config.red_threshold:.2f})."
        )
    elif new_state == AlertState.BLACK:
        parts.append(
            f"Risk score ({risk_score:.2f}) exceeded BLACK threshold "
            f"({config.black_threshold:.2f}). Extreme fatigue risk."
        )

    if trend_direction == "rising":
        parts.append(
            f"Risk trend is rising (slope={trend_slope:.4f} over "
            f"{config.trend_window} windows)."
        )

    if model_confidence < 0.5:
        parts.append(
            f"Model confidence is low ({model_confidence:.2f}); "
            "interpret with caution."
        )

    parts.append(f"Persistence count: {persistence_count}.")
    return " ".join(parts)


def update_guardrail(
    risk_score: float,
    model_confidence: float,
    timestamp_s: float,
    state: GuardrailState,
    config: GuardrailConfig | None = None,
) -> AlertState:
    """Process a new risk score and return the current guardrail state.

    Implements T13.1 (transition logic), T13.2 (persistence and trend
    checks), T13.3 (rationale text), and T13.4 (transition logs).
    """
    if config is None:
        config = GuardrailConfig()

    state.update_count += 1
    state.risk_history.append(risk_score)
    state.state_duration += 1

    raw_target = _score_to_raw_state(risk_score, config)
    trend_direction, trend_slope = _compute_trend(state.risk_history, config.trend_window)

    previous_state = state.current_state
    current_severity = previous_state.severity()
    target_severity = raw_target.severity()

    # -----------------------------------------------------------------------
    # Escalation (target is more severe than current)
    # -----------------------------------------------------------------------
    if target_severity > current_severity:
        # Block RED / BLACK escalation if confidence is too low
        if raw_target.severity() >= AlertState.RED.severity() and model_confidence < config.min_confidence_for_red:
            # Don't escalate beyond YELLOW with low confidence
            raw_target = AlertState.YELLOW if current_severity < AlertState.YELLOW.severity() else previous_state

        if state.pending_state == raw_target:
            state.pending_count += 1
        else:
            state.pending_state = raw_target
            state.pending_count = 1

        # Trend-accelerated escalation: halve persistence requirement if rising
        required_persistence = config.escalation_persistence
        if trend_direction == "rising" and trend_slope >= config.trend_rising_threshold:
            required_persistence = max(1, required_persistence // 2)

        if state.pending_count >= required_persistence:
            # Commit escalation
            rationale = _build_rationale(
                new_state=raw_target,
                risk_score=risk_score,
                model_confidence=model_confidence,
                trend_direction=trend_direction,
                trend_slope=trend_slope,
                persistence_count=state.pending_count,
                config=config,
            )
            state.transitions.append(
                GuardrailTransition(
                    timestamp_s=timestamp_s,
                    previous_state=previous_state.value,
                    new_state=raw_target.value,
                    risk_score=risk_score,
                    model_confidence=model_confidence,
                    rationale=rationale,
                    persistence_count=state.pending_count,
                    trend_direction=trend_direction,
                )
            )
            LOGGER.info(
                "Guardrail ESCALATION: %s -> %s | risk=%.4f | %s",
                previous_state.value,
                raw_target.value,
                risk_score,
                rationale,
            )
            state.current_state = raw_target
            state.state_duration = 0
            state.pending_state = None
            state.pending_count = 0

    # -----------------------------------------------------------------------
    # De-escalation (target is less severe than current)
    # -----------------------------------------------------------------------
    elif target_severity < current_severity:
        # Check recovery thresholds (hysteresis)
        can_recover = False
        if state.current_state == AlertState.BLACK and risk_score <= config.red_recovery_threshold:
            can_recover = True
        elif state.current_state == AlertState.RED and risk_score <= config.yellow_recovery_threshold:
            can_recover = True
        elif state.current_state == AlertState.YELLOW and risk_score <= config.green_recovery_threshold:
            can_recover = True

        if can_recover:
            if state.pending_state == raw_target:
                state.pending_count += 1
            else:
                state.pending_state = raw_target
                state.pending_count = 1

            if state.pending_count >= config.de_escalation_persistence:
                # Step down by one level (not jump multiple levels at once)
                if state.current_state == AlertState.BLACK:
                    next_state = AlertState.RED
                elif state.current_state == AlertState.RED:
                    next_state = AlertState.YELLOW
                else:
                    next_state = AlertState.GREEN

                rationale = (
                    f"Risk score ({risk_score:.2f}) has remained below recovery threshold "
                    f"for {state.pending_count} consecutive windows. "
                    f"De-escalating from {state.current_state.value} to {next_state.value}."
                )
                state.transitions.append(
                    GuardrailTransition(
                        timestamp_s=timestamp_s,
                        previous_state=state.current_state.value,
                        new_state=next_state.value,
                        risk_score=risk_score,
                        model_confidence=model_confidence,
                        rationale=rationale,
                        persistence_count=state.pending_count,
                        trend_direction=trend_direction,
                    )
                )
                LOGGER.info(
                    "Guardrail DE-ESCALATION: %s -> %s | risk=%.4f | %s",
                    state.current_state.value,
                    next_state.value,
                    risk_score,
                    rationale,
                )
                state.current_state = next_state
                state.state_duration = 0
                state.pending_state = None
                state.pending_count = 0
        else:
            # Not below hysteresis band — don't start counting
            state.pending_state = None
            state.pending_count = 0

    # -----------------------------------------------------------------------
    # Same level — reset pending if it drifted
    # -----------------------------------------------------------------------
    else:
        state.pending_state = None
        state.pending_count = 0

    return state.current_state


# ---------------------------------------------------------------------------
# Batch processing (replay)
# ---------------------------------------------------------------------------

def run_guardrail_sequence(
    risk_scores: list[float],
    model_confidences: list[float] | None = None,
    timestamps_s: list[float] | None = None,
    config: GuardrailConfig | None = None,
) -> GuardrailState:
    """Process a sequence of risk scores through the guardrail engine.

    Returns the final GuardrailState with all transitions logged.
    """
    if config is None:
        config = GuardrailConfig()

    if model_confidences is None:
        model_confidences = [1.0] * len(risk_scores)

    if timestamps_s is None:
        timestamps_s = [float(i) for i in range(len(risk_scores))]

    state = GuardrailState()

    for score, confidence, ts in zip(risk_scores, model_confidences, timestamps_s):
        update_guardrail(
            risk_score=score,
            model_confidence=confidence,
            timestamp_s=ts,
            state=state,
            config=config,
        )

    LOGGER.info(
        "Guardrail sequence complete: %d updates, %d transitions, final state: %s",
        state.update_count,
        len(state.transitions),
        state.current_state.value,
    )
    return state


# ---------------------------------------------------------------------------
# Artifact persistence (T13.4)
# ---------------------------------------------------------------------------

def write_guardrail_artifacts(
    state: GuardrailState,
    transition_log_path: str | Path,
    summary_path: str | Path,
) -> None:
    """Persist timestamped transition logs and summary for post-hoc review."""
    log_path = Path(transition_log_path)
    sum_path = Path(summary_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)

    transition_records = [
        {
            "timestamp_s": t.timestamp_s,
            "previous_state": t.previous_state,
            "new_state": t.new_state,
            "risk_score": t.risk_score,
            "model_confidence": t.model_confidence,
            "rationale": t.rationale,
            "persistence_count": t.persistence_count,
            "trend_direction": t.trend_direction,
        }
        for t in state.transitions
    ]

    log_path.write_text(json.dumps(transition_records, indent=2), encoding="utf-8")

    # Compute state distribution
    risk_list = list(state.risk_history)
    state_counts: dict[str, int] = {"GREEN": 0, "YELLOW": 0, "RED": 0, "BLACK": 0}
    for t in state.transitions:
        state_counts[t.new_state] = state_counts.get(t.new_state, 0) + 1

    summary = {
        "total_updates": state.update_count,
        "total_transitions": len(state.transitions),
        "final_state": state.current_state.value,
        "transition_counts": state_counts,
        "risk_score_stats": {
            "mean": float(sum(risk_list) / len(risk_list)) if risk_list else 0.0,
            "min": float(min(risk_list)) if risk_list else 0.0,
            "max": float(max(risk_list)) if risk_list else 0.0,
        },
    }
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Guardrail artifacts written: %s, %s", log_path, sum_path)
