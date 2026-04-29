"""Guardrail logic for intervention states.

Turns a continuous risk score into user-facing states (`GREEN`, `YELLOW`,
`RED`, and optional `BLACK`) with hysteresis, persistence checks, and
transition logs that can be replayed later.
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


# Alert States

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


# Configuration

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

    # ----- Optional human-readable metadata (preserved through round-trip) ----
    profile_name: str = ""                 # e.g. "default", "calibrated", "personalized"
    profile_description: str = ""

    # ----- Neuro-symbolic MetabolicMismatchRule (dual-head model) -------
    # The rule fires when the user is *engaged but their hemodynamic
    # readiness has dropped*, i.e. attention_prob >= mismatch_attention_min
    # AND metabolic_prob <= mismatch_metabolic_max for at least
    # mismatch_persistence consecutive cycles.
    mismatch_attention_min: float = 0.60
    mismatch_metabolic_max: float = 0.40
    mismatch_persistence: int = 3
    mismatch_min_risk: float = 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    _NUMERIC_FIELDS = (
        "yellow_threshold", "red_threshold", "black_threshold",
        "green_recovery_threshold", "yellow_recovery_threshold",
        "red_recovery_threshold",
        "trend_rising_threshold", "min_confidence_for_red",
        "mismatch_attention_min", "mismatch_metabolic_max", "mismatch_min_risk",
    )
    _INT_FIELDS = (
        "escalation_persistence", "de_escalation_persistence", "trend_window",
        "mismatch_persistence",
    )
    _BOOL_FIELDS = ("enable_black",)
    _STR_FIELDS = ("profile_name", "profile_description")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GuardrailConfig":
        """Build a config from a (possibly partial) dict of overrides.

        Unknown keys are ignored after a debug-level log so older configs
        keep working when new fields are added.
        """
        kwargs: dict[str, Any] = {}
        for key in cls._NUMERIC_FIELDS:
            if key in payload:
                kwargs[key] = float(payload[key])
        for key in cls._INT_FIELDS:
            if key in payload:
                kwargs[key] = int(payload[key])
        for key in cls._BOOL_FIELDS:
            if key in payload:
                kwargs[key] = bool(payload[key])
        for key in cls._STR_FIELDS:
            if key in payload:
                kwargs[key] = str(payload[key])

        unknown = set(payload) - set(
            cls._NUMERIC_FIELDS + cls._INT_FIELDS + cls._BOOL_FIELDS + cls._STR_FIELDS
        )
        if unknown:
            LOGGER.debug("GuardrailConfig.from_dict ignoring unknown keys: %s", sorted(unknown))

        cfg = cls(**kwargs)
        cfg._validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "GuardrailConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls.from_dict(payload)
        if not cfg.profile_name:
            cfg.profile_name = Path(path).stem
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: getattr(self, k) for k in self._NUMERIC_FIELDS},
            **{k: getattr(self, k) for k in self._INT_FIELDS},
            **{k: getattr(self, k) for k in self._BOOL_FIELDS},
            **{k: getattr(self, k) for k in self._STR_FIELDS},
        }

    def _validate(self) -> None:
        """Enforce monotonicity and sane ranges so a bad config fails fast."""
        if not (0.0 <= self.green_recovery_threshold
                < self.yellow_threshold
                <= self.yellow_recovery_threshold
                < self.red_threshold
                <= self.red_recovery_threshold
                < self.black_threshold <= 1.0):
            raise ValueError(
                "GuardrailConfig thresholds must satisfy: "
                "0 <= green_recovery < yellow <= yellow_recovery "
                "< red <= red_recovery < black <= 1; got "
                f"{self.to_dict()}"
            )
        for k in self._INT_FIELDS:
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be >= 1, got {getattr(self, k)}")
        if not (0.0 <= self.mismatch_metabolic_max <= 1.0):
            raise ValueError(
                "GuardrailConfig mismatch_metabolic_max must satisfy "
                "0 <= mismatch_metabolic_max <= 1, "
                f"got {self.mismatch_metabolic_max}"
            )
        if not (0.0 <= self.mismatch_attention_min <= 1.0):
            raise ValueError(
                "GuardrailConfig mismatch_attention_min must satisfy "
                "0 <= mismatch_attention_min <= 1, "
                f"got {self.mismatch_attention_min}"
            )
        if not (0.0 <= self.mismatch_min_risk <= 1.0):
            raise ValueError(
                "GuardrailConfig mismatch_min_risk must satisfy "
                "0 <= mismatch_min_risk <= 1, "
                f"got {self.mismatch_min_risk}"
            )


# Transition record

@dataclass
class GuardrailTransition:
    """Timestamped record of a state change."""

    timestamp_s: float
    previous_state: str
    new_state: str
    risk_score: float
    model_confidence: float
    rationale: str
    persistence_count: int
    trend_direction: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


# Engine state

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


# Core engine

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
    """Generate human-readable rationale for a non-GREEN transition."""
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

    Implements (transition logic), (persistence and trend
    checks), (rationale text), and (transition logs).
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


# Neuro-symbolic MetabolicMismatchRule (dual-head model)

@dataclass
class MismatchEvent:
    """Timestamped record of a metabolic-mismatch flag transition."""

    timestamp_s: float
    active: bool
    attention_prob: float
    metabolic_prob: float
    mismatch_prob: float
    persistence_count: int
    rationale: str


@dataclass
class MismatchState:
    """Mutable state for the MetabolicMismatchRule.

    Tracks how long the symbolic AND has been satisfied so we don't fire on
    a single noisy frame.  Holding the rationale makes it cheap for the
    dashboard to display "you've been engaged but under-supplied for X
    cycles -> consider a break".
    """

    active: bool = False
    pending_count: int = 0
    last_attention_prob: float = 0.0
    last_metabolic_prob: float = 1.0
    last_mismatch_prob: float = 0.0
    events: list[MismatchEvent] = field(default_factory=list)


def evaluate_metabolic_mismatch(
    attention_prob: float,
    metabolic_prob: float,
    timestamp_s: float,
    state: MismatchState,
    risk_score: float | None = None,
    config: GuardrailConfig | None = None,
) -> MismatchEvent | None:
    """Symbolic AND-rule on the dual-head outputs.

    Returns a :class:`MismatchEvent` *only* when the active flag flips this
    cycle (rising or falling edge), so callers can append it to a transition
    log without de-duping.  Updates the supplied ``state`` in place either
    way so the caller can query ``state.active`` every frame.
    """
    # Backward compatibility: older call sites passed `config` as the fifth
    # positional argument before `risk_score` was introduced.
    if isinstance(risk_score, GuardrailConfig) and config is None:
        config = risk_score
        risk_score = None

    if config is None:
        config = GuardrailConfig()

    state.last_attention_prob = float(attention_prob)
    state.last_metabolic_prob = float(metabolic_prob)
    soft_and = float(attention_prob) * (1.0 - float(metabolic_prob))
    state.last_mismatch_prob = soft_and

    rule_satisfied = (
        attention_prob >= config.mismatch_attention_min
        and metabolic_prob <= config.mismatch_metabolic_max
        and (risk_score is None or risk_score >= config.mismatch_min_risk)
    )

    event: MismatchEvent | None = None

    if rule_satisfied:
        state.pending_count += 1
        if not state.active and state.pending_count >= config.mismatch_persistence:
            state.active = True
            rationale = (
                f"User is engaged (P(attention)={attention_prob:.2f} >= "
                f"{config.mismatch_attention_min:.2f}) but hemodynamic "
                f"readiness has dropped (P(metabolic_ready)={metabolic_prob:.2f} "
                f"<= {config.mismatch_metabolic_max:.2f})"
                + (
                    f" while fatigue risk is elevated ({risk_score:.2f} >= "
                    f"{config.mismatch_min_risk:.2f})"
                    if risk_score is not None and config.mismatch_min_risk > 0.0
                    else ""
                )
                + " for "
                f"{state.pending_count} consecutive cycles. Recommend break."
            )
            event = MismatchEvent(
                timestamp_s=timestamp_s,
                active=True,
                attention_prob=float(attention_prob),
                metabolic_prob=float(metabolic_prob),
                mismatch_prob=soft_and,
                persistence_count=state.pending_count,
                rationale=rationale,
            )
            state.events.append(event)
            LOGGER.info("MetabolicMismatchRule ACTIVATED: %s", rationale)
    else:
        if state.active:
            rationale = (
                "Hemodynamic readiness recovered or attention dropped — "
                f"P(attention)={attention_prob:.2f}, "
                f"P(metabolic_ready)={metabolic_prob:.2f}"
                + (
                    f", risk={risk_score:.2f}"
                    if risk_score is not None
                    else ""
                )
                + ". Mismatch cleared."
            )
            state.active = False
            event = MismatchEvent(
                timestamp_s=timestamp_s,
                active=False,
                attention_prob=float(attention_prob),
                metabolic_prob=float(metabolic_prob),
                mismatch_prob=soft_and,
                persistence_count=state.pending_count,
                rationale=rationale,
            )
            state.events.append(event)
            LOGGER.info("MetabolicMismatchRule cleared: %s", rationale)
        state.pending_count = 0

    return event


def write_mismatch_artifacts(
    state: MismatchState,
    output_path: str | Path,
) -> None:
    """Persist the mismatch transition log for post-hoc review."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "timestamp_s": e.timestamp_s,
            "active": e.active,
            "attention_prob": e.attention_prob,
            "metabolic_prob": e.metabolic_prob,
            "mismatch_prob": e.mismatch_prob,
            "persistence_count": e.persistence_count,
            "rationale": e.rationale,
        }
        for e in state.events
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# Batch processing (replay)

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


# Artifact persistence

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
