"""End-to-end live NeuroGauge application.

Glues together:

* :mod:`neuroguage.muse_stream` — pulls EEG / optical / IMU samples from a
  Muse S Athena (or synthetic / replay source) into contract-shaped frames.
* :mod:`neuroguage.inference`   — T12 real-time feature + risk path.
* :mod:`neuroguage.guardrails`  — T13 symbolic alert state machine.
* :mod:`neuroguage.dashboard`   — T14 Matplotlib projection dashboard.

The orchestrator maintains bounded ring buffers for each modality, runs an
inference cycle every ``step_s`` seconds, updates the guardrail engine, and
hands a :class:`~neuroguage.dashboard.DashboardSnapshot` to the renderer.  A
session log is persisted as CSV + JSON when the loop exits so the session
can be replayed, audited, or used as training data.

Usage::

    from neuroguage.live_app import LiveAppConfig, run_live_app
    cfg = LiveAppConfig(source_kind="synthetic", duration_s=60.0)
    summary = run_live_app(cfg)
    print(summary["final_state"], summary["mean_latency_s"])
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .calibration import (
    BaselineStats,
    CalibrationResult,
    PersonalBaselineCalibrator,
)
from .dashboard import (
    DashboardConfig,
    DashboardRenderer,
    DashboardSnapshot,
)
from .guardrails import (
    AlertState,
    GuardrailConfig,
    GuardrailState,
    GuardrailTransition,
    update_guardrail,
    write_guardrail_artifacts,
)
from .inference import (
    InferenceConfig,
    InferenceError,
    InferenceState,
    InferenceUpdate,
    load_inference_models,
    run_inference_cycle,
    write_inference_diagnostics,
)
from .muse_stream import (
    EEG_OUTPUT_COLUMNS,
    EEG_SFREQ_HZ,
    IMU_OUTPUT_COLUMNS,
    IMU_SFREQ_HZ,
    MuseDataSource,
    MuseFrames,
    OPTICAL_OUTPUT_COLUMNS,
    OPTICAL_SFREQ_HZ,
    build_data_source,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LiveAppConfig:
    """User-facing configuration for the live NeuroGauge app."""

    # Source selection
    source_kind: str = "synthetic"               # muse | synthetic | replay
    device_address: Optional[str] = None
    preset: str = "p1041"
    raw_file: Optional[str] = None
    session_id: str = "live"
    manage_bridge: bool = True
    real_time_factor: float = 1.0
    fatigue_trend: float = 0.5
    record_raw_to: Optional[str] = None
    resolve_timeout_s: float = 60.0

    # Loop timing
    duration_s: Optional[float] = None           # None = run until stop()
    step_s: float = 0.5                          # inference cadence (matches EEG step)
    poll_s: float = 0.05                         # wait between ring-buffer reads

    # Ring-buffer sizes (in seconds of native sample rate)
    eeg_buffer_s: float = 4.0
    optical_buffer_s: float = 16.0
    imu_buffer_s: float = 4.0

    # Dashboard
    enable_dashboard: bool = True
    dashboard_block: bool = True                 # True opens an interactive window
    headless_save_every: int = 0                 # >0: save PNG every N updates

    # Output
    output_dir: str | Path = "artifacts/reports/live"
    write_outputs: bool = True

    # Optional safety bounds
    max_inference_cycles: Optional[int] = None
    stop_on_finished_source: bool = True

    # ---- Fix A: JSON guardrail config ---------------------------------
    # If set, GuardrailConfig.from_json(this) is loaded at startup. ``None``
    # means "use the defaults compiled into GuardrailConfig" (back-compat).
    guardrail_config_path: Optional[str] = None

    # ---- Fix B: personal-baseline z-score calibration -----------------
    # ``baseline_duration_s = 0`` disables calibration entirely (back-compat).
    baseline_duration_s: float = 0.0
    baseline_stdev_floor: float = 0.04
    baseline_sigma_span: float = 4.0
    baseline_minimum_samples: int = 5
    save_baseline_to: Optional[str] = None      # auto-derived if None and enabled
    load_baseline_from: Optional[str] = None    # skip baseline phase, use saved one


# ---------------------------------------------------------------------------
# Ring buffers
# ---------------------------------------------------------------------------

@dataclass
class ModalityBuffer:
    """Bounded rolling buffer of the latest samples for one modality."""

    capacity: int
    columns: list[str]
    _rows: deque[list[Any]] = field(default_factory=lambda: deque(), init=False)

    def extend(self, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        for row in frame[self.columns].itertuples(index=False, name=None):
            if len(self._rows) >= self.capacity:
                self._rows.popleft()
            self._rows.append(list(row))

    def as_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(list(self._rows), columns=self.columns)

    def __len__(self) -> int:
        return len(self._rows)


# ---------------------------------------------------------------------------
# Session recorder
# ---------------------------------------------------------------------------

@dataclass
class SessionLog:
    """Collect per-cycle diagnostics to persist at the end of the run."""

    updates: list[dict[str, Any]] = field(default_factory=list)
    transitions: list[GuardrailTransition] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def add_update(
        self,
        update: InferenceUpdate,
        state: AlertState,
        rationale: str,
        calibration: Optional[CalibrationResult] = None,
    ) -> None:
        entry = {
            "timestamp_s": update.timestamp_s,
            "risk_score": update.risk_score,
            "model_confidence": update.model_confidence,
            "latency_s": update.latency_s,
            "fault_flags": list(update.fault_flags),
            "state": state.value,
            "rationale": rationale,
            "diagnostics": update.diagnostics,
        }
        if calibration is not None:
            entry["raw_risk"] = calibration.raw_risk
            entry["normalised_risk"] = calibration.normalised_risk
            entry["z_score"] = calibration.z_score
            entry["calibration_phase"] = calibration.phase
        self.updates.append(entry)

    def append_transition(self, transition: GuardrailTransition) -> None:
        self.transitions.append(transition)


# ---------------------------------------------------------------------------
# Inference model lazy-loader
# ---------------------------------------------------------------------------

def _load_models_or_stub(config: InferenceConfig) -> dict[str, Any]:
    """Load the real model bundle; fall back to an inert stub for dry runs.

    We support running the live app even before Day 4 models are available
    (e.g. in a fresh clone) by synthesising a benign "all features are
    equally important" linear model plus a tree stand-in that maps total
    feature magnitude to risk.  The stub keeps the pipeline flow-testable
    while signalling ``stub=True`` in the diagnostics.
    """

    try:
        models = load_inference_models(config)
        models["stub"] = False
        return models
    except (FileNotFoundError, InferenceError) as exc:
        LOGGER.warning(
            "Production models unavailable (%s) — running with built-in stub. "
            "Train Day 4 models for real risk scores.",
            exc,
        )
        return _make_stub_models()


class _StubProbModel:
    """Deterministic logistic-ish pseudo-model used when no bundle is present."""

    def __init__(self, bias: float = 0.0, gain: float = 0.8) -> None:
        self._bias = bias
        self._gain = gain

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # Collapse all features into a single risk proxy.
        score = np.tanh(self._gain * np.mean(x, axis=1) + self._bias)
        p1 = 0.5 + 0.5 * score
        p1 = np.clip(p1, 0.02, 0.98)
        return np.column_stack([1.0 - p1, p1])


def _make_stub_models() -> dict[str, Any]:
    stub_features = [
        "z_eeg_rel_theta", "z_eeg_rel_alpha", "z_eeg_rel_beta",
        "z_eeg_theta_alpha_ratio", "z_eeg_beta_alpha_ratio",
        "z_optical_trend_slope", "z_optical_mean_change",
        "z_imu_motion_score",
    ]
    return {
        "linear_model": _StubProbModel(bias=-0.2, gain=0.7),
        "tree_model": _StubProbModel(bias=0.0, gain=0.9),
        "fusion_weights": {"linear": 0.5, "tree": 0.5},
        "feature_columns": stub_features,
        "scaler_metadata": None,
        "selected_variant": "stub",
        "stub": True,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class LiveApp:
    """Coordinates the source, inference, guardrails and dashboard."""

    def __init__(
        self,
        config: LiveAppConfig,
        source: Optional[MuseDataSource] = None,
        inference_config: Optional[InferenceConfig] = None,
        guardrail_config: Optional[GuardrailConfig] = None,
        dashboard_config: Optional[DashboardConfig] = None,
    ) -> None:
        self.config = config
        self.source = source or build_data_source(
            source_kind=config.source_kind,
            session_id=config.session_id,
            device_address=config.device_address,
            preset=config.preset,
            raw_file=config.raw_file,
            manage_bridge=config.manage_bridge,
            real_time_factor=config.real_time_factor,
            fatigue_trend=config.fatigue_trend,
            record_raw_to=config.record_raw_to,
            resolve_timeout_s=config.resolve_timeout_s,
        )
        self.inference_config = inference_config or InferenceConfig()

        # Fix A: load guardrail thresholds from JSON if a path was supplied.
        if guardrail_config is not None:
            self.guardrail_config = guardrail_config
        elif config.guardrail_config_path:
            self.guardrail_config = GuardrailConfig.from_json(config.guardrail_config_path)
            LOGGER.info(
                "Loaded guardrail profile '%s' from %s",
                self.guardrail_config.profile_name or "(unnamed)",
                config.guardrail_config_path,
            )
        else:
            self.guardrail_config = GuardrailConfig()

        self.dashboard_config = dashboard_config or DashboardConfig()

        # Fix B: personal-baseline calibrator. Disabled when duration <= 0.
        self.calibrator = PersonalBaselineCalibrator(
            baseline_duration_s=config.baseline_duration_s,
            stdev_floor=config.baseline_stdev_floor,
            sigma_span=config.baseline_sigma_span,
            minimum_samples=config.baseline_minimum_samples,
        )
        if config.load_baseline_from:
            saved = PersonalBaselineCalibrator.load_baseline(config.load_baseline_from)
            self.calibrator.adopt_baseline(saved)
            LOGGER.info(
                "Reusing saved personal baseline from %s", config.load_baseline_from
            )

        self.models: dict[str, Any] = {}
        self.inference_state = InferenceState()
        self.guardrail_state = GuardrailState()

        self.eeg_buffer = ModalityBuffer(
            capacity=int(config.eeg_buffer_s * EEG_SFREQ_HZ * 1.2),
            columns=EEG_OUTPUT_COLUMNS,
        )
        self.optical_buffer = ModalityBuffer(
            capacity=int(config.optical_buffer_s * OPTICAL_SFREQ_HZ * 1.2),
            columns=OPTICAL_OUTPUT_COLUMNS,
        )
        self.imu_buffer = ModalityBuffer(
            capacity=int(config.imu_buffer_s * IMU_SFREQ_HZ * 1.2),
            columns=IMU_OUTPUT_COLUMNS,
        )

        self.log = SessionLog()
        self._renderer: Optional[DashboardRenderer] = None
        self._loop_start_time: float = 0.0
        self._running: bool = False

        # Dashboard accumulators
        self._ts_history: list[float] = []
        self._risk_history: list[float] = []        # personalised (or raw if disabled)
        self._raw_risk_history: list[float] = []    # always raw model output
        self._state_history: list[str] = []
        self._conf_history: list[float] = []
        self._eeg_theta_history: list[float] = []
        self._eeg_alpha_history: list[float] = []
        self._eeg_beta_history: list[float] = []
        self._optical_history: list[float] = []
        self._latency_history: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.models = _load_models_or_stub(self.inference_config)
        self.source.start()
        if self.config.enable_dashboard:
            self._renderer = DashboardRenderer(self.dashboard_config)
            self._renderer.initialise()
        self._loop_start_time = time.time()
        self._running = True
        LOGGER.info(
            "Live app started | source=%s | dashboard=%s | stub_models=%s",
            self.source.source_name,
            bool(self._renderer),
            self.models.get("stub", False),
        )

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self.source.stop()
        finally:
            if self._renderer is not None:
                self._renderer.close()

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def ingest(self, frames: MuseFrames) -> None:
        now = time.time() - self._loop_start_time
        if not frames.eeg.empty:
            self.eeg_buffer.extend(frames.eeg)
            self.inference_state.last_eeg_update_s = now
        if not frames.optical.empty:
            self.optical_buffer.extend(frames.optical)
            self.inference_state.last_optical_update_s = now
        if not frames.imu.empty:
            self.imu_buffer.extend(frames.imu)
            self.inference_state.last_imu_update_s = now

    # ------------------------------------------------------------------
    # Per-cycle
    # ------------------------------------------------------------------

    def run_cycle(self, timestamp_s: float) -> InferenceUpdate:
        update = run_inference_cycle(
            eeg_buffer=self.eeg_buffer.as_frame(),
            optical_buffer=self.optical_buffer.as_frame(),
            imu_buffer=self.imu_buffer.as_frame(),
            timestamp_s=timestamp_s,
            models=self.models,
            config=self.inference_config,
            state=self.inference_state,
        )

        # Fix B: personalise the raw risk against the user's baseline.
        calibration = self.calibrator.update(update.risk_score, timestamp_s)
        risk_for_guardrail = calibration.normalised_risk

        previous_state = self.guardrail_state.current_state
        if calibration.phase == "baseline":
            # Don't fire transitions during the calibration window: hold GREEN
            # so the dashboard can show "Calibrating..." without false alarms.
            new_state = self.guardrail_state.current_state
            self.guardrail_state.risk_history.append(risk_for_guardrail)
            self.guardrail_state.update_count += 1
            self.guardrail_state.state_duration += 1
        else:
            new_state = update_guardrail(
                risk_score=risk_for_guardrail,
                model_confidence=update.model_confidence,
                timestamp_s=timestamp_s,
                state=self.guardrail_state,
                config=self.guardrail_config,
            )

        rationale = ""
        if self.guardrail_state.transitions and (
            previous_state != new_state
            or (self.log.transitions and self.guardrail_state.transitions[-1]
                is not self.log.transitions[-1])
        ):
            transition = self.guardrail_state.transitions[-1]
            if transition not in self.log.transitions:
                self.log.append_transition(transition)
            rationale = transition.rationale

        self.log.add_update(update, new_state, rationale, calibration=calibration)
        self._accumulate_for_dashboard(update, new_state, calibration=calibration)
        return update

    def _accumulate_for_dashboard(
        self,
        update: InferenceUpdate,
        state: AlertState,
        calibration: Optional[CalibrationResult] = None,
    ) -> None:
        eeg_frame = self.eeg_buffer.as_frame()
        optical_frame = self.optical_buffer.as_frame()

        raw = update.feature_vector
        theta = raw.get("eeg_rel_theta", raw.get("z_eeg_rel_theta", 0.0))
        alpha = raw.get("eeg_rel_alpha", raw.get("z_eeg_rel_alpha", 0.0))
        beta  = raw.get("eeg_rel_beta",  raw.get("z_eeg_rel_beta", 0.0))

        if 0.0 <= abs(theta) <= 1.0 and 0.0 <= abs(alpha) <= 1.0:
            theta_val = float(np.clip(theta, 0.0, 1.0))
            alpha_val = float(np.clip(alpha, 0.0, 1.0))
            beta_val  = float(np.clip(beta, 0.0, 1.0))
        else:
            theta_val = alpha_val = beta_val = 0.0

        optical_trend = 0.0
        if not optical_frame.empty:
            opt_cols = [c for c in optical_frame.columns if c.startswith("optics_")]
            if opt_cols:
                optical_trend = float(np.tanh(optical_frame[opt_cols].mean(axis=1).mean() - 1.0))

        # Dashboard / artefact metrics use the *guardrail-facing* risk so the
        # gauge needle on screen matches the alert state. ``raw_risk_history``
        # stays untouched for diagnostics.
        risk_for_display = (
            calibration.normalised_risk if calibration is not None else update.risk_score
        )
        self._ts_history.append(update.timestamp_s)
        self._risk_history.append(risk_for_display)
        self._raw_risk_history.append(update.risk_score)
        self._state_history.append(state.value)
        self._conf_history.append(update.model_confidence)
        self._eeg_theta_history.append(theta_val)
        self._eeg_alpha_history.append(alpha_val)
        self._eeg_beta_history.append(beta_val)
        self._optical_history.append(optical_trend)
        self._latency_history.append(update.latency_s)

    def build_snapshot(self) -> DashboardSnapshot:
        health = self.source.health()
        faults = self.log.updates[-1]["fault_flags"] if self.log.updates else []

        def modality_health(name: str) -> str:
            if name in faults or f"{name}_stale" in faults or f"{name}_buffer_empty" in faults:
                return "FAIL"
            # If the source itself reports modality_healthy == False flag as WARN
            if health.get(f"{name}_healthy") is False:
                return "WARN"
            return "OK"

        elapsed = time.time() - self._loop_start_time

        # Calibration overlay parameters for the dashboard
        if self.calibrator.enabled and not self.calibrator.baseline_ready:
            phase = "baseline"
            if self.calibrator.baseline_duration_s > 0:
                progress = min(
                    1.0,
                    self.calibrator.samples_collected
                    / max(1.0, self.calibrator.baseline_duration_s
                          / max(self.config.step_s, 1e-3)),
                )
            else:
                progress = 1.0
        elif self.calibrator.enabled:
            phase = "active"
            progress = 1.0
        else:
            phase = "passthrough"
            progress = 1.0

        return DashboardSnapshot(
            timestamps=list(self._ts_history),
            eeg_theta=list(self._eeg_theta_history),
            eeg_alpha=list(self._eeg_alpha_history),
            eeg_beta=list(self._eeg_beta_history),
            optical_trend=list(self._optical_history),
            risk_scores=list(self._risk_history),
            alert_states=list(self._state_history),
            model_confidence=list(self._conf_history),
            transitions=[
                {
                    "timestamp_s": t.timestamp_s,
                    "new_state": t.new_state,
                    "previous_state": t.previous_state,
                    "rationale": t.rationale,
                }
                for t in self.guardrail_state.transitions
            ],
            eeg_health=modality_health("eeg"),
            optical_health=modality_health("optical"),
            imu_health=modality_health("imu"),
            model_health="STUB" if self.models.get("stub") else "OK",
            latency_s=self._latency_history[-1] if self._latency_history else 0.0,
            current_state=self.guardrail_state.current_state.value,
            current_risk=self._risk_history[-1] if self._risk_history else 0.0,
            current_confidence=self._conf_history[-1] if self._conf_history else 1.0,
            elapsed_s=elapsed,
            stage_name=self.source.source_name,
            calibration_phase=phase,
            raw_risk=self._raw_risk_history[-1] if self._raw_risk_history else 0.0,
            baseline_progress=progress,
            yellow_threshold=self.guardrail_config.yellow_threshold,
            red_threshold=self.guardrail_config.red_threshold,
            black_threshold=self.guardrail_config.black_threshold,
            guardrail_profile=self.guardrail_config.profile_name,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Drive the ingestion + inference loop until completion."""
        self.start()
        try:
            cycles = 0
            next_cycle_at = self._loop_start_time + self.config.step_s
            animation = None
            if self._renderer is not None and self.config.dashboard_block:
                # Run a non-blocking matplotlib loop alongside the inference
                # cadence.  ``plt.pause()`` yields control to the GUI.
                import matplotlib.pyplot as plt  # noqa: PLC0415
                plt.ion()

            while self._running:
                if self.config.max_inference_cycles is not None and cycles >= self.config.max_inference_cycles:
                    break
                if self.config.duration_s is not None and (time.time() - self._loop_start_time) >= self.config.duration_s:
                    break

                frames = self.source.read_frames()
                self.ingest(frames)

                # Detect source exhaustion (replay mode).
                if (
                    self.config.stop_on_finished_source
                    and hasattr(self.source, "finished")
                    and getattr(self.source, "finished")
                    and frames.empty
                ):
                    break

                now = time.time()
                if now >= next_cycle_at:
                    timestamp_s = now - self._loop_start_time
                    update = self.run_cycle(timestamp_s)
                    cycles += 1
                    next_cycle_at += self.config.step_s

                    if self._renderer is not None:
                        snapshot = self.build_snapshot()
                        self._renderer.render(snapshot)
                        if (
                            self.config.headless_save_every
                            and cycles % self.config.headless_save_every == 0
                            and self.config.write_outputs
                        ):
                            png_path = Path(self.config.output_dir) / f"dashboard_cycle_{cycles:04d}.png"
                            self._renderer.save_snapshot(snapshot, png_path)
                        if self.config.dashboard_block:
                            import matplotlib.pyplot as plt  # noqa: PLC0415
                            plt.pause(0.001)

                    LOGGER.debug(
                        "cycle=%d ts=%.2fs risk=%.3f state=%s conf=%.2f latency=%.3fs faults=%s",
                        cycles, timestamp_s, update.risk_score,
                        self.guardrail_state.current_state.value,
                        update.model_confidence, update.latency_s,
                        update.fault_flags or "none",
                    )

                time.sleep(self.config.poll_s)

            summary = self._finalise()
            return summary
        finally:
            self.stop()

    def _finalise(self) -> dict[str, Any]:
        elapsed = time.time() - self._loop_start_time
        mean_latency = float(np.mean(self._latency_history)) if self._latency_history else 0.0
        max_latency = float(np.max(self._latency_history)) if self._latency_history else 0.0
        summary = {
            "source": self.source.source_name,
            "session_id": self.config.session_id,
            "cycles": len(self._risk_history),
            "elapsed_s": elapsed,
            "final_state": self.guardrail_state.current_state.value,
            "mean_risk": float(np.mean(self._risk_history)) if self._risk_history else 0.0,
            "max_risk": float(np.max(self._risk_history)) if self._risk_history else 0.0,
            "mean_raw_risk": (
                float(np.mean(self._raw_risk_history)) if self._raw_risk_history else 0.0
            ),
            "max_raw_risk": (
                float(np.max(self._raw_risk_history)) if self._raw_risk_history else 0.0
            ),
            "mean_latency_s": mean_latency,
            "max_latency_s": max_latency,
            "within_latency_budget": max_latency <= self.inference_config.latency_budget_s,
            "transition_count": len(self.guardrail_state.transitions),
            "source_health": self.source.health(),
            "stub_models": bool(self.models.get("stub", False)),
            "guardrail_profile": {
                "name": self.guardrail_config.profile_name,
                "description": self.guardrail_config.profile_description,
                "thresholds": self.guardrail_config.to_dict(),
                "source_path": self.config.guardrail_config_path,
            },
            "calibration": self.calibrator.state_for_log(),
        }
        if self.config.write_outputs:
            self._write_artifacts(summary)
        return summary

    def _write_artifacts(self, summary: dict[str, Any]) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        updates_path = out_dir / f"{self.config.session_id}_updates.json"
        write_inference_diagnostics(
            [self._update_from_log_entry(entry) for entry in self.log.updates],
            updates_path,
        )

        transitions_path = out_dir / f"{self.config.session_id}_transitions.json"
        guardrail_summary_path = out_dir / f"{self.config.session_id}_guardrail_summary.json"
        write_guardrail_artifacts(
            self.guardrail_state,
            transition_log_path=transitions_path,
            summary_path=guardrail_summary_path,
        )

        summary_path = out_dir / f"{self.config.session_id}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        session_path = out_dir / f"{self.config.session_id}_session_log.csv"
        pd.DataFrame(self.log.updates).to_csv(session_path, index=False)

        baseline_path: Optional[Path] = None
        if self.calibrator.enabled and self.calibrator.baseline_ready:
            target = (
                Path(self.config.save_baseline_to)
                if self.config.save_baseline_to
                else out_dir / f"{self.config.session_id}_baseline.json"
            )
            baseline_path = self.calibrator.save(target)

        LOGGER.info("Live session artifacts written to %s", out_dir)
        summary["artifacts"] = {
            "updates": str(updates_path),
            "transitions": str(transitions_path),
            "guardrail_summary": str(guardrail_summary_path),
            "summary": str(summary_path),
            "session_log": str(session_path),
            "baseline": str(baseline_path) if baseline_path else None,
        }

    @staticmethod
    def _update_from_log_entry(entry: dict[str, Any]) -> InferenceUpdate:
        return InferenceUpdate(
            timestamp_s=float(entry["timestamp_s"]),
            risk_score=float(entry["risk_score"]),
            model_confidence=float(entry["model_confidence"]),
            feature_vector={},
            latency_s=float(entry["latency_s"]),
            fault_flags=list(entry.get("fault_flags", [])),
            diagnostics=dict(entry.get("diagnostics", {})),
        )


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_live_app(config: LiveAppConfig) -> dict[str, Any]:
    """Instantiate :class:`LiveApp` and run it to completion."""
    app = LiveApp(config=config)
    return app.run()


__all__ = [
    "LiveApp",
    "LiveAppConfig",
    "ModalityBuffer",
    "SessionLog",
    "run_live_app",
]
