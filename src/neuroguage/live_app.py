"""Main live loop for NeuroGauge.

This file wires together stream input, inference, guardrails, and dashboard
updates. It keeps short rolling buffers for each sensor modality, runs one
inference step every cycle, and writes session artifacts when the run ends.
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

from .dashboard import (
    DashboardConfig,
    DashboardRenderer,
    DashboardSnapshot,
)
from .dual_head import (
    EEG_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
)
from .guardrails import (
    AlertState,
    GuardrailConfig,
    GuardrailState,
    GuardrailTransition,
    MismatchEvent,
    MismatchState,
    evaluate_metabolic_mismatch,
    update_guardrail,
    write_guardrail_artifacts,
    write_mismatch_artifacts,
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


# Configuration

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
    # If set, we load GuardrailConfig.from_json(...) at startup. ``None``
    # means "use built-in defaults" for compatibility with older runs.
    guardrail_config_path: Optional[str] = None

    # ---- Fix B: personal-baseline z-score calibration -----------------
    # ``baseline_duration_s = 0`` disables calibration for compatibility.
    baseline_duration_s: float = 0.0
    baseline_stdev_floor: float = 0.04
    baseline_sigma_span: float = 4.0
    baseline_minimum_samples: int = 5
    save_baseline_to: Optional[str] = None      # auto-derived if None and enabled
    load_baseline_from: Optional[str] = None    # skip baseline phase, use saved one

    # ---- Dual-head neuro-symbolic mismatch detector --------------------
    # When enabled, an additional PyTorch model predicts P(attention) and
    # P(metabolic_ready) every cycle and a symbolic AND-rule fires the
    # MISMATCH alert when the operator is engaged but their hemodynamic
    # readiness drops. Defaults to off for compatibility.
    use_dual_head: bool = False
    dual_head_weights_path: Optional[str] = "artifacts/models/dual_head/dual_head.pt"
    dual_head_config_path: Optional[str] = "artifacts/models/dual_head/config.json"
    dual_head_device: str = "cpu"
    dual_head_attention_ema_alpha: float = 0.30
    dual_head_metabolic_ema_alpha: float = 0.30
    adaptive_mismatch_thresholds: bool = False
    adaptive_mismatch_warmup_cycles: int = 48
    adaptive_attention_quantile: float = 0.65
    adaptive_metabolic_quantile: float = 0.35


# Ring buffers

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


# Session recorder

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
        dual_head: Optional[dict[str, float]] = None,
        mismatch_active: Optional[bool] = None,
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
        if dual_head is not None:
            entry["attention_prob"] = float(dual_head.get("attention_prob", 0.0))
            entry["metabolic_prob"] = float(dual_head.get("metabolic_prob", 1.0))
            entry["mismatch_prob"] = float(dual_head.get("mismatch_prob", 0.0))
        if mismatch_active is not None:
            entry["mismatch_active"] = bool(mismatch_active)
        self.updates.append(entry)

    def append_transition(self, transition: GuardrailTransition) -> None:
        self.transitions.append(transition)


# Inference model lazy-loader

def _load_models_or_stub(config: InferenceConfig) -> dict[str, Any]:
    """Load the real model bundle; fall back to an inert stub for dry runs.

    We support running the live app even before baseline models are available
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
            "Train baseline models for real risk scores.",
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


# Main loop

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
        self._base_mismatch_attention_min = self.guardrail_config.mismatch_attention_min
        self._base_mismatch_metabolic_max = self.guardrail_config.mismatch_metabolic_max

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
        self.mismatch_state = MismatchState()

        # Dual-head neuro-symbolic model (loaded lazily in start()).
        self.dual_head_model: Any = None
        self.dual_head_variant: str = "muse_demo"
        self.dual_head_eeg_features: list[str] = list(EEG_FEATURE_COLUMNS)
        self.dual_head_optical_features: list[str] = list(OPTICAL_FEATURE_COLUMNS)
        self._att_ema: Optional[float] = None
        self._meta_ema: Optional[float] = None
        self._adaptive_thresholds_committed: bool = False

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
        self._attention_history: list[float] = []
        self._metabolic_history: list[float] = []
        self._mismatch_active_history: list[bool] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _maybe_load_dual_head(self) -> None:
        """Load the dual-head PyTorch model if enabled and the artifacts exist."""
        if not self.config.use_dual_head:
            return
        weights = self.config.dual_head_weights_path
        cfg_path = self.config.dual_head_config_path
        if not weights or not cfg_path:
            LOGGER.warning(
                "use_dual_head=True but no weights/config path supplied — disabling."
            )
            return
        if not Path(weights).exists() or not Path(cfg_path).exists():
            LOGGER.warning(
                "Dual-head artifacts missing (%s / %s) — train via "
                "`python -m neuroguage.cli train-dual-head`. Disabling.",
                weights, cfg_path,
            )
            return
        try:
            from .dual_head import load_dual_head_model  # noqa: PLC0415
            model, payload = load_dual_head_model(
                weights_path=weights,
                config_path=cfg_path,
                device=self.config.dual_head_device,
            )
            self.dual_head_model = model
            self.dual_head_variant = str(payload.get("model_variant", "muse_demo"))
            self.dual_head_eeg_features = list(payload.get("eeg_features", EEG_FEATURE_COLUMNS))
            self.dual_head_optical_features = list(
                payload.get("optical_features", OPTICAL_FEATURE_COLUMNS)
            )
            LOGGER.info(
                "Dual-head model loaded (eeg_features=%d, optical_features=%d)",
                len(self.dual_head_eeg_features),
                len(self.dual_head_optical_features),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Dual-head load failed (%s) — disabling.", exc)
            self.dual_head_model = None

    def start(self) -> None:
        self.models = _load_models_or_stub(self.inference_config)
        self._maybe_load_dual_head()
        self.source.start()
        if self.config.enable_dashboard:
            self._renderer = DashboardRenderer(self.dashboard_config)
            self._renderer.initialise()
        self._loop_start_time = time.time()
        self._running = True
        LOGGER.info(
            "Live app started | source=%s | dashboard=%s | stub_models=%s | dual_head=%s",
            self.source.source_name,
            bool(self._renderer),
            self.models.get("stub", False),
            bool(self.dual_head_model),
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

    def _run_dual_head_cycle(
        self,
        update: InferenceUpdate,
        timestamp_s: float,
        calibrated_risk: float,
    ) -> Optional[dict[str, float]]:
        """Run the dual-head + symbolic mismatch rule for this cycle.

        Returns ``None`` when the dual-head is disabled or unavailable.
        Otherwise returns the head probabilities and updates
        ``self.mismatch_state`` in place.
        """
        if self.dual_head_model is None:
            return None

        feature_vector = update.feature_vector or {}
        eeg_vec = [float(feature_vector.get(c, 0.0)) for c in self.dual_head_eeg_features]
        opt_vec = [float(feature_vector.get(c, 0.0)) for c in self.dual_head_optical_features]

        from .dual_head import predict_dual_head  # noqa: PLC0415

        outputs = predict_dual_head(
            self.dual_head_model,
            eeg_feature_vector=eeg_vec,
            optical_feature_vector=opt_vec,
            device=self.config.dual_head_device,
        )
        # Temporal stabilisation for less jittery per-window classifications.
        a_alpha = float(np.clip(self.config.dual_head_attention_ema_alpha, 0.0, 1.0))
        m_alpha = float(np.clip(self.config.dual_head_metabolic_ema_alpha, 0.0, 1.0))
        att_raw = float(outputs["attention_prob"])
        meta_raw = float(outputs["metabolic_prob"])
        if self._att_ema is None:
            self._att_ema = att_raw
        else:
            self._att_ema = a_alpha * att_raw + (1.0 - a_alpha) * self._att_ema
        if self._meta_ema is None:
            self._meta_ema = meta_raw
        else:
            self._meta_ema = m_alpha * meta_raw + (1.0 - m_alpha) * self._meta_ema
        outputs["attention_prob"] = float(np.clip(self._att_ema, 0.0, 1.0))
        outputs["metabolic_prob"] = float(np.clip(self._meta_ema, 0.0, 1.0))
        outputs["mismatch_prob"] = float(
            np.clip(outputs["attention_prob"] * (1.0 - outputs["metabolic_prob"]), 0.0, 1.0)
        )

        # Subject-adaptive mismatch thresholds after warmup.
        if (
            self.config.adaptive_mismatch_thresholds
            and not self._adaptive_thresholds_committed
            and len(self._attention_history) >= self.config.adaptive_mismatch_warmup_cycles
            and len(self._metabolic_history) >= self.config.adaptive_mismatch_warmup_cycles
        ):
            att_q = float(np.quantile(self._attention_history, self.config.adaptive_attention_quantile))
            meta_q = float(np.quantile(self._metabolic_history, self.config.adaptive_metabolic_quantile))
            att_blend = 0.8 * self._base_mismatch_attention_min + 0.2 * att_q
            meta_blend = 0.8 * self._base_mismatch_metabolic_max + 0.2 * meta_q
            self.guardrail_config.mismatch_attention_min = float(
                np.clip(
                    att_blend,
                    max(0.0, self._base_mismatch_attention_min - 0.04),
                    min(1.0, self._base_mismatch_attention_min + 0.06),
                )
            )
            self.guardrail_config.mismatch_metabolic_max = float(
                np.clip(
                    meta_blend,
                    max(0.0, self._base_mismatch_metabolic_max - 0.06),
                    min(1.0, self._base_mismatch_metabolic_max + 0.04),
                )
            )
            self._adaptive_thresholds_committed = True
            LOGGER.info(
                "Adaptive mismatch thresholds committed | attention_min=%.3f metabolic_max=%.3f",
                self.guardrail_config.mismatch_attention_min,
                self.guardrail_config.mismatch_metabolic_max,
            )

        evaluate_metabolic_mismatch(
            attention_prob=outputs["attention_prob"],
            metabolic_prob=outputs["metabolic_prob"],
            timestamp_s=timestamp_s,
            state=self.mismatch_state,
            risk_score=calibrated_risk,
            config=self.guardrail_config,
        )
        return outputs

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

        # Dual-head + symbolic mismatch rule (only when calibration is past
        # the baseline phase, to avoid spurious flags before we know what
        # the user's resting hemodynamics look like).
        dual_head_outputs: Optional[dict[str, float]] = None
        if calibration.phase != "baseline":
            dual_head_outputs = self._run_dual_head_cycle(
                update,
                timestamp_s,
                risk_for_guardrail,
            )

        mismatch_active_flag = (
            self.mismatch_state.active if self.dual_head_model is not None else None
        )

        self.log.add_update(
            update,
            new_state,
            rationale,
            calibration=calibration,
            dual_head=dual_head_outputs,
            mismatch_active=mismatch_active_flag,
        )
        self._accumulate_for_dashboard(
            update,
            new_state,
            calibration=calibration,
            dual_head=dual_head_outputs,
            mismatch_active=mismatch_active_flag,
        )
        return update

    def _accumulate_for_dashboard(
        self,
        update: InferenceUpdate,
        state: AlertState,
        calibration: Optional[CalibrationResult] = None,
        dual_head: Optional[dict[str, float]] = None,
        mismatch_active: Optional[bool] = None,
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

        # Dual-head telemetry: persist the most recent probabilities even when
        # the head wasn't run this cycle (e.g. during baseline) so the
        # dashboard timeline stays continuous.
        if dual_head is not None:
            self._attention_history.append(float(dual_head.get("attention_prob", 0.0)))
            self._metabolic_history.append(float(dual_head.get("metabolic_prob", 1.0)))
        elif self._attention_history:
            self._attention_history.append(self._attention_history[-1])
            self._metabolic_history.append(self._metabolic_history[-1])
        else:
            self._attention_history.append(0.0)
            self._metabolic_history.append(1.0)
        self._mismatch_active_history.append(bool(mismatch_active) if mismatch_active else False)

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
            dual_head_enabled=self.dual_head_model is not None,
            attention_prob_history=list(self._attention_history),
            metabolic_prob_history=list(self._metabolic_history),
            mismatch_active_history=list(self._mismatch_active_history),
            mismatch_active=bool(self.mismatch_state.active),
            mismatch_attention_min=self.guardrail_config.mismatch_attention_min,
            mismatch_metabolic_max=self.guardrail_config.mismatch_metabolic_max,
            mismatch_rationale=(
                self.mismatch_state.events[-1].rationale
                if self.mismatch_state.events and self.mismatch_state.events[-1].active
                else ""
            ),
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
            "dual_head": self._dual_head_summary(),
        }
        if self.config.write_outputs:
            self._write_artifacts(summary)
        return summary

    def _dual_head_summary(self) -> dict[str, Any]:
        """Compact summary of dual-head behaviour for the session report."""
        if self.dual_head_model is None:
            return {"enabled": False}

        active_total = sum(1 for v in self._mismatch_active_history if v)
        return {
            "enabled": True,
            "pipeline_stages": {
                "stage_1_attention": True,
                "stage_2_fatigue_related_metabolic": True,
                "stage_3_symbolic_intervention": True,
            },
            "weights_path": self.config.dual_head_weights_path,
            "config_path": self.config.dual_head_config_path,
            "model_variant": self.dual_head_variant,
            "mismatch_events": len(self.mismatch_state.events),
            "mismatch_active_cycles": int(active_total),
            "mismatch_active_fraction": (
                float(active_total) / len(self._mismatch_active_history)
                if self._mismatch_active_history
                else 0.0
            ),
            "mean_attention_prob": (
                float(np.mean(self._attention_history)) if self._attention_history else 0.0
            ),
            "mean_metabolic_prob": (
                float(np.mean(self._metabolic_history)) if self._metabolic_history else 0.0
            ),
            "thresholds": {
                "attention_min": self.guardrail_config.mismatch_attention_min,
                "metabolic_max": self.guardrail_config.mismatch_metabolic_max,
                "persistence": self.guardrail_config.mismatch_persistence,
            },
        }

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

        mismatch_path: Optional[Path] = None
        if self.dual_head_model is not None:
            mismatch_path = out_dir / f"{self.config.session_id}_mismatch_events.json"
            write_mismatch_artifacts(self.mismatch_state, mismatch_path)

        LOGGER.info("Live session artifacts written to %s", out_dir)
        summary["artifacts"] = {
            "updates": str(updates_path),
            "transitions": str(transitions_path),
            "guardrail_summary": str(guardrail_summary_path),
            "summary": str(summary_path),
            "session_log": str(session_path),
            "baseline": str(baseline_path) if baseline_path else None,
            "mismatch_events": str(mismatch_path) if mismatch_path else None,
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


# Convenience entry point

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


# --- Consolidated from calibration.py ---
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
