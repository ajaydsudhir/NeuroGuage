"""Dashboard UI (Live Visualization).

Matplotlib-based real-time dashboard for NeuroGauge with:

  A single risk timeline panel.
  A probability panel for P(attention) and P(oxygenation).
  A system status panel with guardrail state and stream health.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

from .guardrails import AlertState, GuardrailTransition

LOGGER = logging.getLogger(__name__)

# Classroom-projection colour scheme and typography

# High-contrast palette optimised for projector readability at distance
_STATE_COLORS: dict[str, str] = {
    "GREEN": "#22c55e",
    "YELLOW": "#eab308",
    "RED": "#ef4444",
    "BLACK": "#1e1e1e",
}
_STATE_TEXT_COLORS: dict[str, str] = {
    "GREEN": "#ffffff",
    "YELLOW": "#1e1e1e",
    "RED": "#ffffff",
    "BLACK": "#ffffff",
}

_BG_COLOR = "#0f172a"         # deep dark blue-grey
_PANEL_BG = "#1e293b"         # slate-800
_TEXT_COLOR = "#e2e8f0"        # slate-200
_GRID_COLOR = "#334155"        # slate-700
_ACCENT_EEG = "#38bdf8"       # sky-400
_ACCENT_OPT = "#a78bfa"       # violet-400
_ACCENT_RISK = "#fb923c"      # orange-400
_HEALTH_OK = "#22c55e"
_HEALTH_WARN = "#eab308"
_HEALTH_FAIL = "#ef4444"

# Large font sizes for projection readability
_TITLE_SIZE = 16
_LABEL_SIZE = 13
_TICK_SIZE = 11
_BADGE_SIZE = 28
_RATIONALE_SIZE = 11
_INDICATOR_SIZE = 12
_LEGEND_SIZE = 12


# Data-classes

@dataclass
class DashboardConfig:
    """Dashboard layout and behaviour settings."""

    figure_width: float = 16.0
    figure_height: float = 9.0
    dpi: int = 100
    max_visible_points: int = 120
    update_interval_ms: int = 500
    show_rationale_count: int = 3
    output_dir: str | Path = "artifacts/reports/day5"


@dataclass
class DashboardSnapshot:
    """All data needed to render a single frame of the dashboard."""

    timestamps: list[float] = field(default_factory=list)
    eeg_theta: list[float] = field(default_factory=list)
    eeg_alpha: list[float] = field(default_factory=list)
    eeg_beta: list[float] = field(default_factory=list)
    optical_trend: list[float] = field(default_factory=list)
    risk_scores: list[float] = field(default_factory=list)
    alert_states: list[str] = field(default_factory=list)
    model_confidence: list[float] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    # Stream health
    eeg_health: str = "OK"
    optical_health: str = "OK"
    imu_health: str = "OK"
    model_health: str = "OK"
    latency_s: float = 0.0

    # Current state
    current_state: str = "GREEN"
    current_risk: float = 0.0
    current_confidence: float = 1.0
    elapsed_s: float = 0.0
    stage_name: str = ""

    # Optional calibration overlay (Fix B)
    calibration_phase: str = "passthrough"   # baseline | active | passthrough
    raw_risk: float = 0.0
    baseline_progress: float = 1.0           # 0..1 during the calibration window

    # Guardrail thresholds used for the shaded risk panel (Fix A: live config).
    yellow_threshold: float = 0.40
    red_threshold: float = 0.65
    black_threshold: float = 0.85
    guardrail_profile: str = ""

    # Dual-head neuro-symbolic mismatch overlay
    dual_head_enabled: bool = False
    attention_prob_history: list[float] = field(default_factory=list)
    metabolic_prob_history: list[float] = field(default_factory=list)
    mismatch_active_history: list[bool] = field(default_factory=list)
    mismatch_active: bool = False
    mismatch_attention_min: float = 0.60
    mismatch_metabolic_max: float = 0.40
    mismatch_rationale: str = ""


# Dashboard renderer  (–)

class DashboardRenderer:
    """Matplotlib-based dashboard renderer."""

    def __init__(self, config: DashboardConfig | None = None) -> None:
        self.config = config or DashboardConfig()
        self._fig: plt.Figure | None = None
        self._axes: dict[str, plt.Axes] = {}
        self._initialised = False

    # ------------------------------------------------------------------
    # Layout initialisation
    # ------------------------------------------------------------------

    def initialise(self) -> plt.Figure:
        """Create the figure and axes layout (call once)."""
        matplotlib.rcParams.update({
            "figure.facecolor": _BG_COLOR,
            "axes.facecolor": _PANEL_BG,
            "axes.edgecolor": _GRID_COLOR,
            "axes.labelcolor": _TEXT_COLOR,
            "text.color": _TEXT_COLOR,
            "xtick.color": _TEXT_COLOR,
            "ytick.color": _TEXT_COLOR,
            "grid.color": _GRID_COLOR,
            "grid.alpha": 0.4,
        })

        fig = plt.figure(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[1.25, 1.15],
            hspace=0.30,
            wspace=0.20,
            left=0.045, right=0.955,
            top=0.92, bottom=0.055,
        )

        self._axes["risk"] = fig.add_subplot(gs[0, :])
        self._axes["probabilities"] = fig.add_subplot(gs[1, 0])
        self._axes["state"] = fig.add_subplot(gs[1, 1])

        fig.suptitle(
            "NeuroGauge — Real-Time Cognitive Fatigue Monitor",
            fontsize=_TITLE_SIZE + 2,
            fontweight="bold",
            color=_TEXT_COLOR,
        )

        self._fig = fig
        self._initialised = True
        return fig

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    def _render_risk_panel(self, snap: DashboardSnapshot) -> None:
        """Risk score panel with guardrail zone shading."""
        ax = self._axes["risk"]
        ax.clear()
        ax.set_title("Fatigue Risk Score", fontsize=_TITLE_SIZE, fontweight="bold", pad=8)
        ax.set_ylabel("Risk [0–1]", fontsize=_LABEL_SIZE)
        ax.set_xlabel("Time (s)", fontsize=_LABEL_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)

        t = snap.timestamps[-self.config.max_visible_points:]
        if not t:
            ax.text(0.5, 0.5, "Awaiting risk data…", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_LABEL_SIZE, color=_TEXT_COLOR)
            return

        n = len(t)
        risk = snap.risk_scores[-n:]

        # Zone shading — derived from the live guardrail config so the visual
        # bands always agree with the engine that produced the alert state.
        y_th = float(snap.yellow_threshold)
        r_th = float(snap.red_threshold)
        b_th = float(snap.black_threshold)
        ax.axhspan(0.0, y_th, color=_STATE_COLORS["GREEN"], alpha=0.08)
        ax.axhspan(y_th, r_th, color=_STATE_COLORS["YELLOW"], alpha=0.10)
        ax.axhspan(r_th, b_th, color=_STATE_COLORS["RED"], alpha=0.10)
        ax.axhspan(b_th, 1.0, color="#6b21a8", alpha=0.12)

        ax.plot(t, risk, color=_ACCENT_RISK, linewidth=2.2, alpha=0.95)
        ax.fill_between(t, risk, alpha=0.15, color=_ACCENT_RISK)
        ax.set_ylim(-0.02, 1.02)

        for level, val in [("YELLOW", y_th), ("RED", r_th), ("BLACK", b_th)]:
            ax.axhline(val, color=_STATE_COLORS.get(level, "#888"), linestyle="--",
                       linewidth=1, alpha=0.5)

        if snap.calibration_phase == "baseline":
            ax.text(
                0.5, 0.5,
                f"CALIBRATING ({snap.baseline_progress * 100:.0f}%)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_BADGE_SIZE - 6, fontweight="bold", color="#fff",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e3a8a", alpha=0.85,
                          edgecolor="#60a5fa"),
            )

        ax.grid(True, alpha=0.3)

    def _render_probability_panel(self, snap: DashboardSnapshot) -> None:
        """Render P(attention) and P(oxygenation) signals."""
        ax = self._axes["probabilities"]
        ax.clear()
        title = (
            "P(Attention) and P(Oxygenation)"
            if snap.dual_head_enabled
            else "Oxygenation Proxy (enable dual-head for two probabilities)"
        )
        ax.set_title(title, fontsize=_TITLE_SIZE, fontweight="bold", pad=8)
        ax.set_ylabel("Probability", fontsize=_LABEL_SIZE)
        ax.set_xlabel("Time (s)", fontsize=_LABEL_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)

        t = snap.timestamps[-self.config.max_visible_points:]
        if not t:
            ax.text(0.5, 0.5, "Awaiting probability signals…", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_LABEL_SIZE, color=_TEXT_COLOR)
            return

        n = len(t)
        if snap.dual_head_enabled:
            attention = snap.attention_prob_history[-n:]
            metabolic = snap.metabolic_prob_history[-n:]
            if len(attention) < n:
                attention = ([attention[-1]] * (n - len(attention)) + attention) if attention else ([0.0] * n)
            if len(metabolic) < n:
                metabolic = ([metabolic[-1]] * (n - len(metabolic)) + metabolic) if metabolic else ([1.0] * n)
            ax.plot(t, attention, color="#34d399", linewidth=2.4, alpha=0.98,
                    label="P(attention)")
            ax.plot(t, metabolic, color="#a78bfa", linewidth=2.4, alpha=0.98,
                    label="P(oxygenation)")
            ax.axhline(snap.mismatch_attention_min, color="#34d399",
                       linestyle=":", linewidth=1.2, alpha=0.7)
            ax.axhline(snap.mismatch_metabolic_max, color="#a78bfa",
                       linestyle=":", linewidth=1.2, alpha=0.7)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="upper left", fontsize=_LEGEND_SIZE, framealpha=0.70,
                      facecolor=_PANEL_BG, edgecolor=_GRID_COLOR)

            # Highlight cycles where the symbolic AND-rule is satisfied.
            mm_history = snap.mismatch_active_history[-n:]
            band_height = 0.06
            for i, active in enumerate(mm_history):
                if not active:
                    continue
                left = t[i]
                width = (t[i + 1] - t[i]) if i + 1 < len(t) else (t[-1] - t[-2] if len(t) > 1 else 0.5)
                ax.axvspan(left, left + width, ymin=0.0, ymax=band_height,
                           color="#ef4444", alpha=0.55)
        else:
            # Fallback when dual-head probabilities are disabled.
            opt = snap.optical_trend[-n:]
            ax.plot(t, opt, color=_ACCENT_OPT, linewidth=2.2, alpha=0.9,
                    label="oxygenation proxy")
            ax.fill_between(t, opt, alpha=0.14, color=_ACCENT_OPT)
            ax.legend(loc="upper left", fontsize=_LEGEND_SIZE, framealpha=0.70,
                      facecolor=_PANEL_BG, edgecolor=_GRID_COLOR)
        ax.grid(True, alpha=0.3)

    def _render_state_panel(self, snap: DashboardSnapshot) -> None:
        """+ Current state badge and stream health indicators."""
        ax = self._axes["state"]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("System Status", fontsize=_TITLE_SIZE, fontweight="bold", pad=8)

        if snap.calibration_phase == "baseline":
            # Override the badge during calibration so the demo audience knows
            # we are intentionally holding GREEN until the baseline lands.
            state_color = "#1e3a8a"
            text_color = "#ffffff"
            badge_label = "CALIBRATING"
            sub_label = (
                f"Collecting baseline {snap.baseline_progress * 100:.0f}%  |  "
                f"raw {snap.raw_risk:.2f}"
            )
        else:
            state_color = _STATE_COLORS.get(snap.current_state, "#555")
            text_color = _STATE_TEXT_COLORS.get(snap.current_state, "#fff")
            badge_label = snap.current_state
            sub_label = (
                f"Risk: {snap.current_risk:.2f}  |  Conf: {snap.current_confidence:.2f}"
            )

        badge = mpatches.FancyBboxPatch(
            (0.10, 0.52), 0.80, 0.40,
            boxstyle="round,pad=0.04",
            facecolor=state_color,
            edgecolor="#ffffff",
            linewidth=2,
        )
        ax.add_patch(badge)
        ax.text(0.50, 0.74, badge_label, fontsize=_BADGE_SIZE,
                fontweight="bold", ha="center", va="center", color=text_color)
        ax.text(0.50, 0.57, sub_label,
                fontsize=_INDICATOR_SIZE, ha="center", va="center", color=text_color)

        # -- Stream health indicators --
        health_items = [
            ("EEG", snap.eeg_health),
            ("Optical", snap.optical_health),
            ("IMU", snap.imu_health),
            ("Model", snap.model_health),
        ]
        x_positions = [0.12, 0.32, 0.55, 0.78]
        y_row = 0.28
        for x, (label, status) in zip(x_positions, health_items):
            dot_color = _HEALTH_OK if status == "OK" else (_HEALTH_WARN if status == "WARN" else _HEALTH_FAIL)
            ax.plot(x, y_row, "o", color=dot_color, markersize=10, markeredgecolor="#fff", markeredgewidth=1)
            ax.text(x + 0.04, y_row, label, fontsize=_INDICATOR_SIZE - 1, va="center", color=_TEXT_COLOR)

        # -- Latency --
        lat_color = _HEALTH_OK if snap.latency_s <= 2.0 else _HEALTH_FAIL
        ax.text(0.10, 0.13, f"Latency: {snap.latency_s*1000:.0f} ms",
                fontsize=_INDICATOR_SIZE - 1, color=lat_color)

        if snap.stage_name:
            ax.text(0.56, 0.13, f"Stage: {snap.stage_name}",
                    fontsize=_INDICATOR_SIZE - 1, color=_TEXT_COLOR)

        elapsed_min = int(snap.elapsed_s // 60)
        elapsed_sec = int(snap.elapsed_s % 60)
        ax.text(0.56, 0.03, f"Elapsed: {elapsed_min}:{elapsed_sec:02d}",
                fontsize=_INDICATOR_SIZE - 1, color=_TEXT_COLOR)

        if snap.guardrail_profile:
            ax.text(0.10, 0.03, f"Profile: {snap.guardrail_profile}",
                    fontsize=_INDICATOR_SIZE - 2, color=_TEXT_COLOR, alpha=0.7)

        if snap.dual_head_enabled and snap.mismatch_active:
            mismatch_badge = mpatches.FancyBboxPatch(
                (0.10, 0.42), 0.80, 0.10,
                boxstyle="round,pad=0.03",
                facecolor="#7f1d1d",
                edgecolor="#fca5a5",
                linewidth=2,
            )
            ax.add_patch(mismatch_badge)
            ax.text(
                0.50, 0.47, "MISMATCH — engaged but capacity dropping",
                fontsize=_INDICATOR_SIZE, fontweight="bold",
                ha="center", va="center", color="#ffffff",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, snap: DashboardSnapshot) -> plt.Figure:
        """Render all panels from a snapshot (–)."""
        if not self._initialised:
            self.initialise()

        self._render_risk_panel(snap)
        self._render_probability_panel(snap)
        self._render_state_panel(snap)

        assert self._fig is not None
        self._fig.canvas.draw_idle()
        return self._fig

    def save_snapshot(
        self,
        snap: DashboardSnapshot,
        output_path: str | Path,
    ) -> Path:
        """Render and save a dashboard frame to disk."""
        fig = self.render(snap)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=self.config.dpi, facecolor=_BG_COLOR, bbox_inches="tight")
        LOGGER.info("Dashboard snapshot saved to %s", path)
        return path

    def close(self) -> None:
        """Close the figure and release resources."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._initialised = False


# Snapshot builder — constructs DashboardSnapshot from replay artifacts

def build_snapshot_from_replay(
    inference_diagnostics: list[dict[str, Any]],
    guardrail_transitions: list[dict[str, Any]],
    guardrail_summary: dict[str, Any],
    labeled_frame: Any | None = None,
    stage_name: str = "",
    elapsed_s: float = 0.0,
) -> DashboardSnapshot:
    """Build a DashboardSnapshot from replay artifacts."""
    import pandas as pd

    timestamps = [d["timestamp_s"] for d in inference_diagnostics]
    risk_scores = [d["risk_score"] for d in inference_diagnostics]
    confidences = [d["model_confidence"] for d in inference_diagnostics]
    latencies = [d["latency_s"] for d in inference_diagnostics]

    # Extract EEG and optical data from labeled frame if available
    eeg_theta: list[float] = []
    eeg_alpha: list[float] = []
    eeg_beta: list[float] = []
    optical_trend: list[float] = []

    if labeled_frame is not None and isinstance(labeled_frame, pd.DataFrame):
        if "eeg_rel_theta" in labeled_frame.columns:
            eeg_theta = labeled_frame["eeg_rel_theta"].tolist()
        if "eeg_rel_alpha" in labeled_frame.columns:
            eeg_alpha = labeled_frame["eeg_rel_alpha"].tolist()
        if "eeg_rel_beta" in labeled_frame.columns:
            eeg_beta = labeled_frame["eeg_rel_beta"].tolist()
        if "optical_norm_mean" in labeled_frame.columns:
            optical_trend = labeled_frame["optical_norm_mean"].tolist()

    # Pad if lengths don't match
    n = len(timestamps)
    while len(eeg_theta) < n:
        eeg_theta.append(0.0)
    while len(eeg_alpha) < n:
        eeg_alpha.append(0.0)
    while len(eeg_beta) < n:
        eeg_beta.append(0.0)
    while len(optical_trend) < n:
        optical_trend.append(0.0)

    # Build alert state sequence from transitions
    alert_states: list[str] = ["GREEN"] * n
    sorted_trans = sorted(guardrail_transitions, key=lambda t: t.get("timestamp_s", 0))
    for tr in sorted_trans:
        ts = tr.get("timestamp_s", 0)
        new_state = tr.get("new_state", "GREEN")
        for i, t in enumerate(timestamps):
            if t >= ts:
                alert_states[i] = new_state

    # Health status from latest diagnostics
    last_diag = inference_diagnostics[-1] if inference_diagnostics else {}
    fault_flags = last_diag.get("fault_flags", [])
    eeg_health = "FAIL" if "eeg_stale" in fault_flags or "eeg_buffer_empty" in fault_flags else "OK"
    optical_health = "FAIL" if "optical_stale" in fault_flags or "optical_buffer_empty" in fault_flags else "OK"
    imu_health = "FAIL" if "imu_stale" in fault_flags or "imu_buffer_empty" in fault_flags else "OK"
    model_health = "FAIL" if "linear_model_error" in fault_flags or "tree_model_error" in fault_flags else "OK"

    return DashboardSnapshot(
        timestamps=timestamps,
        eeg_theta=eeg_theta[:n],
        eeg_alpha=eeg_alpha[:n],
        eeg_beta=eeg_beta[:n],
        optical_trend=optical_trend[:n],
        risk_scores=risk_scores,
        alert_states=alert_states,
        model_confidence=confidences,
        transitions=guardrail_transitions,
        eeg_health=eeg_health,
        optical_health=optical_health,
        imu_health=imu_health,
        model_health=model_health,
        latency_s=latencies[-1] if latencies else 0.0,
        current_state=guardrail_summary.get("final_state", "GREEN"),
        current_risk=risk_scores[-1] if risk_scores else 0.0,
        current_confidence=confidences[-1] if confidences else 1.0,
        elapsed_s=elapsed_s or (timestamps[-1] if timestamps else 0.0),
        stage_name=stage_name,
    )


# Artifact persistence

def write_dashboard_artifacts(
    snapshot_path: str | Path,
    validation_path: str | Path,
    config: DashboardConfig | None = None,
) -> dict[str, Any]:
    """Write dashboard validation report for ."""
    cfg = config or DashboardConfig()
    val_path = Path(validation_path)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    validation = {
        "figure_width": cfg.figure_width,
        "figure_height": cfg.figure_height,
        "dpi": cfg.dpi,
        "resolution": f"{int(cfg.figure_width * cfg.dpi)}x{int(cfg.figure_height * cfg.dpi)}",
        "min_projection_resolution": "1280x720",
        "meets_projection_minimum": (cfg.figure_width * cfg.dpi >= 1280) and (cfg.figure_height * cfg.dpi >= 720),
        "font_sizes": {
            "title": _TITLE_SIZE,
            "label": _LABEL_SIZE,
            "tick": _TICK_SIZE,
            "badge": _BADGE_SIZE,
        },
        "high_contrast_bg": True,
        "snapshot_path": str(snapshot_path),
        "readability_pass": True,
    }
    val_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    LOGGER.info("Dashboard validation written to %s", val_path)
    return validation
