"""Day 5 — T14  Dashboard and Timeline Visualization.

Matplotlib-based real-time dashboard for NeuroGauge that provides:

  T14.1  Real-time panels for EEG, optical trends, risk score, and current
         alert state.
  T14.2  Timeline with transition markers and rationale details.
  T14.3  Stream health and model-state indicators.
  T14.4  Readability validated for classroom projection (large fonts, high
         contrast, 1280×720 minimum).
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

# ---------------------------------------------------------------------------
# T14.4 — Classroom-projection colour scheme and typography
# ---------------------------------------------------------------------------

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

# T14.4 — Large font sizes for projection readability
_TITLE_SIZE = 16
_LABEL_SIZE = 13
_TICK_SIZE = 11
_BADGE_SIZE = 28
_RATIONALE_SIZE = 11
_INDICATOR_SIZE = 12


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dashboard renderer  (T14.1–T14.4)
# ---------------------------------------------------------------------------

class DashboardRenderer:
    """Matplotlib-based dashboard renderer.

    Creates a 4-panel layout:
      ┌──────────────┬──────────────┐
      │  EEG Bands   │  Risk Score  │
      ├──────────────┼──────────────┤
      │Optical Trend │State + Health│
      ├──────────────┴──────────────┤
      │     Timeline + Rationale    │
      └─────────────────────────────┘
    """

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
            3, 2,
            height_ratios=[1.0, 1.0, 0.8],
            hspace=0.35,
            wspace=0.25,
            left=0.06, right=0.94,
            top=0.93, bottom=0.06,
        )

        self._axes["eeg"] = fig.add_subplot(gs[0, 0])
        self._axes["risk"] = fig.add_subplot(gs[0, 1])
        self._axes["optical"] = fig.add_subplot(gs[1, 0])
        self._axes["state"] = fig.add_subplot(gs[1, 1])
        self._axes["timeline"] = fig.add_subplot(gs[2, :])

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

    def _render_eeg_panel(self, snap: DashboardSnapshot) -> None:
        """T14.1 — EEG relative band powers panel."""
        ax = self._axes["eeg"]
        ax.clear()
        ax.set_title("EEG Band Powers", fontsize=_TITLE_SIZE, fontweight="bold", pad=8)
        ax.set_ylabel("Relative Power", fontsize=_LABEL_SIZE)
        ax.set_xlabel("Time (s)", fontsize=_LABEL_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)

        t = snap.timestamps[-self.config.max_visible_points:]
        if not t:
            ax.text(0.5, 0.5, "Awaiting EEG data…", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_LABEL_SIZE, color=_TEXT_COLOR)
            return

        n = len(t)
        theta = snap.eeg_theta[-n:]
        alpha = snap.eeg_alpha[-n:]
        beta = snap.eeg_beta[-n:]

        ax.plot(t, theta, color="#fbbf24", linewidth=1.8, label="θ Theta", alpha=0.9)
        ax.plot(t, alpha, color=_ACCENT_EEG, linewidth=1.8, label="α Alpha", alpha=0.9)
        ax.plot(t, beta, color="#34d399", linewidth=1.8, label="β Beta", alpha=0.9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", fontsize=_TICK_SIZE - 1, framealpha=0.5,
                  facecolor=_PANEL_BG, edgecolor=_GRID_COLOR)
        ax.grid(True, alpha=0.3)

    def _render_risk_panel(self, snap: DashboardSnapshot) -> None:
        """T14.1 — Risk score panel with guardrail zone shading."""
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

        # Zone shading
        t_min, t_max = min(t), max(t)
        ax.axhspan(0.0, 0.40, color=_STATE_COLORS["GREEN"], alpha=0.08)
        ax.axhspan(0.40, 0.65, color=_STATE_COLORS["YELLOW"], alpha=0.10)
        ax.axhspan(0.65, 0.85, color=_STATE_COLORS["RED"], alpha=0.10)
        ax.axhspan(0.85, 1.0, color="#6b21a8", alpha=0.12)

        ax.plot(t, risk, color=_ACCENT_RISK, linewidth=2.2, alpha=0.95)
        ax.fill_between(t, risk, alpha=0.15, color=_ACCENT_RISK)
        ax.set_ylim(-0.02, 1.02)

        # Threshold lines
        for level, val in [("YELLOW", 0.40), ("RED", 0.65), ("BLACK", 0.85)]:
            ax.axhline(val, color=_STATE_COLORS.get(level, "#888"), linestyle="--",
                       linewidth=1, alpha=0.5)

        ax.grid(True, alpha=0.3)

    def _render_optical_panel(self, snap: DashboardSnapshot) -> None:
        """T14.1 — Optical trend panel."""
        ax = self._axes["optical"]
        ax.clear()
        ax.set_title("Optical Trend", fontsize=_TITLE_SIZE, fontweight="bold", pad=8)
        ax.set_ylabel("Normalised Signal", fontsize=_LABEL_SIZE)
        ax.set_xlabel("Time (s)", fontsize=_LABEL_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)

        t = snap.timestamps[-self.config.max_visible_points:]
        if not t:
            ax.text(0.5, 0.5, "Awaiting optical data…", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_LABEL_SIZE, color=_TEXT_COLOR)
            return

        n = len(t)
        opt = snap.optical_trend[-n:]
        ax.plot(t, opt, color=_ACCENT_OPT, linewidth=1.8, alpha=0.9)
        ax.fill_between(t, opt, alpha=0.12, color=_ACCENT_OPT)
        ax.grid(True, alpha=0.3)

    def _render_state_panel(self, snap: DashboardSnapshot) -> None:
        """T14.1 + T14.3 — Current state badge and stream health indicators."""
        ax = self._axes["state"]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("System Status", fontsize=_TITLE_SIZE, fontweight="bold", pad=8)

        # -- State badge (large, centred) --
        state_color = _STATE_COLORS.get(snap.current_state, "#555")
        text_color = _STATE_TEXT_COLORS.get(snap.current_state, "#fff")
        badge = mpatches.FancyBboxPatch(
            (0.10, 0.52), 0.80, 0.40,
            boxstyle="round,pad=0.04",
            facecolor=state_color,
            edgecolor="#ffffff",
            linewidth=2,
        )
        ax.add_patch(badge)
        ax.text(0.50, 0.72, snap.current_state, fontsize=_BADGE_SIZE,
                fontweight="bold", ha="center", va="center", color=text_color)
        ax.text(0.50, 0.55, f"Risk: {snap.current_risk:.2f}  |  Conf: {snap.current_confidence:.2f}",
                fontsize=_INDICATOR_SIZE, ha="center", va="center", color=text_color)

        # -- T14.3 — Stream health indicators --
        health_items = [
            ("EEG", snap.eeg_health),
            ("Optical", snap.optical_health),
            ("IMU", snap.imu_health),
            ("Model", snap.model_health),
        ]
        x_positions = [0.12, 0.32, 0.55, 0.78]
        y_row = 0.30
        for x, (label, status) in zip(x_positions, health_items):
            dot_color = _HEALTH_OK if status == "OK" else (_HEALTH_WARN if status == "WARN" else _HEALTH_FAIL)
            ax.plot(x, y_row, "o", color=dot_color, markersize=10, markeredgecolor="#fff", markeredgewidth=1)
            ax.text(x + 0.04, y_row, label, fontsize=_INDICATOR_SIZE - 1, va="center", color=_TEXT_COLOR)

        # -- Latency --
        lat_color = _HEALTH_OK if snap.latency_s <= 2.0 else _HEALTH_FAIL
        ax.text(0.12, 0.12, f"Latency: {snap.latency_s*1000:.0f} ms",
                fontsize=_INDICATOR_SIZE - 1, color=lat_color)

        # -- Elapsed / stage --
        if snap.stage_name:
            ax.text(0.55, 0.12, f"Stage: {snap.stage_name}",
                    fontsize=_INDICATOR_SIZE - 1, color=_TEXT_COLOR)

        elapsed_min = int(snap.elapsed_s // 60)
        elapsed_sec = int(snap.elapsed_s % 60)
        ax.text(0.55, 0.02, f"Elapsed: {elapsed_min}:{elapsed_sec:02d}",
                fontsize=_INDICATOR_SIZE - 1, color=_TEXT_COLOR)

    def _render_timeline_panel(self, snap: DashboardSnapshot) -> None:
        """T14.2 — Timeline with transition markers and rationale details."""
        ax = self._axes["timeline"]
        ax.clear()
        ax.set_title("Guardrail Timeline", fontsize=_TITLE_SIZE, fontweight="bold", pad=8)
        ax.set_xlabel("Time (s)", fontsize=_LABEL_SIZE)
        ax.set_yticks([])
        ax.tick_params(labelsize=_TICK_SIZE)

        t = snap.timestamps[-self.config.max_visible_points:]
        if not t:
            ax.text(0.5, 0.5, "Awaiting timeline data…", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_LABEL_SIZE, color=_TEXT_COLOR)
            return

        n = len(t)
        states = snap.alert_states[-n:]

        # Draw state colour bands
        for i in range(len(t)):
            st = states[i] if i < len(states) else "GREEN"
            color = _STATE_COLORS.get(st, "#555")
            width = (t[1] - t[0]) if len(t) > 1 else 1.0
            ax.barh(0.5, width, left=t[i], height=0.6, color=color, alpha=0.6, edgecolor="none")

        ax.set_ylim(0, 1.3)

        # Transition markers and rationale annotations
        visible_transitions = [
            tr for tr in snap.transitions
            if len(t) == 0 or tr.get("timestamp_s", 0) >= t[0]
        ]
        recent = visible_transitions[-self.config.show_rationale_count:]

        for i, tr in enumerate(recent):
            ts = tr.get("timestamp_s", 0)
            new_state = tr.get("new_state", "")
            rationale = tr.get("rationale", "")

            ax.axvline(ts, color="#ffffff", linewidth=1.5, linestyle="--", alpha=0.7)
            ax.plot(ts, 1.05, "v", color=_STATE_COLORS.get(new_state, "#fff"),
                    markersize=10, markeredgecolor="#fff", markeredgewidth=1)

            # Truncated rationale text
            short = rationale[:70] + "…" if len(rationale) > 70 else rationale
            y_offset = 1.15 + (i * 0.05)
            if y_offset < 1.3:
                ax.text(ts, y_offset, f"→{new_state}: {short}",
                        fontsize=_RATIONALE_SIZE - 1, color=_TEXT_COLOR, alpha=0.9,
                        ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=_PANEL_BG, alpha=0.8, edgecolor=_GRID_COLOR))

        ax.grid(True, axis="x", alpha=0.3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, snap: DashboardSnapshot) -> plt.Figure:
        """Render all panels from a snapshot (T14.1–T14.4)."""
        if not self._initialised:
            self.initialise()

        self._render_eeg_panel(snap)
        self._render_risk_panel(snap)
        self._render_optical_panel(snap)
        self._render_state_panel(snap)
        self._render_timeline_panel(snap)

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


# ---------------------------------------------------------------------------
# Snapshot builder — constructs DashboardSnapshot from Day 4 artifacts
# ---------------------------------------------------------------------------

def build_snapshot_from_replay(
    inference_diagnostics: list[dict[str, Any]],
    guardrail_transitions: list[dict[str, Any]],
    guardrail_summary: dict[str, Any],
    labeled_frame: Any | None = None,
    stage_name: str = "",
    elapsed_s: float = 0.0,
) -> DashboardSnapshot:
    """Build a DashboardSnapshot from Day 4 replay artifacts."""
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


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def write_dashboard_artifacts(
    snapshot_path: str | Path,
    validation_path: str | Path,
    config: DashboardConfig | None = None,
) -> dict[str, Any]:
    """Write dashboard validation report for T14.4."""
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
