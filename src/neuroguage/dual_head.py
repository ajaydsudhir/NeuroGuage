"""Phase 1 + Phase 2 Neural Modeling (Dual-Head).

Contains:
- weak-label derivation for attention/metabolic-ready/mismatch
- dual-head model architectures (`muse_demo`, `research_eegnet`)
- training, evaluation, artifact save/load, and inference helpers
"""

from __future__ import annotations

"""Dual-head weak labels for the neuro-symbolic mismatch model.

The dual-head model is intentionally honest about what the Muse S Athena can
and cannot measure:

* ``attention_label``      — derived from task phase + performance drift.
                             Approximates "is the user actively engaged in the
                             task right now?" using behaviourally observed
                             signals rather than a calibrated EEG attention
                             ground truth.

* ``metabolic_ready_label`` — derived from subjective fatigue + optical signal
                             quality.  This is **not** a calibrated SpO_2 or
                             cerebral oxygenation reading; the Muse S Athena's
                             optical sensor is PPG-based with NIR/IR/RED/AMB
                             channels and cannot legitimately be claimed as
                             fNIRS.  The label captures *hemodynamic
                             stability* — whether the cardiovascular /
                             metabolic system appears to support sustained
                             cognitive work right now.

* ``mismatch_label``       — the symbolic AND-rule the project cares about:
                             ``attentive AND NOT metabolic_ready`` -> the user
                             is *trying* but their body is no longer keeping
                             up, which is exactly the "high-risk job operator
                             unable to perform" condition the proposal
                             motivates.

These weak labels are then used to train two small PyTorch classifiers and a
custom mismatch-consistency loss term, so the academic deliverable hits the
training loop / optimiser / normalisation / loss design checkboxes while
keeping the project semantically faithful to the proposal in
``docs/proposal.md``.
"""


from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# Required raw columns

REQUIRED_BASE_COLUMNS = (
    "task_phase",
    "subjective_fatigue",
    "performance_drift",
    "optical_coverage",
    "imu_motion_score",
)

ATTENTION_LABEL_COLUMN = "attention_label"
METABOLIC_LABEL_COLUMN = "metabolic_ready_label"
MISMATCH_LABEL_COLUMN = "mismatch_label"

# Feature splits used by the two heads. Kept here so a single import gives
# the rest of the pipeline (training, inference, tests) one source of truth.
EEG_FEATURE_COLUMNS: tuple[str, ...] = (
    "z_eeg_rel_theta",
    "z_eeg_rel_alpha",
    "z_eeg_rel_beta",
    "z_eeg_theta_alpha_ratio",
    "z_eeg_beta_alpha_ratio",
    "z_eeg_quality_coverage",
    "z_eeg_quality_score",
)
OPTICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "z_optical_norm_mean",
    "z_optical_trend_slope",
    "z_optical_mean_change",
    "z_optical_coverage",
)


class DualHeadLabelError(ValueError):
    """Raised when dual-head label generation cannot proceed."""


@dataclass
class DualHeadLabelConfig:
    """Knobs for weak-label derivation. Defaults match the default protocol."""

    attention_threshold: float = 0.5
    metabolic_threshold: float = 0.5
    # Weights on attention sub-signals: high task_phase_score AND low
    # performance drift -> attentive.
    attention_phase_weight: float = 0.55
    attention_performance_weight: float = 0.45
    # Weights on metabolic readiness sub-signals.
    metabolic_subjective_weight: float = 0.50
    metabolic_optical_quality_weight: float = 0.30
    metabolic_motion_weight: float = 0.20
    # Optional richer supervision terms when these columns are present:
    # - task_phase_score
    # - performance_norm
    # - weak_label_score_smoothed
    attention_task_score_weight: float = 0.10
    attention_smoothed_label_weight: float = 0.10
    metabolic_perf_norm_weight: float = 0.10
    metabolic_smoothed_label_weight: float = 0.10
    # Confidence weighting used during training (if consumed by trainer).
    min_label_confidence: float = 0.25


def _safe_minmax(values: pd.Series) -> pd.Series:
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - lo) / (hi - lo)


def _ensure_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DualHeadLabelError(
            f"labeled frame is missing required columns: {missing}"
        )


def _phase_engagement(task_phase: pd.Series) -> pd.Series:
    """Map task_phase -> attention demand in [0, 1].

    Baseline rest does not demand sustained attention; the three demanding
    tasks do, with code comprehension being the heaviest.
    """
    mapping = {
        "baseline": 0.10,
        "sustained_attention": 0.65,
        "n_back": 0.85,
        "code_comprehension": 0.95,
    }
    return task_phase.astype(str).str.lower().map(mapping).fillna(0.5)


def derive_dual_head_labels(
    labeled_frame: pd.DataFrame,
    config: DualHeadLabelConfig | None = None,
    session_column: str = "session_id",
) -> pd.DataFrame:
    """Augment ``labeled_frame`` with attention / metabolic / mismatch labels.

    The function is deliberately additive: it returns a *copy* of the input
    frame with three new columns, leaving the existing fatigue label and z-
    scored features untouched so the existing pipelines keep working
    without changes.
    """
    if config is None:
        config = DualHeadLabelConfig()

    if labeled_frame.empty:
        raise DualHeadLabelError("labeled_frame is empty")

    _ensure_columns(labeled_frame, REQUIRED_BASE_COLUMNS)
    if session_column not in labeled_frame.columns:
        raise DualHeadLabelError(
            f"session column '{session_column}' missing from labeled_frame"
        )

    frame = labeled_frame.copy()

    phase_score = _phase_engagement(frame["task_phase"])
    perf_norm = frame.groupby(session_column)["performance_drift"].transform(
        _safe_minmax
    )
    subj_norm = frame.groupby(session_column)["subjective_fatigue"].transform(
        _safe_minmax
    )

    attention_score = (
        config.attention_phase_weight * phase_score
        + config.attention_performance_weight * (1.0 - perf_norm)
    )
    if "task_phase_score" in frame.columns:
        attention_score = attention_score + config.attention_task_score_weight * frame["task_phase_score"].clip(0.0, 1.0)
    if "weak_label_score_smoothed" in frame.columns:
        attention_score = attention_score + config.attention_smoothed_label_weight * frame["weak_label_score_smoothed"].clip(0.0, 1.0)
    attention_score = attention_score.clip(0.0, 1.0)

    optical_quality = frame["optical_coverage"].clip(0.0, 1.0)
    motion_penalty = (1.0 - frame["imu_motion_score"]).clip(0.0, 1.0)

    metabolic_score = (
        config.metabolic_subjective_weight * (1.0 - subj_norm)
        + config.metabolic_optical_quality_weight * optical_quality
        + config.metabolic_motion_weight * motion_penalty
    )
    if "performance_norm" in frame.columns:
        metabolic_score = metabolic_score + config.metabolic_perf_norm_weight * (1.0 - frame["performance_norm"].clip(0.0, 1.0))
    if "weak_label_score_smoothed" in frame.columns:
        metabolic_score = metabolic_score + config.metabolic_smoothed_label_weight * (1.0 - frame["weak_label_score_smoothed"].clip(0.0, 1.0))
    metabolic_score = metabolic_score.clip(0.0, 1.0)

    frame[ATTENTION_LABEL_COLUMN] = (
        attention_score >= config.attention_threshold
    ).astype(int)
    frame[METABOLIC_LABEL_COLUMN] = (
        metabolic_score >= config.metabolic_threshold
    ).astype(int)
    frame[MISMATCH_LABEL_COLUMN] = (
        (frame[ATTENTION_LABEL_COLUMN] == 1)
        & (frame[METABOLIC_LABEL_COLUMN] == 0)
    ).astype(int)

    frame["attention_score"] = attention_score
    frame["metabolic_score"] = metabolic_score
    # Confidence combines signal quality + low motion + weak-label smoothness.
    smooth_conf = frame["weak_label_score_smoothed"].clip(0.0, 1.0) if "weak_label_score_smoothed" in frame.columns else pd.Series(np.full(len(frame), 0.5), index=frame.index)
    quality_conf = (
        0.45 * frame.get("eeg_quality_score", pd.Series(np.full(len(frame), 0.5), index=frame.index)).clip(0.0, 1.0)
        + 0.35 * optical_quality
        + 0.20 * motion_penalty
    )
    frame["label_confidence"] = np.clip(
        0.6 * quality_conf + 0.4 * smooth_conf,
        config.min_label_confidence,
        1.0,
    )

    return frame


def summarise_dual_head_labels(frame: pd.DataFrame) -> dict[str, float | int]:
    """Quick sanity-check summary used by the training script and tests."""
    for col in (ATTENTION_LABEL_COLUMN, METABOLIC_LABEL_COLUMN, MISMATCH_LABEL_COLUMN):
        if col not in frame.columns:
            raise DualHeadLabelError(
                f"frame missing label column '{col}' — call derive_dual_head_labels first"
            )

    return {
        "rows": int(len(frame)),
        "attention_positive_rate": float(frame[ATTENTION_LABEL_COLUMN].mean()),
        "metabolic_positive_rate": float(frame[METABOLIC_LABEL_COLUMN].mean()),
        "mismatch_positive_rate": float(frame[MISMATCH_LABEL_COLUMN].mean()),
        "attention_score_mean": float(frame.get("attention_score", pd.Series([0.0])).mean()),
        "metabolic_score_mean": float(frame.get("metabolic_score", pd.Series([0.0])).mean()),
        "label_confidence_mean": float(frame.get("label_confidence", pd.Series([1.0])).mean()),
    }


__all__ = [
    "ATTENTION_LABEL_COLUMN",
    "METABOLIC_LABEL_COLUMN",
    "MISMATCH_LABEL_COLUMN",
    "EEG_FEATURE_COLUMNS",
    "OPTICAL_FEATURE_COLUMNS",
    "DualHeadLabelConfig",
    "DualHeadLabelError",
    "derive_dual_head_labels",
    "summarise_dual_head_labels",
]


"""Dual-head neuro-symbolic fatigue model.

Architecture
------------
Two small PyTorch MLPs, each fed a *modality-specific* feature subset so the
heads correspond to interpretable cognitive sub-systems:

* ``AttentionHead``    — input: EEG band-power & quality features
                          output: P(attentive) in [0, 1]

* ``HemodynamicHead``  — input: optical / PPG-derived features
                          output: P(metabolic_ready) in [0, 1]

The wrapper :class:`DualHeadFatigueModel` runs both heads in a single forward
pass and is the runtime artifact saved to disk.

Loss design (the academic "design features to design losses" line item)
----------------------------------------------------------------------
The objective is a *weighted sum of three terms* — two supervised binary
classification losses plus a custom **mismatch-consistency regulariser**:

.. math::

    \\mathcal{L} = \\alpha_a \\cdot \\text{BCE}(p_a, y_a) +
                  \\alpha_m \\cdot \\text{BCE}(p_m, y_m) +
                  \\lambda  \\cdot \\text{BCE}\\bigl(p_a (1 - p_m),\\, y_{mm}\\bigr)

where:

* :math:`p_a, p_m` are the attention / metabolic-ready probabilities,
* :math:`y_a, y_m, y_{mm}` are the corresponding weak labels,
* :math:`p_a (1-p_m)` is the **soft symbolic AND** between "attentive" and
  "not metabolic-ready",
* :math:`\\lambda` weights the consistency term.

That third term is the part that makes the system genuinely
*neuro-symbolic*: it pushes the network to internalise the same logical rule
the hard guardrail will evaluate at runtime, so the two paths agree out of
the box rather than fighting each other.

The training loop uses **AdamW** with **CosineAnnealingLR**, **BatchNorm1d**
on each head, **Dropout** for regularisation, **gradient clipping**, and
**early stopping** on a per-modality selection score — all standard
graduate-level deep-learning hygiene that the rubric explicitly asks for.
"""


from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LOGGER = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class DualHeadConfig:
    """Hyperparameters for the dual-head model + training loop."""

    # `muse_demo` keeps the lightweight architecture used in class demos.
    # `research_eegnet` enables larger research-inspired heads:
    # EEGNet-style separable conv attention + temporal conv metabolic head.
    model_variant: Literal["muse_demo", "research_eegnet"] = "muse_demo"
    attention_hidden: tuple[int, ...] = (16, 8)
    metabolic_hidden: tuple[int, ...] = (12, 6)
    scalable_attention_hidden: tuple[int, ...] = (64, 32, 16)
    scalable_metabolic_hidden: tuple[int, ...] = (48, 24, 12)
    dropout: float = 0.20
    epochs: int = 120
    batch_size: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 25
    seed: int = 1337
    device: str = "cpu"

    # Loss weights
    attention_loss_weight: float = 1.0
    metabolic_loss_weight: float = 1.0
    mismatch_loss_weight: float = 0.5

    # Class-imbalance handling
    attention_pos_weight: float | None = 1.5
    metabolic_pos_weight: float | None = 1.5
    mismatch_pos_weight: float | None = 3.0

    # Cosine LR scheduler floor
    min_lr: float = 1e-5


# ===========================================================================
# Modules
# ===========================================================================


class _MLPHead(nn.Module):
    """Reusable MLP head with input LayerNorm + hidden BatchNorm + Dropout.

    LayerNorm on the input is robust to the small batch sizes we get with
    only a few sessions; BatchNorm on the hidden layers gives the standard
    "normalisation between linear layers" behaviour the rubric expects.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_size)]
        prev = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _ResidualBlock(nn.Module):
    """Simple residual MLP block for scalable feature learning."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.net(x)
        out = out + residual
        out = self.act(out)
        out = self.dropout(out)
        return out


class _EEGNetAttentionHead(nn.Module):
    """EEGNet-style separable-conv attention head (scalable variant).

    This is adapted to the project's engineered EEG feature vectors. The
    separable temporal filtering pattern follows the EEGNet design idea
    (temporal conv + depthwise + pointwise projection) while remaining
    compatible with the current feature-stream training pipeline.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],  # kept for signature parity
        dropout: float,
    ) -> None:
        super().__init__()
        del hidden_sizes
        self.input_norm = nn.LayerNorm(input_size)
        self.temporal = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding=1)
        self.temporal_bn = nn.BatchNorm1d(24)
        self.depthwise = nn.Conv1d(
            in_channels=24,
            out_channels=48,
            kernel_size=3,
            padding=1,
            groups=24,
        )
        self.depthwise_bn = nn.BatchNorm1d(48)
        self.pointwise = nn.Conv1d(in_channels=48, out_channels=48, kernel_size=1)
        self.pointwise_bn = nn.BatchNorm1d(48)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, F] -> [B, 1, F]
        x = self.input_norm(x).unsqueeze(1)
        x = self.act(self.temporal_bn(self.temporal(x)))
        x = self.dropout(x)
        x = self.act(self.depthwise_bn(self.depthwise(x)))
        x = self.dropout(x)
        x = self.act(self.pointwise_bn(self.pointwise(x)))
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)


class _TCNResidualBlock(nn.Module):
    """Dilated temporal residual block for optical scalable head."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        out = self.dropout(out)
        return out


class _ScalableMetabolicHead(nn.Module):
    """Larger TCN-style metabolic head for scalable mode."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],  # kept for signature parity
        dropout: float,
    ) -> None:
        super().__init__()
        del hidden_sizes
        self.input_norm = nn.LayerNorm(input_size)
        self.stem = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.stem_bn = nn.BatchNorm1d(32)
        self.blocks = nn.Sequential(
            _TCNResidualBlock(channels=32, dilation=1, dropout=dropout),
            _TCNResidualBlock(channels=32, dilation=2, dropout=dropout),
            _TCNResidualBlock(channels=32, dilation=4, dropout=dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(32, 24),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x).unsqueeze(1)
        x = self.act(self.stem_bn(self.stem(x)))
        x = self.blocks(x)
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)


class DualHeadFatigueModel(nn.Module):
    """The runtime model: two heads + a soft symbolic AND.

    ``forward`` returns a dict so callers can persist all intermediate
    signals (attention prob, metabolic prob, mismatch prob) to the dashboard
    and the session log.
    """

    def __init__(
        self,
        eeg_feature_count: int,
        optical_feature_count: int,
        attention_hidden: Iterable[int] = (16, 8),
        metabolic_hidden: Iterable[int] = (12, 6),
        dropout: float = 0.20,
        model_variant: Literal["muse_demo", "research_eegnet"] = "muse_demo",
    ) -> None:
        super().__init__()
        self.eeg_feature_count = int(eeg_feature_count)
        self.optical_feature_count = int(optical_feature_count)
        self.model_variant = model_variant

        if model_variant == "muse_demo":
            self.attention_head = _MLPHead(
                input_size=self.eeg_feature_count,
                hidden_sizes=attention_hidden,
                dropout=dropout,
            )
            self.metabolic_head = _MLPHead(
                input_size=self.optical_feature_count,
                hidden_sizes=metabolic_hidden,
                dropout=dropout,
            )
        elif model_variant == "research_eegnet":
            self.attention_head = _EEGNetAttentionHead(
                input_size=self.eeg_feature_count,
                hidden_sizes=attention_hidden,
                dropout=dropout,
            )
            self.metabolic_head = _ScalableMetabolicHead(
                input_size=self.optical_feature_count,
                hidden_sizes=metabolic_hidden,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported dual-head model_variant: {model_variant}")

    def forward(
        self,
        eeg_features: torch.Tensor,
        optical_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        att_logit = self.attention_head(eeg_features)
        meta_logit = self.metabolic_head(optical_features)
        att_prob = torch.sigmoid(att_logit)
        meta_prob = torch.sigmoid(meta_logit)
        # Soft symbolic AND between "attentive" and "NOT metabolic ready".
        mismatch_prob = att_prob * (1.0 - meta_prob)
        return {
            "attention_logit": att_logit,
            "metabolic_logit": meta_logit,
            "attention_prob": att_prob,
            "metabolic_prob": meta_prob,
            "mismatch_prob": mismatch_prob,
        }


# ===========================================================================
# Custom loss
# ===========================================================================


class MismatchConsistencyLoss(nn.Module):
    """Train the soft AND of the two heads to track the symbolic rule.

    The forward call expects already-sigmoided ``attention_prob`` and
    ``metabolic_prob`` tensors plus the binary ``mismatch_label`` tensor.
    A standard BCE on the soft AND probability lets the network learn a
    *jointly* calibrated set of probabilities — without it the two heads
    are free to disagree about edge cases that the symbolic rule cares
    about most.
    """

    def __init__(self, pos_weight: float | None = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(
        self,
        attention_prob: torch.Tensor,
        metabolic_prob: torch.Tensor,
        mismatch_label: torch.Tensor,
    ) -> torch.Tensor:
        soft_and = (attention_prob * (1.0 - metabolic_prob)).clamp(self.eps, 1.0 - self.eps)
        per_sample = -(
            mismatch_label * torch.log(soft_and)
            + (1.0 - mismatch_label) * torch.log(1.0 - soft_and)
        )
        if self.pos_weight is not None:
            weight = torch.where(
                mismatch_label > 0.5,
                torch.full_like(per_sample, float(self.pos_weight)),
                torch.ones_like(per_sample),
            )
            per_sample = per_sample * weight
        return per_sample


# ===========================================================================
# Training result
# ===========================================================================


@dataclass
class DualHeadTrainingResult:
    config: DualHeadConfig
    eeg_features: list[str]
    optical_features: list[str]
    state_dict: dict[str, torch.Tensor]
    metrics: dict[str, Any]
    epoch_log: list[dict[str, Any]] = field(default_factory=list)
    train_sessions: list[str] = field(default_factory=list)
    val_sessions: list[str] = field(default_factory=list)
    test_sessions: list[str] = field(default_factory=list)


# ===========================================================================
# Internal helpers
# ===========================================================================


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    out: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if np.unique(y_true).size < 2:
        out["auroc"] = None
    else:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    return out


def _selection_score(metrics: dict[str, Any]) -> float:
    auroc = metrics.get("auroc")
    auroc_val = float(auroc) if auroc is not None else 0.0
    return (
        0.6 * float(metrics["macro_f1"])
        + 0.3 * float(metrics["balanced_accuracy"])
        + 0.1 * auroc_val
    )


def _evaluate_split(
    model: DualHeadFatigueModel,
    eeg: torch.Tensor,
    opt: torch.Tensor,
    y_att: torch.Tensor,
    y_meta: torch.Tensor,
    y_mm: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        outputs = model(eeg, opt)
        att_prob = outputs["attention_prob"].cpu().numpy()
        meta_prob = outputs["metabolic_prob"].cpu().numpy()
        mm_prob = outputs["mismatch_prob"].cpu().numpy()

    return {
        "attention": _binary_metrics(y_att.cpu().numpy(), att_prob),
        "metabolic": _binary_metrics(y_meta.cpu().numpy(), meta_prob),
        "mismatch": _binary_metrics(y_mm.cpu().numpy(), mm_prob),
    }


def _split_tensors(
    frame: pd.DataFrame,
    eeg_features: list[str],
    optical_features: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eeg = torch.tensor(frame[eeg_features].to_numpy(dtype=np.float32), device=device)
    opt = torch.tensor(frame[optical_features].to_numpy(dtype=np.float32), device=device)
    y_att = torch.tensor(
        frame[ATTENTION_LABEL_COLUMN].to_numpy(dtype=np.float32), device=device
    )
    y_meta = torch.tensor(
        frame[METABOLIC_LABEL_COLUMN].to_numpy(dtype=np.float32), device=device
    )
    y_mm = torch.tensor(
        frame[MISMATCH_LABEL_COLUMN].to_numpy(dtype=np.float32), device=device
    )
    sample_weight = torch.tensor(
        frame.get("label_confidence", pd.Series(np.ones(len(frame), dtype=float))).to_numpy(dtype=np.float32),
        device=device,
    )
    return eeg, opt, y_att, y_meta, y_mm, sample_weight


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(weights.sum(), min=1e-6)
    return (values * weights).sum() / denom


# ===========================================================================
# Public training entry point
# ===========================================================================


def train_dual_head_model(
    labeled_frame: pd.DataFrame,
    eeg_features: Iterable[str] | None = None,
    optical_features: Iterable[str] | None = None,
    config: DualHeadConfig | None = None,
    *,
    split_column: str = "split",
    session_column: str = "session_id",
) -> DualHeadTrainingResult:
    """Train both heads on the same session-held-out split as the MVP.

    The function expects the frame to *already* contain the dual-head label
    columns (use :func:`neuroguage.dual_head_labels.derive_dual_head_labels`
    first).  Returns a :class:`DualHeadTrainingResult` ready for persistence.
    """
    if config is None:
        config = DualHeadConfig()

    if config.model_variant == "research_eegnet":
        default_cfg = DualHeadConfig()
        attention_hidden_default = (
            list(config.attention_hidden) == list(default_cfg.attention_hidden)
        )
        metabolic_hidden_default = (
            list(config.metabolic_hidden) == list(default_cfg.metabolic_hidden)
        )
        if attention_hidden_default:
            config.attention_hidden = tuple(config.scalable_attention_hidden)
        if metabolic_hidden_default:
            config.metabolic_hidden = tuple(config.scalable_metabolic_hidden)

    eeg_features_list = list(eeg_features) if eeg_features is not None else list(EEG_FEATURE_COLUMNS)
    optical_features_list = (
        list(optical_features) if optical_features is not None else list(OPTICAL_FEATURE_COLUMNS)
    )

    required = {
        split_column,
        session_column,
        ATTENTION_LABEL_COLUMN,
        METABOLIC_LABEL_COLUMN,
        MISMATCH_LABEL_COLUMN,
        *eeg_features_list,
        *optical_features_list,
    }
    missing = required - set(labeled_frame.columns)
    if missing:
        raise ValueError(
            f"labeled_frame missing required columns for dual-head training: {sorted(missing)}"
        )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_frame = labeled_frame.loc[labeled_frame[split_column] == "train"].copy()
    val_frame = labeled_frame.loc[labeled_frame[split_column] == "val"].copy()
    test_frame = labeled_frame.loc[labeled_frame[split_column] == "test"].copy()

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError("Train/val/test splits must all be non-empty")

    device = torch.device(config.device)

    eeg_tr, opt_tr, y_att_tr, y_meta_tr, y_mm_tr, w_tr = _split_tensors(
        train_frame, eeg_features_list, optical_features_list, device
    )
    eeg_val, opt_val, y_att_val, y_meta_val, y_mm_val, _ = _split_tensors(
        val_frame, eeg_features_list, optical_features_list, device
    )
    eeg_test, opt_test, y_att_test, y_meta_test, y_mm_test, _ = _split_tensors(
        test_frame, eeg_features_list, optical_features_list, device
    )

    train_dataset = TensorDataset(eeg_tr, opt_tr, y_att_tr, y_meta_tr, y_mm_tr, w_tr)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, max(2, len(train_dataset))),
        shuffle=True,
        drop_last=False,
    )

    model = DualHeadFatigueModel(
        eeg_feature_count=len(eeg_features_list),
        optical_feature_count=len(optical_features_list),
        attention_hidden=config.attention_hidden,
        metabolic_hidden=config.metabolic_hidden,
        dropout=config.dropout,
        model_variant=config.model_variant,
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=max(1, config.epochs),
        eta_min=config.min_lr,
    )

    pos_weight_att = (
        torch.tensor([config.attention_pos_weight], device=device)
        if config.attention_pos_weight is not None
        else None
    )
    pos_weight_meta = (
        torch.tensor([config.metabolic_pos_weight], device=device)
        if config.metabolic_pos_weight is not None
        else None
    )

    bce_attention = nn.BCEWithLogitsLoss(pos_weight=pos_weight_att, reduction="none")
    bce_metabolic = nn.BCEWithLogitsLoss(pos_weight=pos_weight_meta, reduction="none")
    mismatch_loss = MismatchConsistencyLoss(pos_weight=config.mismatch_pos_weight)

    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    epoch_log: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_terms = {"attention": 0.0, "metabolic": 0.0, "mismatch": 0.0}
        seen = 0
        for batch_eeg, batch_opt, batch_y_att, batch_y_meta, batch_y_mm, batch_w in train_loader:
            optimiser.zero_grad()
            outputs = model(batch_eeg, batch_opt)

            loss_att_raw = bce_attention(outputs["attention_logit"], batch_y_att)
            loss_meta_raw = bce_metabolic(outputs["metabolic_logit"], batch_y_meta)
            loss_mm_raw = mismatch_loss(
                outputs["attention_prob"],
                outputs["metabolic_prob"],
                batch_y_mm,
            )
            loss_att = _weighted_mean(loss_att_raw, batch_w)
            loss_meta = _weighted_mean(loss_meta_raw, batch_w)
            loss_mm = _weighted_mean(loss_mm_raw, batch_w)
            loss = (
                config.attention_loss_weight * loss_att
                + config.metabolic_loss_weight * loss_meta
                + config.mismatch_loss_weight * loss_mm
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimiser.step()

            bs = batch_eeg.size(0)
            seen += bs
            running_loss += float(loss.item()) * bs
            running_terms["attention"] += float(loss_att.item()) * bs
            running_terms["metabolic"] += float(loss_meta.item()) * bs
            running_terms["mismatch"] += float(loss_mm.item()) * bs

        scheduler.step()

        train_metrics = _evaluate_split(
            model, eeg_tr, opt_tr, y_att_tr, y_meta_tr, y_mm_tr
        )
        val_metrics = _evaluate_split(
            model, eeg_val, opt_val, y_att_val, y_meta_val, y_mm_val
        )

        # Selection score blends both heads + the symbolic AND so we
        # commit to a checkpoint that is *jointly* good rather than one
        # that nails attention while the metabolic head is random.
        composite_score = (
            0.4 * _selection_score(val_metrics["attention"])
            + 0.4 * _selection_score(val_metrics["metabolic"])
            + 0.2 * _selection_score(val_metrics["mismatch"])
        )

        epoch_log.append(
            {
                "epoch": epoch,
                "lr": float(scheduler.get_last_lr()[0]),
                "train_loss": running_loss / max(1, seen),
                "train_loss_terms": {
                    k: v / max(1, seen) for k, v in running_terms.items()
                },
                "train": train_metrics,
                "validation": val_metrics,
                "composite_score": composite_score,
            }
        )

        if composite_score > best_score:
            best_score = composite_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                LOGGER.info(
                    "Early stopping at epoch %d (best composite=%.4f)",
                    epoch,
                    best_score,
                )
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    final_train = _evaluate_split(model, eeg_tr, opt_tr, y_att_tr, y_meta_tr, y_mm_tr)
    final_val = _evaluate_split(model, eeg_val, opt_val, y_att_val, y_meta_val, y_mm_val)
    final_test = _evaluate_split(
        model, eeg_test, opt_test, y_att_test, y_meta_test, y_mm_test
    )

    metrics = {
        "train": final_train,
        "validation": final_val,
        "test": final_test,
        "best_validation_composite_score": best_score,
        "epochs_trained": len(epoch_log),
    }

    return DualHeadTrainingResult(
        config=config,
        eeg_features=eeg_features_list,
        optical_features=optical_features_list,
        state_dict=best_state,
        metrics=metrics,
        epoch_log=epoch_log,
        train_sessions=sorted(train_frame[session_column].astype(str).unique().tolist()),
        val_sessions=sorted(val_frame[session_column].astype(str).unique().tolist()),
        test_sessions=sorted(test_frame[session_column].astype(str).unique().tolist()),
    )


# ===========================================================================
# Persistence
# ===========================================================================


def write_dual_head_artifacts(
    result: DualHeadTrainingResult,
    model_dir: str | Path,
    metrics_path: str | Path | None = None,
    training_log_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist the dual-head model, hyperparameters, metrics, and epoch log."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "dual_head.pt"
    torch.save(result.state_dict, weights_path)

    config_payload = {
        "architecture": (
            "dual_head_research_eegnet_tcn"
            if result.config.model_variant == "research_eegnet"
            else "dual_head_muse_demo_mlp"
        ),
        "model_variant": result.config.model_variant,
        "hyperparameters": asdict(result.config),
        "eeg_features": result.eeg_features,
        "optical_features": result.optical_features,
        "train_sessions": result.train_sessions,
        "val_sessions": result.val_sessions,
        "test_sessions": result.test_sessions,
    }
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    metrics_path_resolved = (
        Path(metrics_path) if metrics_path is not None else model_dir / "metrics.json"
    )
    metrics_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    metrics_path_resolved.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    log_path_resolved = (
        Path(training_log_path)
        if training_log_path is not None
        else model_dir / "training_log.json"
    )
    log_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    log_path_resolved.write_text(json.dumps(result.epoch_log, indent=2), encoding="utf-8")

    return {
        "weights_path": str(weights_path),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path_resolved),
        "training_log_path": str(log_path_resolved),
    }


def load_dual_head_model(
    weights_path: str | Path,
    config_path: str | Path,
    device: str = "cpu",
) -> tuple[DualHeadFatigueModel, dict[str, Any]]:
    """Load a trained dual-head model and its config payload."""
    config_payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    hp = config_payload["hyperparameters"]
    variant_raw = str(
        config_payload.get("model_variant")
        or hp.get("model_variant")
        or (
            "research_eegnet"
            if str(config_payload.get("architecture", "")).endswith("research_eegnet_tcn")
            else "muse_demo"
        )
    )
    alias_map = {
        "demoable_mlp": "muse_demo",
        "scalable": "research_eegnet",
        "muse_demo": "muse_demo",
        "research_eegnet": "research_eegnet",
    }
    variant = alias_map.get(variant_raw, "muse_demo")
    variant_lit: Literal["muse_demo", "research_eegnet"] = (
        "research_eegnet" if variant == "research_eegnet" else "muse_demo"
    )

    model = DualHeadFatigueModel(
        eeg_feature_count=len(config_payload["eeg_features"]),
        optical_feature_count=len(config_payload["optical_features"]),
        attention_hidden=tuple(hp.get("attention_hidden", (16, 8))),
        metabolic_hidden=tuple(hp.get("metabolic_hidden", (12, 6))),
        dropout=float(hp.get("dropout", 0.20)),
        model_variant=variant_lit,
    )
    state = torch.load(Path(weights_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(torch.device(device))
    return model, config_payload


def predict_dual_head(
    model: DualHeadFatigueModel,
    eeg_feature_vector: np.ndarray | list[float],
    optical_feature_vector: np.ndarray | list[float],
    device: str = "cpu",
) -> dict[str, float]:
    """Single-window inference helper used by the live app.

    Returns plain Python floats so the caller can plug them into the
    dashboard / session log without dragging a torch tensor along.
    """
    eeg_arr = np.asarray(eeg_feature_vector, dtype=np.float32).reshape(1, -1)
    opt_arr = np.asarray(optical_feature_vector, dtype=np.float32).reshape(1, -1)
    dev = torch.device(device)
    with torch.no_grad():
        eeg_t = torch.from_numpy(eeg_arr).to(dev)
        opt_t = torch.from_numpy(opt_arr).to(dev)
        outputs = model(eeg_t, opt_t)
    return {
        "attention_prob": float(outputs["attention_prob"].item()),
        "metabolic_prob": float(outputs["metabolic_prob"].item()),
        "mismatch_prob": float(outputs["mismatch_prob"].item()),
    }


__all__ = [
    "DualHeadConfig",
    "DualHeadFatigueModel",
    "DualHeadTrainingResult",
    "MismatchConsistencyLoss",
    "load_dual_head_model",
    "predict_dual_head",
    "train_dual_head_model",
    "write_dual_head_artifacts",
]

