"""Smoke tests for :mod:`neuroguage.dual_head_model`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroguage.dual_head import (  # noqa: E402
    ATTENTION_LABEL_COLUMN,
    EEG_FEATURE_COLUMNS,
    METABOLIC_LABEL_COLUMN,
    MISMATCH_LABEL_COLUMN,
    OPTICAL_FEATURE_COLUMNS,
)
from neuroguage.dual_head import (  # noqa: E402
    DualHeadConfig,
    DualHeadFatigueModel,
    MismatchConsistencyLoss,
    load_dual_head_model,
    predict_dual_head,
    train_dual_head_model,
    write_dual_head_artifacts,
)


def _make_synthetic_frame(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []

    def session(session_id: str, split: str, fatigued: bool) -> None:
        for window in range(8):
            base_eeg = rng.normal(0.0, 0.4, size=len(EEG_FEATURE_COLUMNS))
            base_opt = rng.normal(0.0, 0.4, size=len(OPTICAL_FEATURE_COLUMNS))
            if fatigued:
                base_eeg += 0.6      # latent "engaged" signal
                base_opt -= 0.6      # latent "metabolic-degraded" signal
            row: dict[str, float | str | int] = {
                "session_id": session_id,
                "split": split,
                ATTENTION_LABEL_COLUMN: int(fatigued),
                METABOLIC_LABEL_COLUMN: int(not fatigued),
                MISMATCH_LABEL_COLUMN: int(fatigued),
            }
            for col, val in zip(EEG_FEATURE_COLUMNS, base_eeg):
                row[col] = float(val)
            for col, val in zip(OPTICAL_FEATURE_COLUMNS, base_opt):
                row[col] = float(val)
            rows.append(row)

    session("train_a", "train", fatigued=False)
    session("train_b", "train", fatigued=True)
    session("val_a", "val", fatigued=True)
    session("val_b", "val", fatigued=False)
    session("test_a", "test", fatigued=True)
    session("test_b", "test", fatigued=False)
    return pd.DataFrame(rows)


def test_dual_head_forward_returns_expected_keys() -> None:
    model = DualHeadFatigueModel(
        eeg_feature_count=len(EEG_FEATURE_COLUMNS),
        optical_feature_count=len(OPTICAL_FEATURE_COLUMNS),
    )
    eeg = torch.randn(4, len(EEG_FEATURE_COLUMNS))
    opt = torch.randn(4, len(OPTICAL_FEATURE_COLUMNS))
    out = model(eeg, opt)
    for key in (
        "attention_logit", "metabolic_logit",
        "attention_prob", "metabolic_prob", "mismatch_prob",
    ):
        assert key in out
    assert out["attention_prob"].shape == (4,)
    assert torch.all((out["mismatch_prob"] >= 0) & (out["mismatch_prob"] <= 1))


def test_scalable_dual_head_forward_returns_expected_keys() -> None:
    model = DualHeadFatigueModel(
        eeg_feature_count=len(EEG_FEATURE_COLUMNS),
        optical_feature_count=len(OPTICAL_FEATURE_COLUMNS),
        model_variant="research_eegnet",
        attention_hidden=(32, 16),
        metabolic_hidden=(24, 12),
    )
    eeg = torch.randn(4, len(EEG_FEATURE_COLUMNS))
    opt = torch.randn(4, len(OPTICAL_FEATURE_COLUMNS))
    out = model(eeg, opt)
    assert out["attention_prob"].shape == (4,)
    assert torch.all((out["attention_prob"] >= 0) & (out["attention_prob"] <= 1))
    assert torch.all((out["metabolic_prob"] >= 0) & (out["metabolic_prob"] <= 1))


def test_mismatch_consistency_loss_reduces_with_aligned_predictions() -> None:
    loss = MismatchConsistencyLoss(pos_weight=2.0)
    label = torch.tensor([1.0, 0.0, 1.0, 0.0])
    aligned = loss(
        attention_prob=torch.tensor([0.95, 0.05, 0.92, 0.10]),
        metabolic_prob=torch.tensor([0.05, 0.95, 0.10, 0.90]),
        mismatch_label=label,
    )
    misaligned = loss(
        attention_prob=torch.tensor([0.10, 0.90, 0.10, 0.90]),
        metabolic_prob=torch.tensor([0.95, 0.05, 0.95, 0.05]),
        mismatch_label=label,
    )
    assert float(aligned.mean()) < float(misaligned.mean())


def test_train_dual_head_model_smoke(tmp_path: Path) -> None:
    frame = _make_synthetic_frame()
    config = DualHeadConfig(
        epochs=15, batch_size=4, learning_rate=5e-3,
        early_stopping_patience=10, dropout=0.1,
    )
    result = train_dual_head_model(frame, config=config)
    assert "validation" in result.metrics
    assert result.metrics["epochs_trained"] >= 1
    assert "macro_f1" in result.metrics["validation"]["attention"]

    artifacts = write_dual_head_artifacts(result, model_dir=tmp_path)
    for key in ("weights_path", "config_path", "metrics_path", "training_log_path"):
        assert Path(artifacts[key]).exists()


def test_load_and_predict_dual_head(tmp_path: Path) -> None:
    frame = _make_synthetic_frame()
    config = DualHeadConfig(epochs=10, batch_size=4, dropout=0.05)
    result = train_dual_head_model(frame, config=config)
    artifacts = write_dual_head_artifacts(result, model_dir=tmp_path)

    model, payload = load_dual_head_model(
        weights_path=artifacts["weights_path"],
        config_path=artifacts["config_path"],
    )
    assert payload["model_variant"] == "muse_demo"
    assert payload["architecture"] == "dual_head_muse_demo_mlp"
    eeg_vec = [0.5] * len(EEG_FEATURE_COLUMNS)
    opt_vec = [0.5] * len(OPTICAL_FEATURE_COLUMNS)
    out = predict_dual_head(model, eeg_vec, opt_vec)
    for key in ("attention_prob", "metabolic_prob", "mismatch_prob"):
        assert 0.0 <= out[key] <= 1.0


def test_train_load_predict_scalable_variant(tmp_path: Path) -> None:
    frame = _make_synthetic_frame()
    config = DualHeadConfig(
        model_variant="research_eegnet",
        epochs=10,
        batch_size=4,
        dropout=0.10,
    )
    result = train_dual_head_model(frame, config=config)
    artifacts = write_dual_head_artifacts(result, model_dir=tmp_path)

    model, payload = load_dual_head_model(
        weights_path=artifacts["weights_path"],
        config_path=artifacts["config_path"],
    )
    assert payload["model_variant"] == "research_eegnet"
    assert payload["architecture"] == "dual_head_research_eegnet_tcn"

    eeg_vec = [0.3] * len(EEG_FEATURE_COLUMNS)
    opt_vec = [0.4] * len(OPTICAL_FEATURE_COLUMNS)
    out = predict_dual_head(model, eeg_vec, opt_vec)
    assert 0.0 <= out["attention_prob"] <= 1.0
    assert 0.0 <= out["metabolic_prob"] <= 1.0
    assert 0.0 <= out["mismatch_prob"] <= 1.0
