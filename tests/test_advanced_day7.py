"""Tests for Day 7 — T18 BiGRU-attention advanced model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neuroguage.lifecycle import (
    AdvancedModelConfig,
    BiGRUAttentionClassifier,
    infer_single_window,
    load_advanced_model,
    train_advanced_model,
    write_advanced_artifacts,
)


@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    """Tiny multi-session labeled matrix with a latent signal in a feature."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, float | str | int]] = []
    feature_cols = ["z_alpha", "z_beta", "z_theta"]

    def session_rows(session_id: str, split: str, offset: float, label: int) -> None:
        n = 12
        for i in range(n):
            base = rng.normal(0.0, 0.1, size=3)
            base[0] += offset + (i / n) * 0.2 * (1 if label == 1 else -1)
            rows.append(
                {
                    "session_id": session_id,
                    "split": split,
                    "start_time_s": float(i),
                    "window_id": i,
                    "z_alpha": float(base[0]),
                    "z_beta": float(base[1]),
                    "z_theta": float(base[2]),
                    "fatigue_label": label,
                }
            )

    session_rows("train_a", "train", offset=-0.6, label=0)
    session_rows("train_b", "train", offset=+0.6, label=1)
    session_rows("val_a", "val", offset=+0.6, label=1)
    session_rows("test_a", "test", offset=+0.6, label=1)
    session_rows("test_b", "test", offset=-0.6, label=0)

    frame = pd.DataFrame(rows)
    for col in feature_cols:
        frame[col] = frame[col].astype(float)
    return frame


def test_bigru_attention_shapes() -> None:
    model = BiGRUAttentionClassifier(input_size=4, hidden_size=8, num_layers=1, attention_heads=2)
    x = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5)
    mask[1, 3:] = 0
    out = model(x, mask)
    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()


def test_train_advanced_model_returns_metrics_and_predictions(
    synthetic_frame: pd.DataFrame,
) -> None:
    feature_columns = ["z_alpha", "z_beta", "z_theta"]
    config = AdvancedModelConfig(
        epochs=8,
        batch_size=2,
        hidden_size=8,
        attention_heads=2,
        early_stopping_patience=5,
    )
    result = train_advanced_model(
        labeled_frame=synthetic_frame,
        feature_columns=feature_columns,
        config=config,
    )
    assert set(result.metrics) == {
        "train",
        "validation",
        "test",
        "best_validation_selection_score",
        "epochs_trained",
    }
    for split in ("train", "validation", "test"):
        assert "macro_f1" in result.metrics[split]
        assert "balanced_accuracy" in result.metrics[split]

    assert len(result.predictions) > 0
    assert set(result.predictions.columns) >= {
        "session_id",
        "window_id",
        "risk_score",
        "label",
        "split",
    }
    assert ((result.predictions["risk_score"] >= 0.0) & (result.predictions["risk_score"] <= 1.0)).all()
    assert "train_a" in result.train_sessions
    assert "val_a" in result.val_sessions
    assert set(result.test_sessions) == {"test_a", "test_b"}


def test_write_and_load_advanced_artifacts(
    synthetic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    feature_columns = ["z_alpha", "z_beta", "z_theta"]
    config = AdvancedModelConfig(
        epochs=4,
        batch_size=2,
        hidden_size=8,
        attention_heads=2,
        early_stopping_patience=3,
    )
    result = train_advanced_model(
        labeled_frame=synthetic_frame,
        feature_columns=feature_columns,
        config=config,
    )
    paths = write_advanced_artifacts(
        result=result,
        model_dir=tmp_path / "model",
        metrics_path=tmp_path / "reports" / "metrics.json",
        predictions_path=tmp_path / "reports" / "preds.csv",
        training_log_path=tmp_path / "reports" / "log.json",
    )
    assert Path(paths["weights_path"]).exists()
    assert Path(paths["config_path"]).exists()
    assert Path(paths["metrics_path"]).exists()
    assert Path(paths["predictions_path"]).exists()

    loaded_metrics = json.loads(Path(paths["metrics_path"]).read_text())
    assert "test" in loaded_metrics

    model, config_payload = load_advanced_model(
        weights_path=Path(paths["weights_path"]),
        config_path=Path(paths["config_path"]),
    )
    assert config_payload["architecture"] == "bigru_attention"
    assert config_payload["feature_columns"] == feature_columns

    vec = np.zeros(len(feature_columns), dtype=float)
    score = infer_single_window(model, vec)
    assert 0.0 <= score <= 1.0


def test_train_advanced_model_rejects_empty_splits(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.loc[synthetic_frame["split"] != "val"].copy()
    with pytest.raises(ValueError):
        train_advanced_model(
            labeled_frame=frame,
            feature_columns=["z_alpha", "z_beta", "z_theta"],
            config=AdvancedModelConfig(epochs=2, batch_size=2),
        )
