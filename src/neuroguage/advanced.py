"""Day 7 — T18  Graduate Model Prototype (BiGRU + Attention).

Implements an advanced temporal classifier for per-window fatigue risk that is
directly comparable to the MVP late-fusion model on an identical held-out
split.

Design decisions:

  T18.1  BiGRU-attention architecture
     - Per-session window sequences are fed through a bidirectional GRU to
       capture temporal context.  A scaled dot-product self-attention layer
       mixes global context across timesteps, and a small MLP head produces
       a per-window fatigue logit.  Output shape matches the MVP model so
       downstream guardrail and comparison code is reusable.

  T18.2  Train on the IDENTICAL session-held-out split used by the MVP
     - Accepts the same `labeled_feature_matrix.csv`, honours the existing
       `split` column, and reuses the selected ablation variant's
       `feature_columns.json` so AC-006 ("evaluated on identical held-out
       sessions as MVP") is satisfied by construction.

  T18.3  Latency harness parity with the MVP
     - `infer_single_window` runs one forward pass over a single (padded)
       window sequence so the same `hardening.benchmark_refresh_latency`
       harness can be reused for apples-to-apples latency comparison.

  T18.4  Persist artifacts, config, and evaluation logs
     - Saves model `.pt`, `config.json`, `feature_columns.json`,
       `training_metrics.json`, and `predictions.csv` under a dedicated
       Day 7 directory so Day 7 scripts and the release bundle can ingest
       them without touching MVP artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class AdvancedModelConfig:
    """Configuration for the BiGRU-attention prototype."""

    hidden_size: int = 32
    num_layers: int = 1
    attention_heads: int = 2
    dropout: float = 0.1
    epochs: int = 80
    batch_size: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    max_sequence_length: int = 256
    class_positive_weight: float | None = 2.0
    seed: int = 1337
    early_stopping_patience: int = 15
    device: str = "cpu"


# ===========================================================================
# Dataset
# ===========================================================================

class _SessionSequenceDataset(Dataset):
    """Yields (feature_tensor, label_tensor, mask_tensor) per session."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        session_column: str,
        time_column: str,
        max_length: int,
    ) -> None:
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.max_length = max_length

        self._sessions: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

        for session_id, group in frame.groupby(session_column, sort=True):
            ordered = group.sort_values(time_column) if time_column in group.columns else group
            feats = ordered[feature_columns].to_numpy(dtype=np.float32)
            labels = ordered[label_column].to_numpy(dtype=np.float32)
            window_ids = ordered.get("window_id", pd.Series(range(len(ordered)))).to_numpy()

            if len(feats) > max_length:
                feats = feats[:max_length]
                labels = labels[:max_length]
                window_ids = window_ids[:max_length]

            self._sessions.append((str(session_id), feats, labels, window_ids))

    def __len__(self) -> int:
        return len(self._sessions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        session_id, feats, labels, window_ids = self._sessions[idx]
        length = feats.shape[0]
        padded_feats = np.zeros((self.max_length, feats.shape[1]), dtype=np.float32)
        padded_labels = np.zeros((self.max_length,), dtype=np.float32)
        mask = np.zeros((self.max_length,), dtype=np.float32)

        padded_feats[:length] = feats
        padded_labels[:length] = labels
        mask[:length] = 1.0

        return {
            "session_id": session_id,
            "features": torch.from_numpy(padded_feats),
            "labels": torch.from_numpy(padded_labels),
            "mask": torch.from_numpy(mask),
            "length": length,
            "window_ids": window_ids,
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "session_ids": [b["session_id"] for b in batch],
        "features": torch.stack([b["features"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "lengths": [b["length"] for b in batch],
        "window_ids": [b["window_ids"] for b in batch],
    }


# ===========================================================================
# Model
# ===========================================================================

class BiGRUAttentionClassifier(nn.Module):
    """Per-timestep binary classifier with BiGRU backbone and self-attention.

    For each window t in a session sequence of length T, the model produces a
    fatigue-risk logit by combining:
      - Local temporal context from a bidirectional GRU.
      - Global context via multi-head self-attention across all timesteps
        in the same session.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        attention_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.input_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out = hidden_size * 2

        self.attention = nn.MultiheadAttention(
            embed_dim=gru_out,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(gru_out)

        self.head = nn.Sequential(
            nn.Linear(gru_out, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, F) tensor of window features.
        mask : (B, T) tensor, 1 for real windows, 0 for padding.

        Returns
        -------
        logits : (B, T) tensor of per-window logits.
        """
        x = self.input_norm(x)
        gru_out, _ = self.gru(x)

        key_padding_mask = (mask < 0.5)  # True marks padded positions
        attn_out, _ = self.attention(
            gru_out,
            gru_out,
            gru_out,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.attn_norm(gru_out + attn_out)
        logits = self.head(fused).squeeze(-1)
        return logits


# ===========================================================================
# Training
# ===========================================================================

@dataclass
class AdvancedTrainingResult:
    config: AdvancedModelConfig
    feature_columns: list[str]
    input_size: int
    max_sequence_length: int
    state_dict: dict[str, Any]
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    train_sessions: list[str]
    val_sessions: list[str]
    test_sessions: list[str]
    epoch_log: list[dict[str, Any]] = field(default_factory=list)


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


def _evaluate_loader(
    model: BiGRUAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    all_true: list[float] = []
    all_prob: list[float] = []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            labels = batch["labels"]
            mask = batch["mask"].to(device)
            logits = model(feats, mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            for bi, session_id in enumerate(batch["session_ids"]):
                length = batch["lengths"][bi]
                window_ids = batch["window_ids"][bi]
                for wi in range(length):
                    rows.append(
                        {
                            "session_id": session_id,
                            "window_id": int(window_ids[wi]) if wi < len(window_ids) else wi,
                            "risk_score": float(probs[bi, wi]),
                            "label": float(labels[bi, wi].item()),
                        }
                    )
                    all_true.append(float(labels[bi, wi].item()))
                    all_prob.append(float(probs[bi, wi]))

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return {"macro_f1": 0.0, "balanced_accuracy": 0.0, "auroc": None}, predictions
    metrics = _binary_metrics(np.array(all_true), np.array(all_prob))
    return metrics, predictions


def train_advanced_model(
    labeled_frame: pd.DataFrame,
    feature_columns: list[str],
    config: AdvancedModelConfig | None = None,
    *,
    split_column: str = "split",
    label_column: str = "fatigue_label",
    session_column: str = "session_id",
    time_column: str = "start_time_s",
) -> AdvancedTrainingResult:
    """Train the BiGRU-attention classifier (T18.2).

    Uses the same split column as the MVP so the held-out session partition
    is identical.  Returns a result bundle the caller can persist and
    compare against the MVP on equal footing.
    """
    if config is None:
        config = AdvancedModelConfig()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    required = {split_column, label_column, session_column}
    missing = required - set(labeled_frame.columns)
    if missing:
        raise ValueError(f"labeled_frame missing required columns: {sorted(missing)}")

    frame = labeled_frame.copy()
    frame[label_column] = frame[label_column].astype(int)

    train_frame = frame.loc[frame[split_column] == "train"].copy()
    val_frame = frame.loc[frame[split_column] == "val"].copy()
    test_frame = frame.loc[frame[split_column] == "test"].copy()

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError("Train/val/test splits must all be non-empty for advanced training")

    device = torch.device(config.device)
    train_ds = _SessionSequenceDataset(
        train_frame,
        feature_columns,
        label_column,
        session_column,
        time_column,
        config.max_sequence_length,
    )
    val_ds = _SessionSequenceDataset(
        val_frame,
        feature_columns,
        label_column,
        session_column,
        time_column,
        config.max_sequence_length,
    )
    test_ds = _SessionSequenceDataset(
        test_frame,
        feature_columns,
        label_column,
        session_column,
        time_column,
        config.max_sequence_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, collate_fn=_collate)

    input_size = len(feature_columns)
    model = BiGRUAttentionClassifier(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        attention_heads=config.attention_heads,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    pos_weight = (
        torch.tensor([config.class_positive_weight], device=device)
        if config.class_positive_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    epoch_log: list[dict[str, Any]] = []
    best_val_score = -float("inf")
    best_state: dict[str, Any] | None = None
    patience_counter = 0

    def _selection_score(metrics: dict[str, Any]) -> float:
        auroc = metrics.get("auroc")
        auroc_val = float(auroc) if auroc is not None else 0.0
        return (
            0.7 * float(metrics["macro_f1"])
            + 0.2 * float(metrics["balanced_accuracy"])
            + 0.1 * auroc_val
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch in train_loader:
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(feats, mask)
            per_element = criterion(logits, labels)
            mask_sum = mask.sum().clamp_min(1.0)
            loss = (per_element * mask).sum() / mask_sum

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            batch_size = feats.size(0)
            epoch_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_metrics, _ = _evaluate_loader(model, train_loader, device)
        val_metrics, _ = _evaluate_loader(model, val_loader, device)
        score = _selection_score(val_metrics)

        epoch_log.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, seen),
                "train": train_metrics,
                "validation": val_metrics,
                "selection_score": score,
            }
        )

        if score > best_val_score:
            best_val_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                LOGGER.info("Early stopping at epoch %d (best score %.4f)", epoch, best_val_score)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    train_metrics, _ = _evaluate_loader(model, train_loader, device)
    val_metrics, val_predictions = _evaluate_loader(model, val_loader, device)
    test_metrics, test_predictions = _evaluate_loader(model, test_loader, device)

    val_predictions = val_predictions.assign(split="val")
    test_predictions = test_predictions.assign(split="test")
    predictions = pd.concat([val_predictions, test_predictions], ignore_index=True)

    metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "best_validation_selection_score": best_val_score,
        "epochs_trained": len(epoch_log),
    }

    train_sessions = sorted(train_frame[session_column].astype(str).unique().tolist())
    val_sessions = sorted(val_frame[session_column].astype(str).unique().tolist())
    test_sessions = sorted(test_frame[session_column].astype(str).unique().tolist())

    return AdvancedTrainingResult(
        config=config,
        feature_columns=list(feature_columns),
        input_size=input_size,
        max_sequence_length=config.max_sequence_length,
        state_dict=best_state,
        metrics=metrics,
        predictions=predictions,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        test_sessions=test_sessions,
        epoch_log=epoch_log,
    )


# ===========================================================================
# Persistence
# ===========================================================================

def write_advanced_artifacts(
    result: AdvancedTrainingResult,
    model_dir: Path,
    metrics_path: Path,
    predictions_path: Path,
    training_log_path: Path,
) -> dict[str, str]:
    """Persist advanced model artifacts (T18.4).

    Produces a self-contained bundle the comparison + release scripts can
    reference without re-training.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "bigru_attention.pt"
    torch.save(result.state_dict, weights_path)

    config_payload = {
        "architecture": "bigru_attention",
        "hyperparameters": asdict(result.config),
        "input_size": result.input_size,
        "max_sequence_length": result.max_sequence_length,
        "feature_columns": result.feature_columns,
        "train_sessions": result.train_sessions,
        "val_sessions": result.val_sessions,
        "test_sessions": result.test_sessions,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    (model_dir / "feature_columns.json").write_text(
        json.dumps(result.feature_columns, indent=2), encoding="utf-8"
    )

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(predictions_path, index=False)

    training_log_path = Path(training_log_path)
    training_log_path.parent.mkdir(parents=True, exist_ok=True)
    training_log_path.write_text(
        json.dumps(result.epoch_log, indent=2), encoding="utf-8"
    )

    return {
        "weights_path": str(weights_path),
        "config_path": str(model_dir / "config.json"),
        "feature_columns_path": str(model_dir / "feature_columns.json"),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "training_log_path": str(training_log_path),
    }


# ===========================================================================
# Runtime utilities for latency parity (T18.3)
# ===========================================================================

def load_advanced_model(
    weights_path: Path,
    config_path: Path,
    device: str = "cpu",
) -> tuple[BiGRUAttentionClassifier, dict[str, Any]]:
    """Load a trained advanced model for inference / latency benchmarking."""
    config_payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    hp = config_payload["hyperparameters"]

    model = BiGRUAttentionClassifier(
        input_size=int(config_payload["input_size"]),
        hidden_size=int(hp["hidden_size"]),
        num_layers=int(hp["num_layers"]),
        attention_heads=int(hp["attention_heads"]),
        dropout=float(hp["dropout"]),
    )
    state = torch.load(Path(weights_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(torch.device(device))
    return model, config_payload


def infer_single_window(
    model: BiGRUAttentionClassifier,
    window_features: np.ndarray,
    device: str = "cpu",
) -> float:
    """Single-window inference helper for latency benchmarking.

    Accepts a 1-D feature vector, runs a minimal (B=1, T=1) forward pass,
    and returns the sigmoid risk score.  This is the lightest possible
    invocation so the MVP and advanced path are compared on an equivalent
    per-window cycle.
    """
    dev = torch.device(device)
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(window_features, dtype=np.float32)).view(1, 1, -1).to(dev)
        mask = torch.ones(1, 1, dtype=torch.float32, device=dev)
        logit = model(x, mask).view(-1)
        return float(torch.sigmoid(logit).item())
