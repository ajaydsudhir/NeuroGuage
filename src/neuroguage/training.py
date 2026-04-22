from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


class TrainingError(ValueError):
    """Raised when Day 3 MVP model training cannot proceed."""


@dataclass
class ModelSelectionResult:
    name: str
    estimator: Any
    parameters: dict[str, Any]
    validation_metrics: dict[str, Any]


@dataclass
class TrainingResult:
    linear_result: ModelSelectionResult
    tree_result: ModelSelectionResult
    fusion_weights: dict[str, float]
    metrics: dict[str, Any]
    feature_columns: list[str]
    scaler_metadata: dict[str, Any] | None


def _safe_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_pred = (probabilities >= 0.5).astype(int)

    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        metrics["auroc"] = None
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, probabilities))

    return metrics


def _selection_score(metrics: dict[str, Any]) -> float:
    auroc = metrics.get("auroc")
    auroc_score = float(auroc) if auroc is not None else 0.0
    return (0.7 * float(metrics["macro_f1"])) + (0.2 * float(metrics["balanced_accuracy"])) + (0.1 * auroc_score)


def _calibrate_estimator(base_estimator: Any, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    class_counts = np.bincount(y_train)
    min_count = int(class_counts.min()) if class_counts.size > 1 else 0

    if min_count < 2:
        return base_estimator

    cv = min(3, min_count)
    calibrator = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=cv)
    calibrator.fit(x_train, y_train)
    return calibrator


def _fit_linear_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> ModelSelectionResult:
    candidates: list[ModelSelectionResult] = []

    for c_value in [0.1, 1.0, 10.0]:
        estimator = LogisticRegression(
            C=c_value,
            max_iter=600,
            class_weight="balanced",
            random_state=random_state,
        )
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_val)[:, 1]
        metrics = _safe_metrics(y_val, probabilities)
        candidates.append(
            ModelSelectionResult(
                name="logistic_regression",
                estimator=estimator,
                parameters={"C": c_value},
                validation_metrics=metrics,
            )
        )

    lda_configs = [
        {"solver": "svd", "shrinkage": None},
        {"solver": "lsqr", "shrinkage": "auto"},
        {"solver": "lsqr", "shrinkage": 0.1},
        {"solver": "lsqr", "shrinkage": 0.3},
    ]

    for config in lda_configs:
        estimator = LinearDiscriminantAnalysis(**config)
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_val)[:, 1]
        metrics = _safe_metrics(y_val, probabilities)
        candidates.append(
            ModelSelectionResult(
                name="lda",
                estimator=estimator,
                parameters=config,
                validation_metrics=metrics,
            )
        )

    best = max(candidates, key=lambda candidate: _selection_score(candidate.validation_metrics))
    return best


def _fit_tree_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> ModelSelectionResult:
    candidates: list[ModelSelectionResult] = []

    tree_configs = [
        {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 160, "max_depth": 6, "min_samples_leaf": 1},
        {"n_estimators": 220, "max_depth": 8, "min_samples_leaf": 2},
    ]

    for config in tree_configs:
        estimator = RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
            **config,
        )
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_val)[:, 1]
        metrics = _safe_metrics(y_val, probabilities)
        candidates.append(
            ModelSelectionResult(
                name="random_forest",
                estimator=estimator,
                parameters=config,
                validation_metrics=metrics,
            )
        )

    best = max(candidates, key=lambda candidate: _selection_score(candidate.validation_metrics))
    return best


def _select_feature_columns(frame: pd.DataFrame) -> list[str]:
    z_columns = [column for column in frame.columns if column.startswith("z_")]
    if not z_columns:
        raise TrainingError("No normalized feature columns found. Expected columns prefixed with 'z_'.")
    return sorted(z_columns)


def run_training_pipeline(
    labeled_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    split_column: str = "split",
    label_column: str = "fatigue_label",
    random_state: int = 42,
    scaler_metadata: dict[str, Any] | None = None,
) -> TrainingResult:
    if labeled_frame.empty:
        raise TrainingError("labeled_frame is empty")

    required_columns = [split_column, label_column]
    missing = [column for column in required_columns if column not in labeled_frame.columns]
    if missing:
        raise TrainingError(f"labeled_frame missing required columns: {missing}")

    frame = labeled_frame.copy()
    frame[label_column] = frame[label_column].astype(int)

    if feature_columns is None:
        feature_columns = _select_feature_columns(frame)

    train_frame = frame.loc[frame[split_column] == "train"].copy()
    val_frame = frame.loc[frame[split_column] == "val"].copy()
    test_frame = frame.loc[frame[split_column] == "test"].copy()

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise TrainingError("Train/val/test splits must all be non-empty")

    if train_frame[label_column].nunique() < 2:
        raise TrainingError("Training split must include at least two classes")

    x_train = train_frame[feature_columns].to_numpy(dtype=float)
    y_train = train_frame[label_column].to_numpy(dtype=int)
    x_val = val_frame[feature_columns].to_numpy(dtype=float)
    y_val = val_frame[label_column].to_numpy(dtype=int)
    x_test = test_frame[feature_columns].to_numpy(dtype=float)
    y_test = test_frame[label_column].to_numpy(dtype=int)

    linear_result = _fit_linear_candidates(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=random_state,
    )
    tree_result = _fit_tree_candidates(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=random_state,
    )

    calibrated_linear = _calibrate_estimator(linear_result.estimator, x_train, y_train)
    calibrated_tree = _calibrate_estimator(tree_result.estimator, x_train, y_train)

    linear_val_prob = calibrated_linear.predict_proba(x_val)[:, 1]
    tree_val_prob = calibrated_tree.predict_proba(x_val)[:, 1]

    linear_val_metrics = _safe_metrics(y_val, linear_val_prob)
    tree_val_metrics = _safe_metrics(y_val, tree_val_prob)

    linear_weight = max(0.05, _selection_score(linear_val_metrics))
    tree_weight = max(0.05, _selection_score(tree_val_metrics))
    weight_sum = linear_weight + tree_weight
    fusion_weights = {
        "linear": float(linear_weight / weight_sum),
        "tree": float(tree_weight / weight_sum),
    }

    fused_val_prob = (fusion_weights["linear"] * linear_val_prob) + (fusion_weights["tree"] * tree_val_prob)
    fused_test_prob = (fusion_weights["linear"] * calibrated_linear.predict_proba(x_test)[:, 1]) + (
        fusion_weights["tree"] * calibrated_tree.predict_proba(x_test)[:, 1]
    )

    metrics = {
        "validation": {
            "linear": linear_val_metrics,
            "tree": tree_val_metrics,
            "late_fusion": _safe_metrics(y_val, fused_val_prob),
        },
        "test": {
            "linear": _safe_metrics(y_test, calibrated_linear.predict_proba(x_test)[:, 1]),
            "tree": _safe_metrics(y_test, calibrated_tree.predict_proba(x_test)[:, 1]),
            "late_fusion": _safe_metrics(y_test, fused_test_prob),
        },
        "selected_models": {
            "linear": {
                "name": linear_result.name,
                "parameters": linear_result.parameters,
            },
            "tree": {
                "name": tree_result.name,
                "parameters": tree_result.parameters,
            },
        },
    }

    linear_result.estimator = calibrated_linear
    tree_result.estimator = calibrated_tree

    return TrainingResult(
        linear_result=linear_result,
        tree_result=tree_result,
        fusion_weights=fusion_weights,
        metrics=metrics,
        feature_columns=feature_columns,
        scaler_metadata=scaler_metadata,
    )


def write_training_artifacts(
    result: TrainingResult,
    model_output_dir: str | Path,
    metrics_output_path: str | Path,
    manifest_output_path: str | Path,
) -> None:
    output_dir = Path(model_output_dir)
    metrics_path = Path(metrics_output_path)
    manifest_path = Path(manifest_output_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    linear_model_path = output_dir / "linear_model_calibrated.pkl"
    tree_model_path = output_dir / "tree_model_calibrated.pkl"
    fusion_config_path = output_dir / "late_fusion_config.json"
    feature_columns_path = output_dir / "feature_columns.json"
    scaler_metadata_path = output_dir / "scaler_metadata.json"

    with linear_model_path.open("wb") as linear_handle:
        pickle.dump(result.linear_result.estimator, linear_handle)
    with tree_model_path.open("wb") as tree_handle:
        pickle.dump(result.tree_result.estimator, tree_handle)

    fusion_config = {
        "weights": result.fusion_weights,
        "linear_model": {
            "name": result.linear_result.name,
            "parameters": result.linear_result.parameters,
        },
        "tree_model": {
            "name": result.tree_result.name,
            "parameters": result.tree_result.parameters,
        },
    }
    fusion_config_path.write_text(json.dumps(fusion_config, indent=2), encoding="utf-8")
    feature_columns_path.write_text(json.dumps(result.feature_columns, indent=2), encoding="utf-8")

    if result.scaler_metadata is not None:
        scaler_metadata_path.write_text(json.dumps(result.scaler_metadata, indent=2), encoding="utf-8")

    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    manifest = {
        "linear_model_path": str(linear_model_path),
        "tree_model_path": str(tree_model_path),
        "fusion_config_path": str(fusion_config_path),
        "feature_columns_path": str(feature_columns_path),
        "metrics_path": str(metrics_path),
    }
    if result.scaler_metadata is not None:
        manifest["scaler_metadata_path"] = str(scaler_metadata_path)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
