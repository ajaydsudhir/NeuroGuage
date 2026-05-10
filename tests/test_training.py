from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neuroguage.pipeline import run_feature_builder_pipeline
from neuroguage.pipeline import run_labeling_pipeline
from neuroguage.pipeline import run_training_pipeline, write_training_artifacts


FIXTURES = Path("tests/fixtures/windows")


def _labeled_frame() -> pd.DataFrame:
    eeg = pd.read_csv(FIXTURES / "eeg_window_features_sessions.csv")
    optical_imu = pd.read_csv(FIXTURES / "optical_imu_window_features_sessions.csv")
    annotations = pd.read_csv(FIXTURES / "session_annotations.csv")

    feature_result = run_feature_builder_pipeline(
        eeg_feature_frame=eeg,
        optical_imu_feature_frame=optical_imu,
        split_seed=42,
    )
    label_result = run_labeling_pipeline(
        feature_frame=feature_result.normalized_frame,
        annotation_frame=annotations,
        split_seed=42,
    )
    return label_result.labeled_frame


def test_training_pipeline_runs_with_calibrated_late_fusion() -> None:
    result = run_training_pipeline(
        labeled_frame=_labeled_frame(),
        random_state=42,
        scaler_metadata={"strategy": "train_only_zscore"},
    )

    assert result.linear_result.name in {"logistic_regression", "lda"}
    assert result.tree_result.name == "random_forest"
    assert 0.0 <= result.fusion_weights["linear"] <= 1.0
    assert 0.0 <= result.fusion_weights["tree"] <= 1.0
    assert abs((result.fusion_weights["linear"] + result.fusion_weights["tree"]) - 1.0) < 1e-9


def test_training_artifacts_are_persisted(tmp_path: Path) -> None:
    result = run_training_pipeline(
        labeled_frame=_labeled_frame(),
        random_state=42,
        scaler_metadata={"strategy": "train_only_zscore"},
    )

    model_dir = tmp_path / "models"
    metrics_path = tmp_path / "reports" / "metrics.json"
    manifest_path = tmp_path / "reports" / "manifest.json"

    write_training_artifacts(
        result=result,
        model_output_dir=model_dir,
        metrics_output_path=metrics_path,
        manifest_output_path=manifest_path,
    )

    assert (model_dir / "linear_model_calibrated.pkl").exists()
    assert (model_dir / "tree_model_calibrated.pkl").exists()
    assert (model_dir / "late_fusion_config.json").exists()
    assert (model_dir / "feature_columns.json").exists()
    assert metrics_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "linear_model_path" in manifest
    assert "tree_model_path" in manifest
