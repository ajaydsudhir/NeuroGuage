from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .training import TrainingResult, run_training_pipeline, write_training_artifacts


class EvaluationError(ValueError):
    """Raised when Day 4 MVP evaluation or ablation analysis cannot proceed."""


@dataclass
class AblationRun:
    name: str
    feature_columns: list[str]
    imu_gating_applied: bool
    gated_row_count: int
    training_result: TrainingResult
    robustness: dict[str, Any]
    composite_score: float


@dataclass
class EvaluationResult:
    ablation_runs: dict[str, AblationRun]
    selected_variant: str
    selection_rationale: str
    summary: dict[str, Any]


def _metric_or_zero(metric_value: float | None) -> float:
    return float(metric_value) if metric_value is not None else 0.0


def _select_z_feature_columns(frame: pd.DataFrame) -> list[str]:
    z_columns = sorted([column for column in frame.columns if column.startswith("z_")])
    if not z_columns:
        raise EvaluationError("No normalized feature columns found. Expected columns prefixed with 'z_'.")
    return z_columns


def build_ablation_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    z_columns = _select_z_feature_columns(frame)

    eeg_columns = sorted([column for column in z_columns if column.startswith("z_eeg_")])
    optical_columns = sorted([column for column in z_columns if column.startswith("z_optical_")])
    imu_columns = sorted([column for column in z_columns if column.startswith("z_imu_")])
    quality_columns = sorted(
        [
            column
            for column in z_columns
            if column in {"z_confidence_score", "z_downweight_factor", "z_alignment_gap_s"}
        ]
    )

    if not eeg_columns:
        raise EvaluationError("EEG normalized features were not found for EEG-only ablation")
    if not optical_columns:
        raise EvaluationError("Optical normalized features were not found for EEG+optical ablation")

    feature_sets = {
        "eeg_only": eeg_columns,
        "eeg_optical": sorted(eeg_columns + optical_columns),
        "eeg_optical_imu_gated": sorted(eeg_columns + optical_columns + imu_columns + quality_columns),
    }

    if not feature_sets["eeg_optical_imu_gated"]:
        raise EvaluationError("IMU-gated ablation feature set is empty")

    return feature_sets


def build_variant_scaler_metadata(
    base_scaler_metadata: dict[str, Any] | None,
    feature_columns: list[str],
) -> dict[str, Any] | None:
    if base_scaler_metadata is None:
        return None

    statistics = base_scaler_metadata.get("statistics")
    if not isinstance(statistics, dict):
        return None

    raw_feature_names = [column.removeprefix("z_") for column in feature_columns]
    variant_statistics = {
        feature_name: statistics[feature_name]
        for feature_name in raw_feature_names
        if feature_name in statistics
    }

    if not variant_statistics:
        return None

    return {
        "strategy": str(base_scaler_metadata.get("strategy", "train_only_zscore")),
        "split_column": str(base_scaler_metadata.get("split_column", "split")),
        "feature_columns": raw_feature_names,
        "statistics": variant_statistics,
    }


def apply_imu_quality_gate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    motion_score_column: str = "imu_motion_score",
    downweight_column: str = "downweight_factor",
    high_motion_threshold: float = 0.35,
) -> tuple[pd.DataFrame, int]:
    if motion_score_column not in frame.columns:
        raise EvaluationError(f"Missing required IMU motion score column: {motion_score_column}")
    if downweight_column not in frame.columns:
        raise EvaluationError(f"Missing required downweight column: {downweight_column}")

    gated = frame.copy()
    high_motion_mask = gated[motion_score_column].astype(float) >= float(high_motion_threshold)
    downweights = gated[downweight_column].astype(float).clip(0.2, 1.0)

    gated.loc[high_motion_mask, feature_columns] = gated.loc[high_motion_mask, feature_columns].mul(
        downweights.loc[high_motion_mask],
        axis=0,
    )

    return gated, int(high_motion_mask.sum())


def _build_robustness_summary(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    macro_gap = abs(float(validation_metrics["macro_f1"]) - float(test_metrics["macro_f1"]))
    balanced_gap = abs(
        float(validation_metrics["balanced_accuracy"]) - float(test_metrics["balanced_accuracy"])
    )
    auroc_gap = abs(_metric_or_zero(validation_metrics.get("auroc")) - _metric_or_zero(test_metrics.get("auroc")))

    consistency_penalty = macro_gap + (0.5 * balanced_gap) + (0.25 * auroc_gap)
    consistency_score = max(0.0, 1.0 - consistency_penalty)

    return {
        "macro_f1_gap": float(macro_gap),
        "balanced_accuracy_gap": float(balanced_gap),
        "auroc_gap": float(auroc_gap),
        "consistency_penalty": float(consistency_penalty),
        "consistency_score": float(consistency_score),
    }


def _build_composite_score(test_metrics: dict[str, Any], robustness: dict[str, Any]) -> float:
    perf_score = (
        0.5 * float(test_metrics["macro_f1"])
        + 0.3 * float(test_metrics["balanced_accuracy"])
        + 0.2 * _metric_or_zero(test_metrics.get("auroc"))
    )
    return float(perf_score - (0.2 * float(robustness["consistency_penalty"])))


def run_evaluation_pipeline(
    labeled_frame: pd.DataFrame,
    random_state: int = 42,
    base_scaler_metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    if labeled_frame.empty:
        raise EvaluationError("labeled_frame is empty")
    if "split" not in labeled_frame.columns:
        raise EvaluationError("labeled_frame is missing required 'split' column")
    if "fatigue_label" not in labeled_frame.columns:
        raise EvaluationError("labeled_frame is missing required 'fatigue_label' column")

    feature_sets = build_ablation_feature_sets(labeled_frame)

    ablation_runs: dict[str, AblationRun] = {}

    for variant_name, feature_columns in feature_sets.items():
        variant_frame = labeled_frame.copy()
        gated_row_count = 0

        if variant_name == "eeg_optical_imu_gated":
            variant_frame, gated_row_count = apply_imu_quality_gate(
                frame=variant_frame,
                feature_columns=feature_columns,
            )

        scaler_metadata = build_variant_scaler_metadata(
            base_scaler_metadata=base_scaler_metadata,
            feature_columns=feature_columns,
        )

        training_result = run_training_pipeline(
            labeled_frame=variant_frame,
            feature_columns=feature_columns,
            random_state=random_state,
            scaler_metadata=scaler_metadata,
        )

        validation_metrics = training_result.metrics["validation"]["late_fusion"]
        test_metrics = training_result.metrics["test"]["late_fusion"]
        robustness = _build_robustness_summary(validation_metrics=validation_metrics, test_metrics=test_metrics)
        composite_score = _build_composite_score(test_metrics=test_metrics, robustness=robustness)

        ablation_runs[variant_name] = AblationRun(
            name=variant_name,
            feature_columns=feature_columns,
            imu_gating_applied=variant_name == "eeg_optical_imu_gated",
            gated_row_count=gated_row_count,
            training_result=training_result,
            robustness=robustness,
            composite_score=composite_score,
        )

    selected_variant = max(
        ablation_runs,
        key=lambda variant: (
            ablation_runs[variant].composite_score,
            float(ablation_runs[variant].training_result.metrics["test"]["late_fusion"]["macro_f1"]),
        ),
    )
    selected_run = ablation_runs[selected_variant]
    selected_test_metrics = selected_run.training_result.metrics["test"]["late_fusion"]

    selection_rationale = (
        "Selected variant '{variant}' because it achieved the best composite score ({score:.4f}) "
        "while maintaining consistency (penalty={penalty:.4f}) and robust held-out performance "
        "(macro_f1={macro:.4f}, balanced_accuracy={balanced:.4f}, auroc={auroc})."
    ).format(
        variant=selected_variant,
        score=selected_run.composite_score,
        penalty=float(selected_run.robustness["consistency_penalty"]),
        macro=float(selected_test_metrics["macro_f1"]),
        balanced=float(selected_test_metrics["balanced_accuracy"]),
        auroc=selected_test_metrics.get("auroc"),
    )

    summary = {
        "ablation_count": int(len(ablation_runs)),
        "selected_variant": selected_variant,
        "selection_rationale": selection_rationale,
        "variant_scores": {
            name: {
                "composite_score": float(run.composite_score),
                "test_late_fusion": run.training_result.metrics["test"]["late_fusion"],
                "consistency_penalty": float(run.robustness["consistency_penalty"]),
            }
            for name, run in ablation_runs.items()
        },
    }

    return EvaluationResult(
        ablation_runs=ablation_runs,
        selected_variant=selected_variant,
        selection_rationale=selection_rationale,
        summary=summary,
    )


def write_evaluation_artifacts(
    result: EvaluationResult,
    model_output_root: str | Path,
    report_output_root: str | Path,
    ablation_report_output_path: str | Path,
    summary_output_path: str | Path,
    selection_output_path: str | Path,
    narrative_output_path: str | Path,
) -> None:
    model_root = Path(model_output_root)
    report_root = Path(report_output_root)
    ablation_report_path = Path(ablation_report_output_path)
    summary_path = Path(summary_output_path)
    selection_path = Path(selection_output_path)
    narrative_path = Path(narrative_output_path)

    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    ablation_report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    narrative_path.parent.mkdir(parents=True, exist_ok=True)

    ablation_payload: dict[str, Any] = {
        "ablations": {},
    }

    for variant_name, ablation_run in result.ablation_runs.items():
        variant_model_dir = model_root / "ablations" / variant_name
        variant_report_dir = report_root / "ablations"
        variant_report_dir.mkdir(parents=True, exist_ok=True)

        variant_metrics_path = variant_report_dir / f"{variant_name}_training_metrics.json"
        variant_manifest_path = variant_report_dir / f"{variant_name}_training_manifest.json"

        write_training_artifacts(
            result=ablation_run.training_result,
            model_output_dir=variant_model_dir,
            metrics_output_path=variant_metrics_path,
            manifest_output_path=variant_manifest_path,
        )

        ablation_payload["ablations"][variant_name] = {
            "feature_columns": ablation_run.feature_columns,
            "feature_column_count": int(len(ablation_run.feature_columns)),
            "imu_gating_applied": bool(ablation_run.imu_gating_applied),
            "gated_row_count": int(ablation_run.gated_row_count),
            "robustness": ablation_run.robustness,
            "composite_score": float(ablation_run.composite_score),
            "metrics": ablation_run.training_result.metrics,
            "training_manifest_path": str(variant_manifest_path),
        }

    selected_variant_payload = ablation_payload["ablations"][result.selected_variant]

    selection_payload = {
        "selected_variant": result.selected_variant,
        "selected_training_manifest_path": selected_variant_payload["training_manifest_path"],
        "selected_models": selected_variant_payload["metrics"]["selected_models"],
        "selected_test_metrics": selected_variant_payload["metrics"]["test"]["late_fusion"],
        "selected_composite_score": selected_variant_payload["composite_score"],
        "selection_rationale": result.selection_rationale,
    }

    ranked_variants = sorted(
        result.ablation_runs.values(),
        key=lambda run: run.composite_score,
        reverse=True,
    )

    narrative_lines = [
        "# Day 4 MVP Evaluation Summary",
        "",
        "## Production Selection",
        "",
        f"Selected variant: {result.selected_variant}",
        f"Reason: {result.selection_rationale}",
        "",
        "## Variant Ranking (Best to Worst)",
        "",
    ]
    for rank, variant_run in enumerate(ranked_variants, start=1):
        metrics = variant_run.training_result.metrics["test"]["late_fusion"]
        narrative_lines.append(
            (
                f"{rank}. {variant_run.name}: composite={variant_run.composite_score:.4f}, "
                f"macro_f1={float(metrics['macro_f1']):.4f}, "
                f"balanced_accuracy={float(metrics['balanced_accuracy']):.4f}, "
                f"auroc={metrics.get('auroc')}"
            )
        )

    ablation_report_path.write_text(json.dumps(ablation_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    selection_path.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    narrative_path.write_text("\n".join(narrative_lines) + "\n", encoding="utf-8")
