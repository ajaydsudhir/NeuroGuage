"""Day 4 — T11 MVP Evaluation and Ablation Report runner.

Subtasks covered:
  T11.1  Compute AUROC, macro F1, balanced accuracy on held-out sessions.
  T11.2  Run ablations: EEG-only, EEG+optical, EEG+optical+IMU-gated.
  T11.3  Select production MVP model based on robustness and metric consistency.
  T11.4  Publish evaluation summary for demo narrative and technical review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap so the script can run from repository root.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.evaluation import run_evaluation_pipeline, write_evaluation_artifacts  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELED_FEATURE_PATH = Path("artifacts/features/day3/labeled_feature_matrix.csv")
SCALER_METADATA_PATH = Path("artifacts/models/day3/scaler_metadata.json")

MODEL_OUTPUT_ROOT = Path("artifacts/models/day4")
REPORT_OUTPUT_ROOT = Path("artifacts/reports/day4")

ABLATION_REPORT_PATH = REPORT_OUTPUT_ROOT / "ablation_report.json"
SUMMARY_PATH = REPORT_OUTPUT_ROOT / "evaluation_summary.json"
SELECTION_PATH = REPORT_OUTPUT_ROOT / "model_selection.json"
NARRATIVE_PATH = REPORT_OUTPUT_ROOT / "evaluation_narrative.md"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import pandas as pd

    print("[T11] Day 4 MVP Evaluation and Ablation Report")
    print("=" * 60)

    # --- Load labeled features from Day 3 ---
    if not LABELED_FEATURE_PATH.exists():
        print(f"ERROR: labeled feature matrix not found at {LABELED_FEATURE_PATH}")
        return 1

    labeled_frame = pd.read_csv(LABELED_FEATURE_PATH)
    print(f"  Loaded labeled feature matrix: {len(labeled_frame)} rows")

    # --- Load scaler metadata ---
    scaler_metadata = None
    if SCALER_METADATA_PATH.exists():
        scaler_metadata = json.loads(SCALER_METADATA_PATH.read_text(encoding="utf-8"))
        print(f"  Loaded scaler metadata from {SCALER_METADATA_PATH}")
    else:
        print("  WARNING: scaler metadata not found, proceeding without it")

    # --- Run evaluation pipeline (T11.1, T11.2, T11.3) ---
    print("\n[T11.1] Computing AUROC, macro F1, balanced accuracy on held-out sessions...")
    print("[T11.2] Running ablations: EEG-only, EEG+optical, EEG+optical+IMU-gated...")

    result = run_evaluation_pipeline(
        labeled_frame=labeled_frame,
        random_state=42,
        base_scaler_metadata=scaler_metadata,
    )

    print(f"\n[T11.3] Production MVP model selected: {result.selected_variant}")
    print(f"  Rationale: {result.selection_rationale}")

    # --- Print summary table ---
    print("\n  Ablation Results:")
    print(f"  {'Variant':<28} {'Composite':>10} {'Macro-F1':>10} {'Bal-Acc':>10} {'AUROC':>10} {'Penalty':>10}")
    print("  " + "-" * 78)
    for name, run in sorted(result.ablation_runs.items(), key=lambda x: x[1].composite_score, reverse=True):
        test_metrics = run.training_result.metrics["test"]["late_fusion"]
        print(
            f"  {name:<28} {run.composite_score:>10.4f} "
            f"{float(test_metrics['macro_f1']):>10.4f} "
            f"{float(test_metrics['balanced_accuracy']):>10.4f} "
            f"{test_metrics.get('auroc', 'N/A'):>10} "
            f"{float(run.robustness['consistency_penalty']):>10.4f}"
        )

    # --- Write all artifacts (T11.4) ---
    print(f"\n[T11.4] Publishing evaluation artifacts to {REPORT_OUTPUT_ROOT}/")
    write_evaluation_artifacts(
        result=result,
        model_output_root=MODEL_OUTPUT_ROOT,
        report_output_root=REPORT_OUTPUT_ROOT,
        ablation_report_output_path=ABLATION_REPORT_PATH,
        summary_output_path=SUMMARY_PATH,
        selection_output_path=SELECTION_PATH,
        narrative_output_path=NARRATIVE_PATH,
    )

    print("\n  Artifacts written:")
    for path in [ABLATION_REPORT_PATH, SUMMARY_PATH, SELECTION_PATH, NARRATIVE_PATH]:
        status = "OK" if path.exists() else "MISSING"
        print(f"    [{status}] {path}")

    print("\n[T11] Day 4 MVP Evaluation COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
