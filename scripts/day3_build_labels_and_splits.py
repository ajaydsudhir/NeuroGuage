from pathlib import Path
import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.labeling import run_labeling_pipeline, write_labeling_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Day 3 weak labels and session-held-out split manifests.")
    parser.add_argument(
        "--feature-input",
        default="artifacts/features/day3/unified_feature_matrix_normalized.csv",
    )
    parser.add_argument(
        "--annotation-input",
        default="tests/fixtures/day3/session_annotations.csv",
    )
    parser.add_argument(
        "--labeled-output",
        default="artifacts/features/day3/labeled_feature_matrix.csv",
    )
    parser.add_argument(
        "--split-manifest-output",
        default="artifacts/reports/day3/split_manifest.json",
    )
    parser.add_argument(
        "--split-table-output",
        default="artifacts/reports/day3/split_manifest.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/reports/day3/labeling_summary.json",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--disable-smoothing", action="store_true")
    parser.add_argument("--smoothing-window", type=int, default=3)
    args = parser.parse_args()

    feature_frame = pd.read_csv(args.feature_input)
    annotations = pd.read_csv(args.annotation_input)

    result = run_labeling_pipeline(
        feature_frame=feature_frame,
        annotation_frame=annotations,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        enable_smoothing=not args.disable_smoothing,
        smoothing_window=args.smoothing_window,
    )

    write_labeling_artifacts(
        result=result,
        labeled_output_path=args.labeled_output,
        split_manifest_output_path=args.split_manifest_output,
        split_table_output_path=args.split_table_output,
        summary_output_path=args.summary_output,
    )

    print(json.dumps(result.summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
