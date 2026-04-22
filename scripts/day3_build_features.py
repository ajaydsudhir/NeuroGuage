from pathlib import Path
import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.feature_builder import run_feature_builder_pipeline, write_feature_builder_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Day 3 unified and normalized feature matrices.")
    parser.add_argument(
        "--eeg-features-input",
        default="tests/fixtures/day3/eeg_window_features_sessions.csv",
    )
    parser.add_argument(
        "--optical-imu-features-input",
        default="tests/fixtures/day3/optical_imu_window_features_sessions.csv",
    )
    parser.add_argument(
        "--unified-output",
        default="artifacts/features/day3/unified_feature_matrix.csv",
    )
    parser.add_argument(
        "--normalized-output",
        default="artifacts/features/day3/unified_feature_matrix_normalized.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/reports/day3/feature_builder_summary.json",
    )
    parser.add_argument(
        "--normalization-output",
        default="artifacts/reports/day3/normalization_metadata.json",
    )
    parser.add_argument(
        "--leakage-output",
        default="artifacts/reports/day3/feature_leakage_report.json",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--alignment-tolerance-s", type=float, default=2.5)
    parser.add_argument("--dataset-version", default="day3_v1")
    parser.add_argument("--feature-schema-version", default="v1")
    args = parser.parse_args()

    eeg_features = pd.read_csv(args.eeg_features_input)
    optical_imu_features = pd.read_csv(args.optical_imu_features_input)

    result = run_feature_builder_pipeline(
        eeg_feature_frame=eeg_features,
        optical_imu_feature_frame=optical_imu_features,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        alignment_tolerance_s=args.alignment_tolerance_s,
        dataset_version=args.dataset_version,
        feature_schema_version=args.feature_schema_version,
    )

    write_feature_builder_artifacts(
        result=result,
        unified_output_path=args.unified_output,
        normalized_output_path=args.normalized_output,
        summary_output_path=args.summary_output,
        normalization_output_path=args.normalization_output,
        leakage_output_path=args.leakage_output,
    )

    print(json.dumps(result.summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
