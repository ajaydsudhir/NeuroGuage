from pathlib import Path
import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.training import run_training_pipeline, write_training_artifacts


def _read_optional_json(path: str | Path) -> dict | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Day 3 MVP models and calibrated late-fusion scorer.")
    parser.add_argument(
        "--labeled-input",
        default="artifacts/features/day3/labeled_feature_matrix.csv",
    )
    parser.add_argument(
        "--scaler-metadata-input",
        default="artifacts/reports/day3/normalization_metadata.json",
    )
    parser.add_argument(
        "--model-output-dir",
        default="artifacts/models/day3",
    )
    parser.add_argument(
        "--metrics-output",
        default="artifacts/reports/day3/training_metrics.json",
    )
    parser.add_argument(
        "--manifest-output",
        default="artifacts/reports/day3/training_manifest.json",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    labeled_frame = pd.read_csv(args.labeled_input)
    scaler_metadata = _read_optional_json(args.scaler_metadata_input)

    result = run_training_pipeline(
        labeled_frame=labeled_frame,
        random_state=args.random_state,
        scaler_metadata=scaler_metadata,
    )

    write_training_artifacts(
        result=result,
        model_output_dir=args.model_output_dir,
        metrics_output_path=args.metrics_output,
        manifest_output_path=args.manifest_output,
    )

    print(json.dumps(result.metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
