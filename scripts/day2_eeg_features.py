from pathlib import Path
import sys
import argparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.eeg_features import extract_eeg_window_features, write_eeg_feature_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Day 2 EEG preprocessing and feature extraction.")
    parser.add_argument(
        "--synchronized-input",
        default="artifacts/processed/day2/synchronized_session.csv",
    )
    parser.add_argument(
        "--eeg-windows-input",
        default="artifacts/processed/day2/eeg_fast_windows.csv",
    )
    parser.add_argument(
        "--features-output",
        default="artifacts/processed/day2/eeg_window_features.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/reports/day2/eeg_feature_summary.json",
    )
    args = parser.parse_args()

    synchronized = pd.read_csv(args.synchronized_input)
    eeg_windows = pd.read_csv(args.eeg_windows_input)

    result = extract_eeg_window_features(
        synchronized_frame=synchronized,
        eeg_windows=eeg_windows,
    )

    write_eeg_feature_artifacts(
        result=result,
        feature_output_path=args.features_output,
        summary_output_path=args.summary_output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
