from pathlib import Path
import sys
import argparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.windowing import build_windowing_layers, write_windowing_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Day 2 windowing layer on synchronized session.")
    parser.add_argument(
        "--synchronized-input",
        default="artifacts/processed/day2/synchronized_session.csv",
    )
    parser.add_argument(
        "--eeg-output",
        default="artifacts/processed/day2/eeg_fast_windows.csv",
    )
    parser.add_argument(
        "--optical-output",
        default="artifacts/processed/day2/optical_slow_windows.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/reports/day2/windowing_summary.json",
    )
    parser.add_argument("--eeg-window-size", type=float, default=2.0)
    parser.add_argument("--eeg-step", type=float, default=0.5)
    parser.add_argument("--optical-window-size", type=float, default=8.0)
    parser.add_argument("--optical-step", type=float, default=2.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.synchronized_input)

    result = build_windowing_layers(
        synchronized_frame=frame,
        eeg_window_size_s=args.eeg_window_size,
        eeg_step_s=args.eeg_step,
        optical_window_size_s=args.optical_window_size,
        optical_step_s=args.optical_step,
    )

    write_windowing_artifacts(
        result=result,
        eeg_output_path=args.eeg_output,
        optical_output_path=args.optical_output,
        summary_output_path=args.summary_output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
