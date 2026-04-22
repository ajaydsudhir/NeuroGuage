from pathlib import Path
import sys
import json
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.ingestion import synchronize_modalities, write_synchronization_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Day 2 multimodal ingestion and timeline alignment.")
    parser.add_argument("--eeg", required=True)
    parser.add_argument("--optical", required=True)
    parser.add_argument("--imu", required=True)
    parser.add_argument("--contract", default="configs/data_contract.json")
    parser.add_argument("--tolerance-s", type=float, default=0.2)
    parser.add_argument(
        "--synchronized-output",
        default="artifacts/processed/day2/synchronized_session.csv",
    )
    parser.add_argument(
        "--diagnostics-output",
        default="artifacts/reports/day2/alignment_diagnostics.json",
    )
    parser.add_argument(
        "--validation-output",
        default="artifacts/reports/day2/synchronization_validation_summary.json",
    )
    args = parser.parse_args()

    result = synchronize_modalities(
        eeg_path=args.eeg,
        optical_path=args.optical,
        imu_path=args.imu,
        contract_path=args.contract,
        tolerance_s=args.tolerance_s,
    )

    write_synchronization_artifacts(
        result=result,
        synchronized_output_path=args.synchronized_output,
        diagnostics_output_path=args.diagnostics_output,
        validation_output_path=args.validation_output,
    )

    print(json.dumps(result.validation_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
