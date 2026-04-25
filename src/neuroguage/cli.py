from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .schema_validator import validate_session_files, write_validation_summary
from .workflows import run_evaluate, run_live, run_replay, run_train


def _common_workflow_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports/day1/entrypoints",
        help="Directory for workflow manifest output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run command in dry-run mode for entry-point verification.",
    )
    return parser


def main_train() -> int:
    args = _common_workflow_parser("Run NeuroGauge training workflow.").parse_args()
    run_train(output_dir=args.output_dir, dry_run=args.dry_run)
    return 0


def main_evaluate() -> int:
    args = _common_workflow_parser("Run NeuroGauge evaluation workflow.").parse_args()
    run_evaluate(output_dir=args.output_dir, dry_run=args.dry_run)
    return 0


def main_run_live() -> int:
    parser = argparse.ArgumentParser(description="Run NeuroGauge live inference workflow.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports/live",
        help="Directory for session artifacts (logs, summary, transitions).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip streaming and only emit the run manifest (entry-point verification).",
    )
    parser.add_argument(
        "--source",
        dest="source_kind",
        choices=("muse", "synthetic", "replay"),
        default="synthetic",
        help="Data source: 'muse' (live Muse S Athena via OpenMuse), "
             "'synthetic' (fake signals), or 'replay' (decoded raw recording).",
    )
    parser.add_argument("--address", dest="device_address", default=None,
                        help="MAC address of the Muse S Athena (required for --source muse).")
    parser.add_argument("--preset", default="p1041",
                        help="OpenMuse preset (default: p1041 = EEG8 + Optics16 + ACC/GYRO).")
    parser.add_argument("--duration", dest="duration_s", type=float, default=None,
                        help="Max session duration in seconds (None = until Ctrl+C).")
    parser.add_argument("--session-id", default="live", help="Label applied to output artifacts.")
    parser.add_argument("--raw-file", default=None,
                        help="Path to a raw OpenMuse recording (required for --source replay).")
    parser.add_argument("--no-dashboard", dest="dashboard", action="store_false",
                        help="Run headlessly (no live matplotlib window).")
    parser.add_argument("--record-raw", dest="record_raw_to", default=None,
                        help="Optional raw BLE packet log (.txt) for later replay/calibration.")
    parser.add_argument("--real-time-factor", type=float, default=1.0,
                        help="Speed multiplier for synthetic/replay sources (1.0 = real time).")
    parser.add_argument("--fatigue-trend", type=float, default=0.5,
                        help="Synthetic source fatigue slope (0.0 alert .. 1.0 drowsy).")
    parser.add_argument("--use-existing-stream", dest="manage_bridge",
                        action="store_false",
                        help="Attach to an `OpenMuse stream` already running in another "
                             "terminal instead of spawning a new BLE bridge.")
    parser.add_argument("--resolve-timeout", dest="resolve_timeout_s", type=float, default=60.0,
                        help="Max seconds to wait for all LSL outlets to come online.")

    # ---- Fix A ---------------------------------------------------------
    parser.add_argument(
        "--guardrail-config", dest="guardrail_config_path", default=None,
        help="Path to a guardrail-thresholds JSON (e.g. configs/guardrails_calibrated.json "
             "or configs/guardrails_personalized.json). Defaults to compiled-in values.",
    )

    # ---- Fix B ---------------------------------------------------------
    parser.add_argument(
        "--baseline-duration", dest="baseline_duration_s", type=float, default=0.0,
        help="Seconds to spend collecting a personal-baseline of the model's risk "
             "output before guardrails fire. 0 disables calibration (back-compat). "
             "Recommended: 20.0 paired with --guardrail-config configs/guardrails_personalized.json.",
    )
    parser.add_argument(
        "--baseline-stdev-floor", dest="baseline_stdev_floor", type=float, default=0.04,
        help="Minimum stdev floor when z-scoring (prevents oversensitivity to noise).",
    )
    parser.add_argument(
        "--baseline-sigma-span", dest="baseline_sigma_span", type=float, default=4.0,
        help="Sigma multiplier mapped to the upper edge of the normalised range.",
    )
    parser.add_argument(
        "--save-baseline", dest="save_baseline_to", default=None,
        help="Write the personal-baseline JSON to this path instead of the "
             "auto-generated <session>_baseline.json under the output directory.",
    )
    parser.add_argument(
        "--load-baseline", dest="load_baseline_from", default=None,
        help="Reuse a previously-saved personal baseline (skips the "
             "calibration window for repeat demos).",
    )

    args = parser.parse_args()

    run_live(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        source_kind=args.source_kind,
        device_address=args.device_address,
        preset=args.preset,
        duration_s=args.duration_s,
        session_id=args.session_id,
        raw_file=args.raw_file,
        dashboard=args.dashboard,
        record_raw_to=args.record_raw_to,
        real_time_factor=args.real_time_factor,
        fatigue_trend=args.fatigue_trend,
        manage_bridge=args.manage_bridge,
        resolve_timeout_s=args.resolve_timeout_s,
        guardrail_config_path=args.guardrail_config_path,
        baseline_duration_s=args.baseline_duration_s,
        baseline_stdev_floor=args.baseline_stdev_floor,
        baseline_sigma_span=args.baseline_sigma_span,
        save_baseline_to=args.save_baseline_to,
        load_baseline_from=args.load_baseline_from,
    )
    return 0


def main_replay() -> int:
    args = _common_workflow_parser("Run NeuroGauge replay workflow.").parse_args()
    run_replay(output_dir=args.output_dir, dry_run=args.dry_run)
    return 0


def main_verify_env() -> int:
    parser = argparse.ArgumentParser(description="Verify the runtime Conda environment for NeuroGauge.")
    parser.add_argument("--expected-env", default="neuroguage")
    parser.add_argument(
        "--output",
        default="artifacts/reports/day1/environment_verification.json",
        help="Path to write verification artifact.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when active conda env does not match expected env.",
    )
    args = parser.parse_args()

    active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    result = {
        "expected_env": args.expected_env,
        "active_env": active_env,
        "match": active_env == args.expected_env,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    if args.strict and not result["match"]:
        return 1
    return 0


def main_decode_raw() -> int:
    """Convert a raw ``OpenMuse record`` .txt file into Day 2 contract CSVs."""
    from .muse_stream import decode_raw_session_to_contract  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Decode an OpenMuse raw recording into eeg/optical/imu CSVs."
    )
    parser.add_argument("--raw-file", required=True, help="Path to the .txt recording file.")
    parser.add_argument("--session-id", required=True, help="Session identifier to stamp rows.")
    parser.add_argument("--output-dir", required=True, help="Directory to write CSVs + metadata.")
    parser.add_argument("--task-type", default="sustained_attention")
    parser.add_argument("--subjective-fatigue-score", type=float, default=5.0)
    args = parser.parse_args()

    paths = decode_raw_session_to_contract(
        raw_file=args.raw_file,
        session_id=args.session_id,
        output_dir=args.output_dir,
        task_type=args.task_type,
        subjective_fatigue_score=args.subjective_fatigue_score,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def main_discover_muse() -> int:
    """Print the MAC address of every discoverable Muse device."""
    from .muse_stream import discover_muse_devices  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Scan for nearby Muse devices.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Scan timeout in seconds.")
    args = parser.parse_args()

    devices = discover_muse_devices(timeout=args.timeout, verbose=True)
    print(json.dumps(devices, indent=2))
    return 0 if devices else 1


def main_validate_schema() -> int:
    parser = argparse.ArgumentParser(description="Validate NeuroGauge Day 1 session schema.")
    parser.add_argument("--eeg", required=True)
    parser.add_argument("--optical", required=True)
    parser.add_argument("--imu", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--contract", default="configs/data_contract.json")
    parser.add_argument(
        "--summary-output",
        default="artifacts/reports/day1/schema_validation_summary.json",
    )
    args = parser.parse_args()

    summary = validate_session_files(
        eeg_path=args.eeg,
        optical_path=args.optical,
        imu_path=args.imu,
        metadata_path=args.metadata,
        contract_path=args.contract,
        fail_fast=True,
    )
    write_validation_summary(summary, args.summary_output)
    return 0


# ---------------------------------------------------------------------------
# Dispatcher — enables `python -m neuroguage.cli <subcommand>`.
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {
    "train": main_train,
    "evaluate": main_evaluate,
    "run-live": main_run_live,
    "replay": main_replay,
    "verify-env": main_verify_env,
    "validate-schema": main_validate_schema,
    "discover-muse": main_discover_muse,
    "decode-raw": main_decode_raw,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``python -m neuroguage.cli <subcommand> …``."""
    import sys  # noqa: PLC0415

    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: python -m neuroguage.cli <subcommand> [options]")
        print("Subcommands:")
        for name in _SUBCOMMANDS:
            print(f"  {name}")
        return 0 if args else 1

    subcommand = args[0]
    entrypoint = _SUBCOMMANDS.get(subcommand)
    if entrypoint is None:
        print(f"Unknown subcommand: {subcommand}", file=sys.stderr)
        print("Run with --help for the subcommand list.", file=sys.stderr)
        return 2

    # Rewrite argv so each ``main_*`` parser sees only its own args.
    sys.argv = [f"neuroguage.cli {subcommand}", *args[1:]]
    return int(entrypoint() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
