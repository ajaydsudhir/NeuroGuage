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
    args = _common_workflow_parser("Run NeuroGauge live inference workflow.").parse_args()
    run_live(output_dir=args.output_dir, dry_run=args.dry_run)
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
