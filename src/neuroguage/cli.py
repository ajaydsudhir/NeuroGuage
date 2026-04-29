"""Primary command entrypoints for training, evaluation, and live runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .pipeline import validate_session_files, write_validation_summary


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
        help="Path to a guardrail-thresholds JSON (for example "
             "configs/guardrails_muse_demo.json or configs/guardrails_research_eegnet.json). "
             "Defaults to compiled-in values.",
    )

    # ---- Fix B ---------------------------------------------------------
    parser.add_argument(
        "--baseline-duration", dest="baseline_duration_s", type=float, default=0.0,
        help="Seconds to spend collecting a personal-baseline of the model's risk "
             "output before guardrails fire. 0 disables calibration for compatibility. "
             "Recommended: 20.0 paired with --guardrail-config configs/guardrails_muse_demo.json.",
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

    # ---- Dual-head neuro-symbolic mismatch detector --------------------
    parser.add_argument(
        "--use-dual-head", dest="use_dual_head", action="store_true",
        help="Run the dual-head PyTorch model (P(attention) + P(metabolic_ready)) "
             "alongside the existing fatigue gauge and overlay the symbolic "
             "MISMATCH alert on the dashboard.",
    )
    parser.add_argument(
        "--dual-head-weights", dest="dual_head_weights_path",
        default="artifacts/models/dual_head/dual_head.pt",
        help="Path to the trained dual-head .pt weights "
             "(produced by `python -m neuroguage.cli train-dual-head`).",
    )
    parser.add_argument(
        "--dual-head-config", dest="dual_head_config_path",
        default="artifacts/models/dual_head/config.json",
        help="Path to the matching dual-head config.json.",
    )
    parser.add_argument(
        "--dual-head-device", dest="dual_head_device", default="cpu",
        help="Torch device for dual-head inference (default: cpu).",
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
        use_dual_head=args.use_dual_head,
        dual_head_weights_path=args.dual_head_weights_path,
        dual_head_config_path=args.dual_head_config_path,
        dual_head_device=args.dual_head_device,
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


def main_train_dual_head() -> int:
    """Train the dual-head neuro-symbolic fatigue model.

    Wires the same arguments the standalone ``scripts/train_dual_head.py``
    script accepts so users can invoke training via either entry point.
    """
    import logging  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from .dual_head import (  # noqa: PLC0415
        DualHeadLabelConfig,
        derive_dual_head_labels,
        summarise_dual_head_labels,
    )
    from .dual_head import (  # noqa: PLC0415
        DualHeadConfig,
        train_dual_head_model,
        write_dual_head_artifacts,
    )

    parser = argparse.ArgumentParser(description="Train the dual-head model.")
    parser.add_argument(
        "--labeled-frame",
        default="artifacts/features/day3/labeled_feature_matrix.csv",
    )
    parser.add_argument("--output-dir", default="artifacts/models/dual_head")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument(
        "--model-variant",
        choices=("muse_demo", "research_eegnet"),
        default="muse_demo",
        help="Dual-head architecture variant.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mismatch-loss-weight", type=float, default=0.5)
    parser.add_argument("--attention-threshold", type=float, default=0.5)
    parser.add_argument("--metabolic-threshold", type=float, default=0.5)
    args = parser.parse_args()

    configure_logging(run_name="train_dual_head")
    log = logging.getLogger("neuroguage.cli.train_dual_head")

    csv_path = Path(args.labeled_frame)
    if not csv_path.exists():
        log.error("Labeled frame not found: %s", csv_path)
        return 2

    frame = pd.read_csv(csv_path)
    label_config = DualHeadLabelConfig(
        attention_threshold=args.attention_threshold,
        metabolic_threshold=args.metabolic_threshold,
    )
    labelled = derive_dual_head_labels(frame, config=label_config)
    label_summary = summarise_dual_head_labels(labelled)
    log.info("Weak label distribution: %s", json.dumps(label_summary, indent=2))

    dual_config = DualHeadConfig(
        model_variant=args.model_variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        mismatch_loss_weight=args.mismatch_loss_weight,
    )
    result = train_dual_head_model(labelled, config=dual_config)

    output_dir = Path(args.output_dir)
    artifacts = write_dual_head_artifacts(
        result,
        model_dir=output_dir,
        metrics_path=output_dir / "metrics.json",
        training_log_path=output_dir / "training_log.json",
    )
    summary = {
        "labeled_frame": str(csv_path),
        "model_variant": args.model_variant,
        "weak_labels": label_summary,
        "metrics": result.metrics,
        "artifacts": artifacts,
        "epochs_trained": len(result.epoch_log),
        "train_sessions": result.train_sessions,
        "val_sessions": result.val_sessions,
        "test_sessions": result.test_sessions,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info(
        "Training complete | val composite=%.4f | test mismatch F1=%.3f",
        result.metrics.get("best_validation_composite_score", 0.0),
        result.metrics["test"]["mismatch"]["macro_f1"],
    )
    return 0


def main_decode_raw() -> int:
    """Convert a raw ``OpenMuse record`` .txt file into session-contract CSVs."""
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
    parser = argparse.ArgumentParser(description="Validate NeuroGauge session schema.")
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


# Dispatcher — enables `python -m neuroguage.cli <subcommand>`.

_SUBCOMMANDS = {
    "train": main_train,
    "train-dual-head": main_train_dual_head,
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


# --- Consolidated from logging_config.py ---
import logging
from datetime import datetime
from pathlib import Path


def configure_logging(run_name: str, log_dir: str | Path = "logs", level: int = logging.INFO) -> Path:
    """Configure shared console and file logging for all workflows."""
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)

    log_path = log_directory / f"{run_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Reset handlers to keep repeated CLI calls from duplicating log output.
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_path


# --- Consolidated from workflows.py ---
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    workflow: str
    status: str
    manifest_path: Path
    details: dict[str, Any] = field(default_factory=dict)


def _run_workflow(
    workflow_name: str,
    output_dir: str | Path,
    dry_run: bool,
    extra_config: dict[str, Any] | None = None,
) -> WorkflowResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_name = f"{workflow_name}_workflow"
    log_path = configure_logging(run_name=run_name)

    timestamp = datetime.now(timezone.utc).isoformat()
    manifest_path = output_path / f"{workflow_name}_manifest.json"

    manifest = {
        "workflow": workflow_name,
        "status": "dry-run" if dry_run else "executed",
        "timestamp_utc": timestamp,
        "log_path": str(log_path),
        "extra_config": extra_config or {},
    }

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    LOGGER.info("Workflow '%s' completed with status '%s'.", workflow_name, manifest["status"])
    LOGGER.info("Manifest written to %s", manifest_path)

    return WorkflowResult(
        workflow=workflow_name,
        status=manifest["status"],
        manifest_path=manifest_path,
    )


def run_train(output_dir: str | Path = "artifacts/reports/day1/entrypoints", dry_run: bool = False) -> WorkflowResult:
    return _run_workflow("train", output_dir=output_dir, dry_run=dry_run)


def run_evaluate(output_dir: str | Path = "artifacts/reports/day1/entrypoints", dry_run: bool = False) -> WorkflowResult:
    return _run_workflow("evaluate", output_dir=output_dir, dry_run=dry_run)


def run_live(
    output_dir: str | Path = "artifacts/reports/live",
    dry_run: bool = False,
    source_kind: str = "synthetic",
    device_address: Optional[str] = None,
    duration_s: Optional[float] = None,
    session_id: str = "live",
    raw_file: Optional[str] = None,
    preset: str = "p1041",
    dashboard: bool = True,
    record_raw_to: Optional[str] = None,
    real_time_factor: float = 1.0,
    fatigue_trend: float = 0.5,
    manage_bridge: bool = True,
    resolve_timeout_s: float = 60.0,
    guardrail_config_path: Optional[str] = None,
    baseline_duration_s: float = 0.0,
    baseline_stdev_floor: float = 0.04,
    baseline_sigma_span: float = 4.0,
    save_baseline_to: Optional[str] = None,
    load_baseline_from: Optional[str] = None,
    use_dual_head: bool = False,
    dual_head_weights_path: Optional[str] = "artifacts/models/dual_head/dual_head.pt",
    dual_head_config_path: Optional[str] = "artifacts/models/dual_head/config.json",
    dual_head_device: str = "cpu",
) -> WorkflowResult:
    """Run the NeuroGauge live fatigue monitor.

    Parameters
    ----------
    source_kind : str
        One of ``"muse"``, ``"synthetic"`` or ``"replay"``.
    device_address : str, optional
        MAC address of the Muse headband (required when ``source_kind='muse'``).
    duration_s : float, optional
        Maximum session length.  ``None`` runs until Ctrl+C or source end.
    dry_run : bool
        When True the workflow only emits a manifest and exits immediately,
        mirroring the entry-point behaviour.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(run_name="run_live_workflow")

    manifest_payload: dict[str, Any] = {
        "workflow": "run_live",
        "status": "dry-run" if dry_run else "executed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
        "source_kind": source_kind,
        "device_address": device_address,
        "preset": preset,
        "session_id": session_id,
        "duration_s": duration_s,
        "raw_file": raw_file,
    }
    manifest_path = output_path / "run_live_manifest.json"

    if dry_run:
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        LOGGER.info("run_live dry-run manifest written to %s", manifest_path)
        return WorkflowResult(
            workflow="run_live",
            status=manifest_payload["status"],
            manifest_path=manifest_path,
            details=manifest_payload,
        )

    # Local import keeps heavy deps (matplotlib, mne_lsl) out of --dry-run paths.
    from .live_app import LiveAppConfig, run_live_app  # noqa: PLC0415

    config = LiveAppConfig(
        source_kind=source_kind,
        device_address=device_address,
        preset=preset,
        session_id=session_id,
        duration_s=duration_s,
        raw_file=raw_file,
        output_dir=output_dir,
        enable_dashboard=dashboard,
        dashboard_block=dashboard,
        record_raw_to=record_raw_to,
        real_time_factor=real_time_factor,
        fatigue_trend=fatigue_trend,
        manage_bridge=manage_bridge,
        resolve_timeout_s=resolve_timeout_s,
        guardrail_config_path=guardrail_config_path,
        baseline_duration_s=baseline_duration_s,
        baseline_stdev_floor=baseline_stdev_floor,
        baseline_sigma_span=baseline_sigma_span,
        save_baseline_to=save_baseline_to,
        load_baseline_from=load_baseline_from,
        use_dual_head=use_dual_head,
        dual_head_weights_path=dual_head_weights_path,
        dual_head_config_path=dual_head_config_path,
        dual_head_device=dual_head_device,
    )
    summary = run_live_app(config)
    manifest_payload["summary"] = summary
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")

    profile = (
        (summary.get("guardrail_profile") or {}).get("name") or "default"
    )
    cal = summary.get("calibration") or {}
    cal_label = "personalised" if cal.get("baseline_ready") else (
        "calibrating" if cal.get("enabled") else "raw"
    )
    LOGGER.info(
        "run_live completed | source=%s | cycles=%d | mean_latency=%.3fs | "
        "final_state=%s | profile=%s | calibration=%s",
        summary.get("source"),
        summary.get("cycles", 0),
        summary.get("mean_latency_s", 0.0),
        summary.get("final_state", "?"),
        profile,
        cal_label,
    )
    return WorkflowResult(
        workflow="run_live",
        status=manifest_payload["status"],
        manifest_path=manifest_path,
        details=manifest_payload,
    )


def run_replay(output_dir: str | Path = "artifacts/reports/day1/entrypoints", dry_run: bool = False) -> WorkflowResult:
    return _run_workflow("replay", output_dir=output_dir, dry_run=dry_run)
