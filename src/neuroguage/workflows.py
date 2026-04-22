from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .logging_config import configure_logging

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
        mirroring the Day 1 entry-point behaviour.
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
    )
    summary = run_live_app(config)
    manifest_payload["summary"] = summary
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")

    LOGGER.info(
        "run_live completed | source=%s | cycles=%d | mean_latency=%.3fs | final_state=%s",
        summary.get("source"),
        summary.get("cycles", 0),
        summary.get("mean_latency_s", 0.0),
        summary.get("final_state", "?"),
    )
    return WorkflowResult(
        workflow="run_live",
        status=manifest_payload["status"],
        manifest_path=manifest_path,
        details=manifest_payload,
    )


def run_replay(output_dir: str | Path = "artifacts/reports/day1/entrypoints", dry_run: bool = False) -> WorkflowResult:
    return _run_workflow("replay", output_dir=output_dir, dry_run=dry_run)
