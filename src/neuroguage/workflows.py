from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    workflow: str
    status: str
    manifest_path: Path


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


def run_live(output_dir: str | Path = "artifacts/reports/day1/entrypoints", dry_run: bool = False) -> WorkflowResult:
    return _run_workflow("run_live", output_dir=output_dir, dry_run=dry_run)


def run_replay(output_dir: str | Path = "artifacts/reports/day1/entrypoints", dry_run: bool = False) -> WorkflowResult:
    return _run_workflow("replay", output_dir=output_dir, dry_run=dry_run)
