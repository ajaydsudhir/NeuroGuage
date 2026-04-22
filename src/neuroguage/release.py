"""Day 6 — T17.4  Release Packaging and Evidence Index.

Collects evaluation outputs, config snapshots, and acceptance evidence into
a single reproducible release folder with an indexed manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

@dataclass
class ReleaseManifest:
    """Summary manifest for a release bundle."""

    version: str
    created_at: str
    commit: str | None
    entries: list[dict[str, Any]]
    evidence: dict[str, str]
    acceptance: dict[str, Any]
    gate_decisions: dict[str, Any]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def build_release_bundle(
    release_dir: Path,
    version: str = "0.1.0",
    commit: str | None = None,
) -> ReleaseManifest:
    """Assemble the release bundle under `release_dir` (T17.4).

    The bundle contains:
      - configs/               : frozen data contract + demo timing snapshot
      - models/                : Day 4 selected model variant
      - reports/               : evaluation, inference, guardrail, dashboard,
                                 stability, latency, acceptance reports
      - checklists/            : MVP + stretch acceptance checklists
      - evidence_index.md      : human-readable index
      - manifest.json          : machine-readable manifest with SHA-256 hashes
    """
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    bundle_map: list[tuple[Path, Path, str]] = [
        (Path("configs/data_contract.json"), release_dir / "configs" / "data_contract.json", "Day 1 data contract"),
        (Path("configs/demo_timing.json"), release_dir / "configs" / "demo_timing.json", "Day 5 demo timing envelope"),
        (Path("artifacts/models/day4/ablations"), release_dir / "models" / "ablations", "Day 4 ablation model variants"),
        (Path("artifacts/models/day3"), release_dir / "models" / "day3_baseline", "Day 3 MVP baseline models"),
        (Path("artifacts/reports/day3"), release_dir / "reports" / "day3", "Day 3 training reports"),
        (Path("artifacts/reports/day4/evaluation_summary.json"), release_dir / "reports" / "day4" / "evaluation_summary.json", "Day 4 evaluation summary"),
        (Path("artifacts/reports/day4/ablation_report.json"), release_dir / "reports" / "day4" / "ablation_report.json", "Day 4 ablation report"),
        (Path("artifacts/reports/day4/evaluation_narrative.md"), release_dir / "reports" / "day4" / "evaluation_narrative.md", "Day 4 evaluation narrative"),
        (Path("artifacts/reports/day4/model_selection.json"), release_dir / "reports" / "day4" / "model_selection.json", "Day 4 model selection"),
        (Path("artifacts/reports/day4/inference_diagnostics.json"), release_dir / "reports" / "day4" / "inference_diagnostics.json", "Day 4 inference diagnostics"),
        (Path("artifacts/reports/day4/guardrail_transition_log.json"), release_dir / "reports" / "day4" / "guardrail_transition_log.json", "Day 4 guardrail transition log"),
        (Path("artifacts/reports/day4/guardrail_summary.json"), release_dir / "reports" / "day4" / "guardrail_summary.json", "Day 4 guardrail summary"),
        (Path("artifacts/reports/day4/day4_integration_summary.json"), release_dir / "reports" / "day4" / "day4_integration_summary.json", "Day 4 integration summary"),
        (Path("artifacts/reports/day5/dashboard_snapshot.png"), release_dir / "reports" / "day5" / "dashboard_snapshot.png", "Day 5 dashboard snapshot"),
        (Path("artifacts/reports/day5/dashboard_validation.json"), release_dir / "reports" / "day5" / "dashboard_validation.json", "Day 5 dashboard validation"),
        (Path("artifacts/reports/day5/demo_summary.json"), release_dir / "reports" / "day5" / "demo_summary.json", "Day 5 demo summary"),
        (Path("artifacts/reports/day5/defect_log.json"), release_dir / "reports" / "day5" / "defect_log.json", "Day 5 dry-run defect log"),
        (Path("artifacts/reports/day5/day5_exit_gate.json"), release_dir / "reports" / "day5" / "day5_exit_gate.json", "Day 5 exit gate evidence"),
        (Path("artifacts/reports/day6/latency_benchmark.json"), release_dir / "reports" / "day6" / "latency_benchmark.json", "Day 6 latency benchmark"),
        (Path("artifacts/reports/day6/stability_report.json"), release_dir / "reports" / "day6" / "stability_report.json", "Day 6 alert-stability report"),
        (Path("artifacts/reports/day6/defect_triage.json"), release_dir / "reports" / "day6" / "defect_triage.json", "Day 6 dry-run defect triage"),
        (Path("artifacts/reports/day6/acceptance_results.json"), release_dir / "reports" / "day6" / "acceptance_results.json", "Day 6 AC-001..AC-005 results"),
        (Path("artifacts/reports/day6/day6_exit_gate.json"), release_dir / "reports" / "day6" / "day6_exit_gate.json", "Day 6 exit gate evidence"),
        (Path("artifacts/reports/day6/mvp_acceptance_ac_001_to_ac_005.md"), release_dir / "checklists" / "mvp_acceptance_ac_001_to_ac_005.md", "Updated MVP acceptance checklist"),
        (Path("docs/day1/checklists/stretch_acceptance_ac_006_to_ac_009.md"), release_dir / "checklists" / "stretch_acceptance_ac_006_to_ac_009.md", "Stretch acceptance checklist"),
        (Path("docs/day6/demo_runbook.md"), release_dir / "docs" / "demo_runbook.md", "T17.2 Demo runbook"),
        (Path("docs/day6/limitations_and_risks.md"), release_dir / "docs" / "limitations_and_risks.md", "T17.3 Limitations and risks"),
        (Path("README.md"), release_dir / "docs" / "README.md", "Updated README"),
        (Path("environment.yml"), release_dir / "env" / "environment.yml", "Conda environment spec"),
        (Path("requirements.txt"), release_dir / "env" / "requirements.txt", "Pip requirements"),
    ]

    entries: list[dict[str, Any]] = []
    for src, dst, label in bundle_map:
        copied = _copy_if_exists(src, dst)
        entry = {
            "label": label,
            "source": str(src),
            "destination": str(dst.relative_to(release_dir)),
            "included": copied,
        }
        if copied and dst.is_file():
            entry["sha256"] = _sha256(dst)
            entry["bytes"] = dst.stat().st_size
        entries.append(entry)

    # ------------------------------------------------------------------
    # Pull acceptance + gate decisions for quick access
    # ------------------------------------------------------------------
    acceptance: dict[str, Any] = {}
    acc_path = Path("artifacts/reports/day6/acceptance_results.json")
    if acc_path.exists():
        acceptance = json.loads(acc_path.read_text(encoding="utf-8"))

    gate_decisions: dict[str, Any] = {}
    for day in ("day1", "day2", "day3", "day4", "day5", "day6"):
        gate_file = Path(f"artifacts/reports/{day}/{day}_exit_gate.json")
        if gate_file.exists():
            try:
                gate_decisions[day] = json.loads(gate_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                gate_decisions[day] = {"error": "malformed"}

    evidence: dict[str, str] = {
        entry["label"]: entry["destination"]
        for entry in entries
        if entry["included"]
    }

    manifest = ReleaseManifest(
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        entries=entries,
        evidence=evidence,
        acceptance=acceptance,
        gate_decisions=gate_decisions,
    )

    # ------------------------------------------------------------------
    # Persist manifest + human-readable index
    # ------------------------------------------------------------------
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": manifest.version,
                "created_at": manifest.created_at,
                "commit": manifest.commit,
                "entries": manifest.entries,
                "evidence": manifest.evidence,
                "acceptance": manifest.acceptance,
                "gate_decisions": manifest.gate_decisions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    index_path = release_dir / "evidence_index.md"
    index_path.write_text(_render_evidence_index(manifest), encoding="utf-8")

    LOGGER.info("Release bundle assembled at %s", release_dir)
    return manifest


def _render_evidence_index(manifest: ReleaseManifest) -> str:
    """Render a human-readable evidence index for the release bundle."""
    lines: list[str] = []
    lines.append("# NeuroGauge Release Evidence Index")
    lines.append("")
    lines.append(f"- Version: {manifest.version}")
    lines.append(f"- Created: {manifest.created_at}")
    if manifest.commit:
        lines.append(f"- Git commit: `{manifest.commit}`")
    lines.append("")

    if manifest.acceptance:
        lines.append("## MVP Acceptance Summary (AC-001..AC-005)")
        lines.append("")
        passed = manifest.acceptance.get("passed_count", 0)
        total = manifest.acceptance.get("total_count", 0)
        lines.append(f"- Passed: **{passed} / {total}**")
        lines.append("")
        for c in manifest.acceptance.get("criteria", []):
            mark = "PASS" if c.get("passed") else "FAIL"
            lines.append(f"- [{mark}] **{c.get('id')}** — {c.get('description')}")
            if c.get("notes"):
                lines.append(f"    - {c.get('notes')}")
        lines.append("")

    lines.append("## Gate Decisions")
    lines.append("")
    for day in sorted(manifest.gate_decisions.keys()):
        lines.append(f"### {day.upper()}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(manifest.gate_decisions[day], indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Included Artifacts")
    lines.append("")
    lines.append("| Label | Path | Included | SHA-256 |")
    lines.append("|-------|------|----------|---------|")
    for e in manifest.entries:
        sha = e.get("sha256", "-")[:16] + ("…" if "sha256" in e else "")
        included = "YES" if e["included"] else "no"
        lines.append(f"| {e['label']} | `{e['destination']}` | {included} | `{sha}` |")

    return "\n".join(lines) + "\n"
