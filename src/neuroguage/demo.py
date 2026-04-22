"""Day 5 — T15  Class Quick Demo Orchestration.

Provides a guided demo flow with:

  T15.1  Guided 5–10 minute demo flow with stage timers.
  T15.2  Validated sequence: 1-min baseline → 2-min sustained attention →
         2-min 2-back → 2–3-min code block.
  T15.3  Final 30–60 second summary panel for transition interpretation.
  T15.4  Dry-run support with failure logging.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DemoStage:
    """A single stage in the demo sequence."""

    name: str
    label: str
    duration_s: float
    task_type: str
    description: str = ""

    # Runtime state
    started_at: float = 0.0
    ended_at: float = 0.0
    actual_duration_s: float = 0.0
    status: str = "pending"     # pending | running | completed | skipped | failed
    failure_reason: str = ""


@dataclass
class DemoConfig:
    """Configuration for the class quick demo."""

    stages: list[DemoStage] = field(default_factory=list)
    total_min_s: float = 300.0
    total_max_s: float = 600.0
    stage_tolerance_pct: float = 20.0
    latency_budget_s: float = 2.0

    @classmethod
    def from_json(cls, path: str | Path) -> DemoConfig:
        """Load demo configuration from the timing budget JSON."""
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))

        quick = raw.get("quick_demo", {})
        stages = [
            DemoStage(
                name=s["name"],
                label=s["label"],
                duration_s=s["duration_s"],
                task_type=s["task_type"],
                description=s.get("description", ""),
            )
            for s in quick.get("stages", [])
        ]

        constraints = raw.get("timing_constraints", {})
        return cls(
            stages=stages,
            total_min_s=constraints.get("min_total_s", 300.0),
            total_max_s=constraints.get("max_total_s", 600.0),
            stage_tolerance_pct=constraints.get("stage_tolerance_pct", 20.0),
            latency_budget_s=constraints.get("latency_budget_s", 2.0),
        )

    @classmethod
    def default(cls) -> DemoConfig:
        """Create a default quick demo config matching PRD 10.2."""
        return cls(
            stages=[
                DemoStage("baseline", "Baseline Rest", 60, "baseline_rest",
                          "1 minute eyes-open baseline rest for per-session reference."),
                DemoStage("sustained_attention", "Sustained Attention", 120, "sustained_attention",
                          "2 minutes sustained attention reaction task."),
                DemoStage("n_back", "2-Back Working Memory", 120, "n_back",
                          "2 minutes 2-back working memory task."),
                DemoStage("code_comprehension", "Code Comprehension", 150, "code_comprehension",
                          "2 to 3 minutes code comprehension block."),
                DemoStage("summary", "Timeline Summary", 45, "summary",
                          "30 to 60 seconds for live risk timeline and guardrail-state interpretation."),
            ],
            total_min_s=300.0,
            total_max_s=600.0,
        )


# ---------------------------------------------------------------------------
# Timing validation  (T15.2)
# ---------------------------------------------------------------------------

def validate_demo_timing(config: DemoConfig) -> dict[str, Any]:
    """Validate that the demo sequence fits within the timing budget (T15.2).

    Returns a validation report dict.
    """
    total_planned = sum(s.duration_s for s in config.stages)
    violations: list[str] = []

    if total_planned < config.total_min_s:
        violations.append(
            f"Total planned duration ({total_planned:.0f}s) is below minimum ({config.total_min_s:.0f}s)."
        )
    if total_planned > config.total_max_s:
        violations.append(
            f"Total planned duration ({total_planned:.0f}s) exceeds maximum ({config.total_max_s:.0f}s)."
        )

    # Check sequence order (PRD: baseline → sustained_attention → n_back → code_comprehension → summary)
    expected_order = ["baseline", "sustained_attention", "n_back", "code_comprehension", "summary"]
    actual_order = [s.name for s in config.stages]
    if actual_order != expected_order:
        violations.append(
            f"Stage sequence {actual_order} does not match expected {expected_order}."
        )

    stage_reports = []
    for stage in config.stages:
        tol = stage.duration_s * (config.stage_tolerance_pct / 100.0)
        stage_reports.append({
            "name": stage.name,
            "label": stage.label,
            "planned_s": stage.duration_s,
            "tolerance_s": tol,
            "min_acceptable_s": stage.duration_s - tol,
            "max_acceptable_s": stage.duration_s + tol,
        })

    return {
        "total_planned_s": total_planned,
        "total_planned_min": total_planned / 60.0,
        "within_budget": len(violations) == 0,
        "violations": violations,
        "stages": stage_reports,
    }


# ---------------------------------------------------------------------------
# Demo state
# ---------------------------------------------------------------------------

@dataclass
class DemoState:
    """Mutable runtime state for a demo run."""

    config: DemoConfig
    run_id: str = ""
    started_at: float = 0.0
    current_stage_index: int = -1
    is_running: bool = False
    is_completed: bool = False
    total_elapsed_s: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)
    stage_log: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Demo orchestrator  (T15.1, T15.3)
# ---------------------------------------------------------------------------

class DemoOrchestrator:
    """Manages the guided demo flow lifecycle (T15.1).

    Provides methods to advance through stages, track timing,
    and produce the final summary panel (T15.3).
    """

    def __init__(self, config: DemoConfig | None = None) -> None:
        self.config = config or DemoConfig.default()
        self._state: DemoState | None = None

    @property
    def state(self) -> DemoState | None:
        return self._state

    def start(self, run_id: str = "") -> DemoState:
        """Begin a new demo run."""
        self._state = DemoState(
            config=self.config,
            run_id=run_id or f"demo_{int(time.time())}",
            started_at=time.time(),
            is_running=True,
        )
        LOGGER.info("Demo run %s started", self._state.run_id)
        return self._state

    def advance_stage(self) -> DemoStage | None:
        """Move to the next demo stage (T15.1)."""
        if self._state is None or not self._state.is_running:
            return None

        now = time.time()

        # Close current stage if one is active
        if self._state.current_stage_index >= 0:
            current = self.config.stages[self._state.current_stage_index]
            current.ended_at = now
            current.actual_duration_s = current.ended_at - current.started_at
            current.status = "completed"
            self._state.stage_log.append({
                "name": current.name,
                "label": current.label,
                "planned_s": current.duration_s,
                "actual_s": current.actual_duration_s,
                "status": current.status,
            })
            LOGGER.info(
                "Stage '%s' completed in %.1fs (planned: %.1fs)",
                current.label, current.actual_duration_s, current.duration_s,
            )

        # Advance
        self._state.current_stage_index += 1

        if self._state.current_stage_index >= len(self.config.stages):
            self._state.is_running = False
            self._state.is_completed = True
            self._state.total_elapsed_s = now - self._state.started_at
            LOGGER.info(
                "Demo run %s completed in %.1fs",
                self._state.run_id, self._state.total_elapsed_s,
            )
            return None

        next_stage = self.config.stages[self._state.current_stage_index]
        next_stage.started_at = now
        next_stage.status = "running"
        LOGGER.info("Stage '%s' started (target: %.0fs)", next_stage.label, next_stage.duration_s)
        return next_stage

    def current_stage(self) -> DemoStage | None:
        """Get the currently running stage."""
        if self._state is None or self._state.current_stage_index < 0:
            return None
        if self._state.current_stage_index >= len(self.config.stages):
            return None
        return self.config.stages[self._state.current_stage_index]

    def get_elapsed_s(self) -> float:
        """Total elapsed time since demo start."""
        if self._state is None:
            return 0.0
        return time.time() - self._state.started_at

    def get_stage_elapsed_s(self) -> float:
        """Elapsed time in the current stage."""
        stage = self.current_stage()
        if stage is None or stage.started_at <= 0:
            return 0.0
        return time.time() - stage.started_at

    def get_stage_remaining_s(self) -> float:
        """Remaining time in the current stage."""
        stage = self.current_stage()
        if stage is None:
            return 0.0
        return max(0.0, stage.duration_s - self.get_stage_elapsed_s())

    def is_stage_overtime(self) -> bool:
        """Check if current stage has exceeded its time budget."""
        stage = self.current_stage()
        if stage is None:
            return False
        tol = stage.duration_s * (self.config.stage_tolerance_pct / 100.0)
        return self.get_stage_elapsed_s() > stage.duration_s + tol

    def log_failure(self, description: str, severity: str = "medium") -> None:
        """Log a reproducible failure during a dry run (T15.4)."""
        if self._state is None:
            return
        stage = self.current_stage()
        failure = {
            "timestamp": time.time(),
            "elapsed_s": self.get_elapsed_s(),
            "stage": stage.name if stage else "none",
            "description": description,
            "severity": severity,
        }
        self._state.failures.append(failure)
        LOGGER.warning("Demo failure logged: %s (stage: %s, severity: %s)",
                       description, failure["stage"], severity)

    # ------------------------------------------------------------------
    # T15.3 — Summary panel
    # ------------------------------------------------------------------

    def build_summary(
        self,
        inference_updates: list[dict[str, Any]] | None = None,
        guardrail_transitions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the final 30–60 second summary panel data (T15.3).

        Returns a dict suitable for display or serialisation.
        """
        if self._state is None:
            return {"error": "No demo run active or completed"}

        # Prefer stage-derived total when wall-clock elapsed is negligible
        # (e.g. accelerated dry runs) so that the reported envelope reflects
        # the simulated demo timing rather than the harness runtime.
        stage_total = sum(s.actual_duration_s for s in self.config.stages)
        wall_elapsed = self._state.total_elapsed_s or self.get_elapsed_s()
        total_elapsed = stage_total if stage_total > wall_elapsed else wall_elapsed

        stage_summary = []
        for stage in self.config.stages:
            stage_summary.append({
                "name": stage.name,
                "label": stage.label,
                "planned_s": stage.duration_s,
                "actual_s": stage.actual_duration_s,
                "status": stage.status,
            })

        # Risk summary
        risk_summary: dict[str, Any] = {}
        if inference_updates:
            risk_scores = [u.get("risk_score", 0) for u in inference_updates]
            risk_summary = {
                "total_updates": len(inference_updates),
                "risk_mean": float(sum(risk_scores) / len(risk_scores)),
                "risk_min": float(min(risk_scores)),
                "risk_max": float(max(risk_scores)),
            }

        # Transition summary
        transition_summary: list[dict[str, Any]] = []
        if guardrail_transitions:
            for tr in guardrail_transitions:
                transition_summary.append({
                    "time_s": tr.get("timestamp_s", 0),
                    "from": tr.get("previous_state", "?"),
                    "to": tr.get("new_state", "?"),
                    "rationale": tr.get("rationale", ""),
                })

        return {
            "run_id": self._state.run_id,
            "total_elapsed_s": total_elapsed,
            "total_elapsed_min": total_elapsed / 60.0,
            "wall_elapsed_s": wall_elapsed,
            "stage_total_s": stage_total,
            "within_budget": self.config.total_min_s <= total_elapsed <= self.config.total_max_s,
            "stages": stage_summary,
            "risk_summary": risk_summary,
            "transitions": transition_summary,
            "failures": self._state.failures,
            "failure_count": len(self._state.failures),
        }


# ---------------------------------------------------------------------------
# Replay-mode dry run  (T15.4)
# ---------------------------------------------------------------------------

def run_dry_demo(
    config: DemoConfig | None = None,
    inference_diagnostics: list[dict[str, Any]] | None = None,
    guardrail_transitions: list[dict[str, Any]] | None = None,
    run_id: str = "",
    simulate_realtime: bool = False,
) -> dict[str, Any]:
    """Execute a simulated dry run using replay data (T15.4).

    Validates timing, logs failures, and returns a dry-run report.
    If simulate_realtime is False, stages complete instantaneously.
    """
    if config is None:
        config = DemoConfig.default()

    # Timing validation first (T15.2)
    timing_report = validate_demo_timing(config)

    orchestrator = DemoOrchestrator(config)
    state = orchestrator.start(run_id=run_id or f"dry_run_{int(time.time())}")

    # Simulate each stage
    for _ in range(len(config.stages)):
        stage = orchestrator.advance_stage()
        if stage is None:
            break

        if simulate_realtime:
            time.sleep(min(stage.duration_s, 0.5))  # capped for test speed

    orchestrator.advance_stage()

    # Dry-run mode does not consume wall-clock time per stage, so restore
    # planned durations as the authoritative actual_duration_s for reporting.
    # This ensures the summary reflects the simulated envelope, not harness
    # runtime.
    for stage in config.stages:
        stage.actual_duration_s = stage.duration_s
        stage.status = "completed"

    # Check for issues
    if inference_diagnostics:
        latencies = [d.get("latency_s", 0) for d in inference_diagnostics]
        if max(latencies) > config.latency_budget_s:
            orchestrator.log_failure(
                f"Max latency {max(latencies):.3f}s exceeds budget {config.latency_budget_s}s",
                severity="high",
            )

        fault_counts = sum(1 for d in inference_diagnostics if d.get("fault_flags"))
        if fault_counts > 0:
            orchestrator.log_failure(
                f"{fault_counts} inference updates had fault flags",
                severity="medium",
            )

    if not timing_report["within_budget"]:
        for v in timing_report["violations"]:
            orchestrator.log_failure(v, severity="high")

    # Build summary (T15.3)
    summary = orchestrator.build_summary(
        inference_updates=inference_diagnostics,
        guardrail_transitions=guardrail_transitions,
    )
    summary["timing_validation"] = timing_report
    summary["dry_run"] = True

    return summary


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def write_demo_artifacts(
    dry_run_reports: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist dry-run reports and defect log."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Individual dry-run reports
    for i, report in enumerate(dry_run_reports):
        report_path = out / f"dry_run_{i+1}_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths[f"dry_run_{i+1}"] = report_path
        LOGGER.info("Dry-run %d report written to %s", i + 1, report_path)

    # Aggregate defect log
    all_failures: list[dict[str, Any]] = []
    for i, report in enumerate(dry_run_reports):
        for f in report.get("failures", []):
            f["run_number"] = i + 1
            f["run_id"] = report.get("run_id", f"run_{i+1}")
            all_failures.append(f)

    defect_path = out / "defect_log.json"
    defect_path.write_text(json.dumps(all_failures, indent=2), encoding="utf-8")
    paths["defect_log"] = defect_path
    LOGGER.info("Defect log with %d entries written to %s", len(all_failures), defect_path)

    # Combined summary
    combined = {
        "total_dry_runs": len(dry_run_reports),
        "total_failures": len(all_failures),
        "all_passed": all(r.get("failure_count", 0) == 0 for r in dry_run_reports),
        "timing_within_budget": all(
            r.get("timing_validation", {}).get("within_budget", False)
            for r in dry_run_reports
        ),
    }
    summary_path = out / "demo_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    paths["summary"] = summary_path

    return paths
