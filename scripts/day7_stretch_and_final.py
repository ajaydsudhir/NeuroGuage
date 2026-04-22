"""Day 7 - T18/T19/T20 orchestrator.

Covers:
  T18   Train BiGRU-attention prototype on identical held-out split, measure
        latency + stability, persist artifacts.
  T19   Compare advanced vs MVP on identical held-out sessions, verify +3pt
        promotion threshold, verify no latency/stability regression, record
        decision note.
  T20   Final packaging + demo rehearsal + RC freeze + status summary.
  G07   Day 7 Exit Gate evidence file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from neuroguage.advanced import (  # noqa: E402
    AdvancedModelConfig,
    infer_single_window,
    load_advanced_model,
    train_advanced_model,
    write_advanced_artifacts,
)
from neuroguage.finalization import (  # noqa: E402
    build_final_demo_bundle,
    build_final_status_summary,
    resolve_stretch_status,
    run_final_rehearsal,
    write_final_status_summary,
    write_rehearsal_report,
)
from neuroguage.guardrails import GuardrailConfig, GuardrailState, update_guardrail  # noqa: E402
from neuroguage.hardening import (  # noqa: E402
    benchmark_refresh_latency,
    run_all_stability_scenarios,
    write_latency_benchmark,
    write_stability_report,
)
from neuroguage.promotion import decide_promotion, write_promotion_decision  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELED_FEATURE_PATH = REPO_ROOT / "artifacts/features/day3/labeled_feature_matrix.csv"

MVP_EVAL_SUMMARY = REPO_ROOT / "artifacts/reports/day4/evaluation_summary.json"
MVP_LATENCY_PATH = REPO_ROOT / "artifacts/reports/day6/latency_benchmark.json"
MVP_STABILITY_PATH = REPO_ROOT / "artifacts/reports/day6/stability_report.json"
MVP_ACCEPTANCE_PATH = REPO_ROOT / "artifacts/reports/day6/acceptance_results.json"
DAY6_EXIT_GATE_PATH = REPO_ROOT / "artifacts/reports/day6/day6_exit_gate.json"
MVP_FEATURE_COLUMNS = REPO_ROOT / "artifacts/models/day4/ablations/eeg_optical/feature_columns.json"

INFERENCE_DIAG_PATH = REPO_ROOT / "artifacts/reports/day4/inference_diagnostics.json"
GUARDRAIL_LOG_PATH = REPO_ROOT / "artifacts/reports/day4/guardrail_transition_log.json"
DEMO_CONFIG_PATH = REPO_ROOT / "configs/demo_timing.json"

DAY7_REPORT_DIR = REPO_ROOT / "artifacts/reports/day7"
DAY7_MODEL_DIR = REPO_ROOT / "artifacts/models/day7/advanced"

ADV_METRICS_PATH = DAY7_REPORT_DIR / "advanced_training_metrics.json"
ADV_LOG_PATH = DAY7_REPORT_DIR / "advanced_training_log.json"
ADV_PREDICTIONS_PATH = DAY7_REPORT_DIR / "advanced_predictions.csv"
ADV_LATENCY_PATH = DAY7_REPORT_DIR / "advanced_latency_benchmark.json"
ADV_STABILITY_PATH = DAY7_REPORT_DIR / "advanced_stability_report.json"
PROMOTION_JSON = DAY7_REPORT_DIR / "promotion_decision.json"
PROMOTION_MD = DAY7_REPORT_DIR / "promotion_decision.md"
REHEARSAL_PATH = DAY7_REPORT_DIR / "rehearsal_report.json"
FINAL_STATUS_JSON = DAY7_REPORT_DIR / "final_status_summary.json"
FINAL_STATUS_MD = DAY7_REPORT_DIR / "final_status_summary.md"
GATE_PATH = DAY7_REPORT_DIR / "day7_exit_gate.json"

RELEASE_DIR = REPO_ROOT / "artifacts/release"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:  # noqa: BLE001
        return None


def _load_json(path: Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stability_dicts_from_report(report: dict | list) -> list[dict]:
    if isinstance(report, list):
        return report
    if isinstance(report, dict):
        return report.get("scenarios", []) or []
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("[Day 7] Graduate Model, Comparison, Packaging, and Final Rehearsal")
    print("=" * 72)
    DAY7_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DAY7_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    day6_gate = _load_json(DAY6_EXIT_GATE_PATH) if DAY6_EXIT_GATE_PATH.exists() else {}
    day6_decision = str(day6_gate.get("phase_gate_decision", "")).upper()
    blocked = any(tok in day6_decision for tok in ("NO-GO", "NO GO", "BLOCK", "DEFER"))
    allowed = (not blocked) and ("GO" in day6_decision)

    advanced_metrics: dict = {}
    advanced_latency_report: dict = {}
    advanced_stability_records: list[dict] = []
    promotion_decision_dict: dict | None = None

    # -----------------------------------------------------------------------
    # T18 - Graduate model prototype (if stretch is allowed)
    # -----------------------------------------------------------------------
    if allowed and LABELED_FEATURE_PATH.exists() and MVP_FEATURE_COLUMNS.exists():
        print("\n[T18.1] Training BiGRU-attention on identical held-out split...")
        labeled_frame = pd.read_csv(LABELED_FEATURE_PATH)
        feature_columns = _load_json(MVP_FEATURE_COLUMNS)

        cfg = AdvancedModelConfig()
        result = train_advanced_model(
            labeled_frame=labeled_frame,
            feature_columns=feature_columns,
            config=cfg,
        )
        artifact_paths = write_advanced_artifacts(
            result=result,
            model_dir=DAY7_MODEL_DIR,
            metrics_path=ADV_METRICS_PATH,
            predictions_path=ADV_PREDICTIONS_PATH,
            training_log_path=ADV_LOG_PATH,
        )
        advanced_metrics = result.metrics
        print(f"  Epochs trained: {advanced_metrics['epochs_trained']}")
        print(f"  Val metrics:    {advanced_metrics['validation']}")
        print(f"  Test metrics:   {advanced_metrics['test']}")
        print(f"  Artifacts:      {artifact_paths['weights_path']}")

        # ------------------------------------------------------------------
        # T18.3 - Latency + stability parity with MVP
        # ------------------------------------------------------------------
        print("\n[T18.3] Benchmarking advanced-model refresh latency...")
        model, _config_payload = load_advanced_model(
            weights_path=Path(artifact_paths["weights_path"]),
            config_path=Path(artifact_paths["config_path"]),
        )

        feature_matrix = labeled_frame[feature_columns].to_numpy(dtype=float)
        n = len(feature_matrix)

        guard_cfg = GuardrailConfig()
        guard_state = GuardrailState()

        def _infer(i: int) -> tuple[float, float]:
            vec = feature_matrix[i % n]
            score = infer_single_window(model, vec)
            confidence = float(labeled_frame.iloc[i % n].get("confidence_score", 0.8))
            return score, confidence

        def _guard(r: float, c: float, ts: float) -> None:
            update_guardrail(r, c, ts, guard_state, guard_cfg)

        bench = benchmark_refresh_latency(
            inference_update_fn=_infer,
            guardrail_update_fn=_guard,
            sample_count=max(50, n * 2),
            budget_s=2.0,
        )
        write_latency_benchmark(bench, ADV_LATENCY_PATH)
        advanced_latency_report = _load_json(ADV_LATENCY_PATH)  # type: ignore[assignment]
        print(f"  Samples: {bench.samples}  mean={bench.mean_s*1000:.3f} ms  "
              f"p95={bench.p95_s*1000:.3f} ms  max={bench.max_s*1000:.3f} ms  "
              f"within_budget={bench.within_budget}")

        print("\n[T18.3] Running stability scenarios with guardrail parity...")
        stability_reports = run_all_stability_scenarios()
        write_stability_report(stability_reports, ADV_STABILITY_PATH)
        advanced_stability_records = [
            {
                "scenario": r.scenario,
                "passed": r.passed,
                "transitions_per_minute": r.transitions_per_minute,
                "max_consecutive_flips": r.max_consecutive_flips,
            }
            for r in stability_reports
        ]
        for r in stability_reports:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.scenario:>24}: per_min={r.transitions_per_minute:5.2f} "
                  f"flips={r.max_consecutive_flips}")

        # ------------------------------------------------------------------
        # T19 - Advanced vs MVP comparison and promotion decision
        # ------------------------------------------------------------------
        print("\n[T19] Comparing advanced vs MVP on identical held-out sessions...")
        mvp_eval = _load_json(MVP_EVAL_SUMMARY)
        mvp_latency = _load_json(MVP_LATENCY_PATH)
        mvp_stability = _stability_dicts_from_report(_load_json(MVP_STABILITY_PATH))

        decision = decide_promotion(
            mvp_eval_summary=mvp_eval,  # type: ignore[arg-type]
            advanced_metrics=advanced_metrics,
            mvp_latency=mvp_latency,  # type: ignore[arg-type]
            advanced_latency=advanced_latency_report,
            mvp_stability=mvp_stability,
            advanced_stability=advanced_stability_records,
            held_out_sessions=result.test_sessions,
        )
        write_promotion_decision(decision, PROMOTION_JSON, PROMOTION_MD)
        promotion_decision_dict = _load_json(PROMOTION_JSON)  # type: ignore[assignment]

        print(f"  [T19.2] metric gate passed: {decision.metric_gate_passed}")
        print(f"  [T19.3] latency gate passed: {decision.latency_gate_passed}")
        print(f"  [T19.3] stability gate passed: {decision.stability_gate_passed}")
        print(f"  [T19.4] decision: {decision.decision}")
    else:
        print("\n[T18/T19] Stretch deferred (Day 6 gate blocked stretch or "
              "required inputs missing).")

    # -----------------------------------------------------------------------
    # T20.1 - Stretch deferral gate resolution
    # -----------------------------------------------------------------------
    stretch = resolve_stretch_status(day6_gate if isinstance(day6_gate, dict) else {}, promotion_decision_dict)
    print(f"\n[T20.1] Stretch status: attempted={stretch.stretch_attempted} "
          f"deferred={stretch.deferred}")
    print(f"         {stretch.reason}")

    # -----------------------------------------------------------------------
    # T20.3 - Final rehearsal under class-demo timing constraints
    # -----------------------------------------------------------------------
    print("\n[T20.3] Running final rehearsal under class-demo timing constraints...")
    mvp_latency_payload = _load_json(MVP_LATENCY_PATH) if MVP_LATENCY_PATH.exists() else {}
    mvp_stability_records = (
        _stability_dicts_from_report(_load_json(MVP_STABILITY_PATH))
        if MVP_STABILITY_PATH.exists()
        else []
    )
    rehearsal = run_final_rehearsal(
        demo_config_path=DEMO_CONFIG_PATH,
        inference_diagnostics_path=INFERENCE_DIAG_PATH,
        guardrail_log_path=GUARDRAIL_LOG_PATH,
        latency_benchmark=mvp_latency_payload,  # type: ignore[arg-type]
        stability_reports=mvp_stability_records,
        run_id="day7_final_rehearsal",
    )
    write_rehearsal_report(rehearsal, REHEARSAL_PATH)
    print(f"  within_budget={rehearsal.within_budget} "
          f"elapsed={rehearsal.total_elapsed_s:.1f}s "
          f"failures={rehearsal.failure_count}")

    # -----------------------------------------------------------------------
    # T20.4 - Final status summary + release-candidate freeze
    # -----------------------------------------------------------------------
    print("\n[T20.4] Composing final status summary + RC bundle...")
    mvp_eval_payload = _load_json(MVP_EVAL_SUMMARY) if MVP_EVAL_SUMMARY.exists() else {}
    mvp_acceptance = _load_json(MVP_ACCEPTANCE_PATH) if MVP_ACCEPTANCE_PATH.exists() else {}

    commit = _git_commit()
    rc_version = "1.0.0-rc1"

    summary = build_final_status_summary(
        rc_version=rc_version,
        commit=commit,
        stretch=stretch,
        rehearsal=rehearsal,
        acceptance_results=mvp_acceptance if isinstance(mvp_acceptance, dict) else {},
        promotion_decision=promotion_decision_dict,
        latency_benchmark=mvp_latency_payload if isinstance(mvp_latency_payload, dict) else {},
        stability_reports=mvp_stability_records,
        mvp_evaluation=mvp_eval_payload if isinstance(mvp_eval_payload, dict) else {},
        release_manifest_path=RELEASE_DIR / "manifest.json",
        release_dir=RELEASE_DIR,
    )
    write_final_status_summary(summary, FINAL_STATUS_JSON, FINAL_STATUS_MD)

    manifest = build_final_demo_bundle(
        release_dir=RELEASE_DIR,
        version=rc_version,
        commit=commit,
        stretch=stretch,
        day7_report_dir=DAY7_REPORT_DIR,
        advanced_model_dir=DAY7_MODEL_DIR,
    )
    included = sum(1 for e in manifest.entries if e["included"])
    print(f"  RC bundle: {RELEASE_DIR}")
    print(f"  Entries total/included: {len(manifest.entries)}/{included}")

    # -----------------------------------------------------------------------
    # G07 - Day 7 exit gate evaluation
    # -----------------------------------------------------------------------
    ac_section = summary["acceptance_ac_001_to_ac_005"]
    g07_1 = (
        ac_section["all_passed"]
        and (RELEASE_DIR / "manifest.json").exists()
        and (RELEASE_DIR / "evidence_index.md").exists()
    )

    if stretch.deferred:
        g07_2 = bool(stretch.reason)  # explicit deferral with reason
    else:
        # Either promoted or kept MVP with documented evidence.
        g07_2 = (
            promotion_decision_dict is not None
            and PROMOTION_JSON.exists()
            and PROMOTION_MD.exists()
        )

    rehearsal_section = summary["rehearsal"]
    g07_3 = (
        rehearsal_section["within_budget"]
        and rehearsal_section["failure_count"] == 0
        and rehearsal_section["latency_within_budget"]
        and rehearsal_section["stability_within_budget"]
    )

    gate_passed = bool(g07_1 and g07_2 and g07_3)
    gate_evidence = {
        "g07_1_mvp_release_finalized": bool(g07_1),
        "g07_2_stretch_promoted_or_deferred": bool(g07_2),
        "g07_3_rehearsal_timing_and_stability": bool(g07_3),
        "gate_passed": gate_passed,
        "rc_version": rc_version,
        "commit": commit,
        "release": {
            "dir": str(RELEASE_DIR),
            "entries_total": len(manifest.entries),
            "entries_included": included,
            "manifest": str(RELEASE_DIR / "manifest.json"),
            "evidence_index": str(RELEASE_DIR / "evidence_index.md"),
        },
        "stretch": {
            "attempted": stretch.stretch_attempted,
            "deferred": stretch.deferred,
            "decision": stretch.decision,
            "reason": stretch.reason,
        },
        "rehearsal": rehearsal_section,
        "acceptance_ac_001_to_ac_005": {
            "passed_count": ac_section["passed_count"],
            "total_count": ac_section["total_count"],
            "all_passed": ac_section["all_passed"],
        },
        "final_status_summary": str(FINAL_STATUS_JSON),
    }
    GATE_PATH.write_text(json.dumps(gate_evidence, indent=2), encoding="utf-8")

    print("\n[G07] Day 7 Exit Gate")
    print(f"  [G07.1] MVP release package finalized:             {'PASS' if g07_1 else 'FAIL'}")
    print(f"  [G07.2] Stretch promoted with evidence or deferred:{'PASS' if g07_2 else 'FAIL'}")
    print(f"  [G07.3] Rehearsal passes timing + stability:       {'PASS' if g07_3 else 'FAIL'}")
    print(f"  Exit gate evidence: {GATE_PATH}")
    print(f"\n[Day 7] {'COMPLETE - G07 PASSED' if gate_passed else 'COMPLETE - G07 FAILED'}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
