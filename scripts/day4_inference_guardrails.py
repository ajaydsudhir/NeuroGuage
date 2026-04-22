"""Day 4 — T12/T13 Live Inference and Guardrail Integration runner.

Runs replay-mode inference on the labeled feature matrix, feeds results
through the guardrail engine, and persists all diagnostic artifacts.

Subtasks covered:
  T12.1–T12.4  Real-time inference service (replay mode validation).
  T13.1–T13.4  Symbolic guardrail engine integration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from neuroguage.inference import (  # noqa: E402
    InferenceConfig,
    load_inference_models,
    run_replay_inference,
    write_inference_diagnostics,
)
from neuroguage.guardrails import (  # noqa: E402
    GuardrailConfig,
    run_guardrail_sequence,
    write_guardrail_artifacts,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELED_FEATURE_PATH = Path("artifacts/features/day3/labeled_feature_matrix.csv")

INFERENCE_DIAG_PATH = Path("artifacts/reports/day4/inference_diagnostics.json")
GUARDRAIL_LOG_PATH = Path("artifacts/reports/day4/guardrail_transition_log.json")
GUARDRAIL_SUMMARY_PATH = Path("artifacts/reports/day4/guardrail_summary.json")
COMBINED_SUMMARY_PATH = Path("artifacts/reports/day4/day4_integration_summary.json")


def main() -> int:
    print("[T12/T13] Day 4 Live Inference & Guardrail Integration")
    print("=" * 60)

    # --- Load data ---
    if not LABELED_FEATURE_PATH.exists():
        print(f"ERROR: labeled feature matrix not found at {LABELED_FEATURE_PATH}")
        return 1

    labeled_frame = pd.read_csv(LABELED_FEATURE_PATH)
    print(f"  Loaded labeled feature matrix: {len(labeled_frame)} rows")

    # --- Load models ---
    inference_config = InferenceConfig()
    print("\n[T12.2] Loading selected MVP model...")

    try:
        models = load_inference_models(inference_config)
        print(f"  Selected variant: {models['selected_variant']}")
        print(f"  Feature columns: {len(models['feature_columns'])}")
    except Exception as exc:
        print(f"  WARNING: Could not load Day 4 models ({exc}); using Day 3 fallback")
        inference_config.model_dir = "artifacts/models/day3"
        inference_config.selection_path = "__nonexistent__"
        models = load_inference_models(inference_config)
        print(f"  Loaded fallback models with {len(models['feature_columns'])} features")

    # --- Run replay inference (T12.1–T12.4) ---
    print("\n[T12.1] Running replay inference over labeled feature matrix...")
    updates = run_replay_inference(
        labeled_frame=labeled_frame,
        models=models,
        config=inference_config,
    )

    risk_scores = [u.risk_score for u in updates]
    latencies = [u.latency_s for u in updates]
    confidences = [u.model_confidence for u in updates]

    print(f"  Inference updates: {len(updates)}")
    print(f"  Risk score range: [{min(risk_scores):.4f}, {max(risk_scores):.4f}]")
    print(f"  Mean latency: {sum(latencies)/len(latencies):.6f}s")
    print(f"  Max latency:  {max(latencies):.6f}s")
    print(f"  All within 2s budget: {all(l <= 2.0 for l in latencies)}")

    # --- Write inference diagnostics (T12.4) ---
    print(f"\n[T12.4] Writing inference diagnostics to {INFERENCE_DIAG_PATH}")
    write_inference_diagnostics(updates, INFERENCE_DIAG_PATH)

    # --- Run guardrail engine (T13.1–T13.4) ---
    print("\n[T13.1] Running guardrail engine over inference results...")
    guardrail_config = GuardrailConfig()
    timestamps = [u.timestamp_s for u in updates]

    guardrail_state = run_guardrail_sequence(
        risk_scores=risk_scores,
        model_confidences=confidences,
        timestamps_s=timestamps,
        config=guardrail_config,
    )

    print(f"  Total updates processed:   {guardrail_state.update_count}")
    print(f"  Total transitions:         {len(guardrail_state.transitions)}")
    print(f"  Final state:               {guardrail_state.current_state.value}")

    if guardrail_state.transitions:
        print("\n  Transition Log:")
        for t in guardrail_state.transitions:
            print(
                f"    t={t.timestamp_s:>8.2f}s | {t.previous_state:>6} -> {t.new_state:<6} | "
                f"risk={t.risk_score:.4f} | {t.rationale[:80]}"
            )

    # --- Write guardrail artifacts (T13.4) ---
    print(f"\n[T13.4] Writing guardrail artifacts...")
    write_guardrail_artifacts(
        state=guardrail_state,
        transition_log_path=GUARDRAIL_LOG_PATH,
        summary_path=GUARDRAIL_SUMMARY_PATH,
    )

    # --- Combined summary ---
    combined = {
        "inference": {
            "update_count": len(updates),
            "risk_score_min": min(risk_scores),
            "risk_score_max": max(risk_scores),
            "risk_score_mean": sum(risk_scores) / len(risk_scores),
            "mean_latency_s": sum(latencies) / len(latencies),
            "max_latency_s": max(latencies),
            "all_within_budget": all(l <= 2.0 for l in latencies),
        },
        "guardrails": {
            "total_transitions": len(guardrail_state.transitions),
            "final_state": guardrail_state.current_state.value,
            "transitions": [
                {
                    "timestamp_s": t.timestamp_s,
                    "from": t.previous_state,
                    "to": t.new_state,
                    "risk_score": t.risk_score,
                    "rationale": t.rationale,
                }
                for t in guardrail_state.transitions
            ],
        },
    }
    COMBINED_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMBINED_SUMMARY_PATH.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"  Combined summary: {COMBINED_SUMMARY_PATH}")

    print("\n  All Day 4 artifacts:")
    for path in [INFERENCE_DIAG_PATH, GUARDRAIL_LOG_PATH, GUARDRAIL_SUMMARY_PATH, COMBINED_SUMMARY_PATH]:
        status = "OK" if path.exists() else "MISSING"
        print(f"    [{status}] {path}")

    print("\n[T12/T13] Day 4 Live Inference & Guardrail Integration COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
