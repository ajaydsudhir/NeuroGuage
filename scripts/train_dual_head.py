"""Train the dual-head neuro-symbolic fatigue model.

Usage::

    python scripts/train_dual_head.py \
        --labeled-frame artifacts/features/labeled/labeled_feature_matrix.csv \
        --output-dir   artifacts/models/dual_head \
        --epochs 120

Reads the same labeled feature matrix used by the MVP and advanced model
were trained on, derives the attention / metabolic-ready / mismatch weak
labels, and trains the dual-head PyTorch model with AdamW + CosineAnnealingLR
+ a custom mismatch-consistency loss.

Designed to be reproducible and CI-friendly:

* Deterministic seeding on every run.
* No extra optional dependencies beyond what the project already pins.
* Writes a small JSON summary the user can attach to a write-up.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from neuroguage.dual_head import (  # noqa: E402
    DualHeadLabelConfig,
    derive_dual_head_labels,
    summarise_dual_head_labels,
)
from neuroguage.dual_head import (  # noqa: E402
    DualHeadConfig,
    train_dual_head_model,
    write_dual_head_artifacts,
)
from neuroguage.cli import configure_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--labeled-frame",
        default=str(REPO_ROOT / "artifacts" / "features" / "labeled" / "labeled_feature_matrix.csv"),
        help="Path to the labeled feature matrix CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "models" / "dual_head"),
        help="Where to write the trained model + metrics + training log.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument(
        "--model-variant",
        choices=("muse_demo", "research_eegnet"),
        default="muse_demo",
        help="Dual-head architecture variant: class-demo Muse model or research EEGNet-style model.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mismatch-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarising attention_score into attention_label.",
    )
    parser.add_argument(
        "--metabolic-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarising metabolic_score into metabolic_ready_label.",
    )
    parser.add_argument(
        "--write-augmented-csv",
        action="store_true",
        help="If set, write the labeled frame augmented with the new label "
             "columns next to the model artifacts (useful for inspection).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging(run_name="train_dual_head")
    log = logging.getLogger(__name__)

    csv_path = Path(args.labeled_frame)
    if not csv_path.exists():
        log.error("Labeled frame not found: %s", csv_path)
        return 2

    log.info("Loading labeled feature matrix from %s", csv_path)
    frame = pd.read_csv(csv_path)
    log.info("Frame shape: %s", frame.shape)

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
    log.info("Training dual-head model with config: %s", dual_config)

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
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Training summary written to %s", summary_path)

    if args.write_augmented_csv:
        augmented_path = output_dir / "labeled_feature_matrix_with_dual_head_labels.csv"
        labelled.to_csv(augmented_path, index=False)
        log.info("Augmented label CSV written to %s", augmented_path)

    log.info(
        "Done. Validation composite=%.4f | Test attention F1=%.3f / metabolic F1=%.3f / mismatch F1=%.3f",
        result.metrics.get("best_validation_composite_score", 0.0),
        result.metrics["test"]["attention"]["macro_f1"],
        result.metrics["test"]["metabolic"]["macro_f1"],
        result.metrics["test"]["mismatch"]["macro_f1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
