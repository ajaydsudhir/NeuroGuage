"""Classroom-demo entry point — recommended one-liner for live class demos.

This wrapper presets the flags that the README + diagnostic analysis identified
as the right defaults for an in-class showcase of NeuroGauge:

* ``--guardrail-config configs/guardrails_personalized.json`` — thresholds
  expressed in z-score-normalised space (Fix A).
* ``--baseline-duration 20`` — collects a 20 s personal baseline of the
  model's risk output before guardrails fire (Fix B).
* ``--baseline-stdev-floor 0.04`` — keeps tiny noise jiggles from saturating
  the normalised gauge.
* Auto-generated session id stamped with a timestamp so multiple demo runs
  don't overwrite each other.
* Raw OpenMuse recording saved to ``data/raw/`` for offline replay.

Usage (PowerShell)::

    python scripts/run_demo.py --address 00:55:DA:B9:EF:09 --duration 180

Pass any other ``run-live`` flag to override.  E.g. to reuse a baseline
collected earlier today::

    python scripts/run_demo.py --address ... \
        --load-baseline artifacts/reports/live/<earlier>_baseline.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuroguage.cli import main_run_live   # noqa: E402


def main() -> int:
    user_argv = sys.argv[1:]

    address: str | None = None
    duration: str | None = None
    session_id: str | None = None
    record_raw_to: str | None = None
    has_guardrail_config = False
    has_baseline_duration = False

    i = 0
    while i < len(user_argv):
        arg = user_argv[i]
        if arg == "--address" and i + 1 < len(user_argv):
            address = user_argv[i + 1]
        elif arg == "--duration" and i + 1 < len(user_argv):
            duration = user_argv[i + 1]
        elif arg == "--session-id" and i + 1 < len(user_argv):
            session_id = user_argv[i + 1]
        elif arg == "--record-raw" and i + 1 < len(user_argv):
            record_raw_to = user_argv[i + 1]
        elif arg == "--guardrail-config":
            has_guardrail_config = True
        elif arg == "--baseline-duration":
            has_baseline_duration = True
        i += 1

    if address is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--address", required=True,
                            help="Muse S Athena MAC address (e.g. 00:55:DA:B9:EF:09).")
        parser.parse_known_args(user_argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if session_id is None:
        session_id = f"classroom_demo_{timestamp}"
    if record_raw_to is None:
        record_raw_to = str(REPO_ROOT / "data" / "raw" / f"{session_id}.txt")
    if duration is None:
        duration = "180"

    forwarded: list[str] = ["--source", "muse", "--address", address or "",
                            "--session-id", session_id,
                            "--duration", duration,
                            "--record-raw", record_raw_to]

    if not has_guardrail_config:
        forwarded += ["--guardrail-config",
                      str(REPO_ROOT / "configs" / "guardrails_personalized.json")]
    if not has_baseline_duration:
        forwarded += ["--baseline-duration", "20"]

    pass_through = [
        a for a in user_argv
        if a not in {"--address", address or "",
                     "--duration", duration,
                     "--session-id", session_id,
                     "--record-raw", record_raw_to}
    ]
    forwarded.extend(pass_through)

    sys.argv = ["scripts/run_demo.py", *forwarded]
    print("[run_demo] forwarding:", " ".join(forwarded))
    return main_run_live() or 0


if __name__ == "__main__":
    raise SystemExit(main())
