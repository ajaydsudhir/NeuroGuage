"""Record a calibration / training session from a Muse S Athena device.

This script drives the three steps required to generate a
NeuroGauge-compatible labelled session:

  1. Discover a Muse S Athena (or use the MAC address you pass in).
  2. Record raw BLE packets via the ``OpenMuse record`` command.
  3. Decode that recording into ``eeg.csv``, ``optical.csv``, ``imu.csv``,
     and ``metadata.json`` under the output directory.

Example usage::

    python scripts/record_calibration_session.py \
        --session-id sub01_baseline_rest \
        --task-type baseline_rest \
        --duration 180 \
        --subjective-fatigue-score 2 \
        --output-dir data/raw/sub01/baseline_rest

The produced folder can be fed straight into Day 2 ingestion:

    python -m neuroguage.cli validate-schema \
        --eeg   data/raw/sub01/baseline_rest/eeg.csv \
        --optical data/raw/sub01/baseline_rest/optical.csv \
        --imu   data/raw/sub01/baseline_rest/imu.csv \
        --metadata data/raw/sub01/baseline_rest/metadata.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.muse_stream import decode_raw_session_to_contract, discover_muse_devices


def _run_openmuse_record(address: str, duration_s: float, preset: str, outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "OpenMuse", "record",
        "--address", address,
        "--duration", f"{duration_s:.0f}",
        "--preset", preset,
        "--outfile", str(outfile),
    ]
    print(f"[record] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a labelled calibration session from a Muse S Athena."
    )
    parser.add_argument("--session-id", required=True,
                        help="Identifier applied to every row of every CSV.")
    parser.add_argument(
        "--task-type",
        default="sustained_attention",
        choices=("baseline_rest", "sustained_attention", "n_back", "code_comprehension"),
        help="Task type recorded to metadata.json.",
    )
    parser.add_argument("--duration", type=float, default=180.0,
                        help="Recording length in seconds (default: 180).")
    parser.add_argument("--subjective-fatigue-score", type=float, default=5.0,
                        help="Self-reported fatigue on 0-10 scale.")
    parser.add_argument("--address", default=None,
                        help="Muse MAC address. Auto-discovered if omitted.")
    parser.add_argument("--preset", default="p1041",
                        help="OpenMuse preset (default: p1041 = full sensor set).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for the decoded CSVs and metadata.")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep the raw .txt recording alongside the decoded CSVs.")
    args = parser.parse_args()

    address = args.address
    if not address:
        print("[discover] scanning for Muse devices (10s)…")
        devices = discover_muse_devices(timeout=10.0, verbose=True)
        if not devices:
            print("[error] no Muse devices found. Power on the headband and ensure BLE is enabled.",
                  file=sys.stderr)
            return 2
        address = devices[0]["address"]
        print(f"[discover] using first device: {devices[0]['name']} ({address})")

    out_dir = Path(args.output_dir)
    raw_path = out_dir / f"{args.session_id}_raw.txt"
    _run_openmuse_record(address=address, duration_s=args.duration, preset=args.preset, outfile=raw_path)

    print(f"[decode] converting {raw_path} -> contract CSVs in {out_dir}")
    paths = decode_raw_session_to_contract(
        raw_file=raw_path,
        session_id=args.session_id,
        output_dir=out_dir,
        task_type=args.task_type,
        subjective_fatigue_score=args.subjective_fatigue_score,
    )

    if not args.keep_raw:
        try:
            raw_path.unlink()
            print(f"[cleanup] removed raw recording {raw_path}")
        except OSError:
            pass

    print("[done] artifacts:")
    for k, v in paths.items():
        print(f"   {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
