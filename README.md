# NeuroGauge

NeuroGauge is a real-time cognitive fatigue monitoring project for the Muse S Athena headband.
It combines:

- signal processing over EEG + optical + IMU streams,
- a neural dual-head model (`attention` + `metabolic readiness`),
- and a neuro-symbolic guardrail layer that raises intervention states.

The live system runs with two variants:

- `muse_demo`: lightweight class-demo variant.
- `research_eegnet`: higher-capacity research-style variant.

## Repository Layout

### Core package (`src/neuroguage`)

- `pipeline.py`: offline data prep, feature building, labeling, train/eval utilities.
- `dual_head.py`: dual-head model architectures, training, and inference helpers.
- `inference.py`: online late-fusion risk inference path.
- `guardrails.py`: symbolic state logic and mismatch rule.
- `live_app.py`: end-to-end runtime orchestration.
- `muse_stream.py`: Muse/OpenMuse stream and replay adapters.
- `dashboard.py`: live dashboard rendering.
- `lifecycle.py`: release/demo/hardening helper flows.
- `cli.py`: command entry points.

### Main scripts (`scripts`)

- `train.py`: train the MVP pipeline.
- `train_dual_head.py`: train the dual-head model (`muse_demo` or `research_eegnet`).
- `run_live.py`: full explicit live run command.
- `run_demo.py`: classroom-friendly wrapper around `run_live.py`.
- `replay.py`: replay-based run from existing data.
- `record_calibration_session.py`: optional helper to record a labeled Muse session.

## Quick Start (Clone to Run)

### 1) Clone and install

```powershell
git clone git@github.com:ajaydsudhir/NeuroGuage.git
cd NeuroGuage
conda env create -f environment.yml
conda activate neuroguage
pip install -e .
```

### 2) Validate installation

```powershell
python -m pytest -q
```

If tests pass, the environment is set up correctly.

## How To Run

### A) Train baseline + dual-head models

```powershell
python scripts/train.py
python scripts/train_dual_head.py --model-variant muse_demo
python scripts/train_dual_head.py --model-variant research_eegnet
```

### B) Run with Muse S Athena (demo setup)

1. Power on the Muse S Athena and confirm Bluetooth is available.
2. Find the device MAC address:

```powershell
python -m neuroguage.cli discover-muse --timeout 10
```

3. Start class demo run:

```powershell
python scripts/run_demo.py --address <MUSE_MAC> --duration 180
```

### C) Run explicit live command

```powershell
python scripts/run_live.py --source muse --address <MUSE_MAC> --duration 300 --use-dual-head
```

### D) Run research variant

```powershell
python scripts/run_live.py --source muse --address <MUSE_MAC> --duration 300 --use-dual-head --dual-head-weights artifacts/models/dual_head_research_eegnet_accv2/dual_head.pt --dual-head-config artifacts/models/dual_head_research_eegnet_accv2/config.json --guardrail-config configs/guardrails_research_eegnet.json
```

## Outputs You Should Check

After a successful run, check `artifacts/reports/live/` for:

- `<session>_summary.json`
- `<session>_session_log.csv`
- `<session>_transitions.json`
- `<session>_mismatch_events.json` (when dual-head is enabled)

These are the main files used to verify run quality and intervention behavior.

## Troubleshooting

- **No Muse found:** confirm headband is on, charged, and not connected to another app.
- **LSL/stream startup fails:** restart OpenMuse bridge and rerun `run_live.py`.
- **Model artifact missing:** rerun the appropriate training script or point `--dual-head-weights` and `--dual-head-config` to an existing trained model directory.

## Remaining docs

- `docs/pipeline_outline.md`: end-to-end processing overview.
- `docs/architecture.md`: module-level architecture summary.
