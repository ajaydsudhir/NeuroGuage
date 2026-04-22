# NeuroGauge Live Runbook — Muse S Athena

This is the canonical step-by-step for running the NeuroGauge fatigue gauge
on a live **Muse S Athena** headband.  Everything is driven from the
`neuroguage` Conda environment and uses the
[OpenMuse](https://github.com/DominiqueMakowski/OpenMuse) Python package as
the BLE + LSL bridge.

The live app streams EEG, fNIRS/PPG optical, and accelerometer/gyroscope
data in real time, runs the MVP fatigue model, applies the symbolic
guardrail engine, and renders the classroom-projection dashboard.

---

## 0. What you need

| Requirement                         | Notes                                                 |
| ----------------------------------- | ----------------------------------------------------- |
| Muse S Athena headband (charged)    | Front LED solid blue when powered on                 |
| Windows 10/11 machine with BLE 5.0+ | Tested on Windows 10.0.26200                          |
| `neuroguage` Conda env              | `conda env create -f environment.yml` (once)         |
| Python deps                         | `pip install -r requirements.txt` (once)             |
| Muse trained/calibrated             | Day 2–4 model artifacts under `artifacts/models/`.   |

> **No models yet?** The app will still run — it falls back to a built-in
> stub model and the dashboard will show `MODEL: STUB` so you know the risk
> scores are not production-calibrated. Use the calibration workflow in
> §4 to collect data and train real models.

---

## 1. One-time setup

```bash
conda activate neuroguage
pip install -r requirements.txt
python -m neuroguage.cli verify-env --strict      # confirms active env
```

The `requirements.txt` pins `OpenMuse`, `mne-lsl`, and `bleak` so BLE +
LSL are available.  If BLE is blocked (Windows privacy settings, Bluetooth
driver not present, etc.), the live source will fall back to raising a
clean error and you can still run in synthetic or replay mode.

Verify new deps at any time:

```bash
python -c "import OpenMuse, mne_lsl, bleak; print(OpenMuse.__version__, mne_lsl.__version__, bleak.__version__)"
```

---

## 2. Discover your Muse (first time per device)

Power the headband on — you should see a **solid blue** LED.

```bash
python -m neuroguage.cli discover-muse --timeout 10
```

You'll get a JSON list like:

```json
[
  {"name": "MuseS-1234", "address": "00:55:DA:B9:FA:20"}
]
```

Copy the `address` — you'll pass it to the live runner.

Alternatively run the OpenMuse scanner directly:

```bash
OpenMuse find
```

---

## 3. Run the live fatigue gauge

### 3a. Dry-run a synthetic demo (no headband required)

Great for classroom rehearsals, laptop-only demos, or CI smoke tests:

```bash
python scripts/run_live.py \
    --source synthetic \
    --duration 60 \
    --session-id rehearsal_01
```

The dashboard opens with live EEG bands, fNIRS trend, risk score, and
guardrail timeline.  Press `Ctrl+C` to stop early; artifacts are saved
under `artifacts/reports/live/`.

### 3b. Live Muse S Athena run

```bash
python scripts/run_live.py \
    --source muse \
    --address 00:55:DA:B9:FA:20 \
    --preset p1041 \
    --session-id classroom_demo_01 \
    --duration 600
```

Behind the scenes we:

1. Spawn `OpenMuse stream --address … --preset p1041` as a child process.
   This creates four LSL outlets: `Muse-EEG`, `Muse-OPTICS`,
   `Muse-ACCGYRO`, and `Muse-BATTERY`.
2. Subscribe to those outlets via `mne-lsl` and push samples into ring
   buffers matching the Day 2 data contract.
3. Every `step_s` seconds (0.5s by default) run one inference cycle →
   guardrail update → dashboard redraw.
4. Write session artifacts to `artifacts/reports/live/<session-id>_*` on
   shutdown.

Key flags:

| Flag                   | Default      | Notes                                                    |
| ---------------------- | ------------ | -------------------------------------------------------- |
| `--preset`             | `p1041`      | EEG8 + Optics16 + ACC/GYRO + BATTERY (full sensor set). |
| `--no-dashboard`       | off          | Headless run; still writes session artifacts.            |
| `--record-raw FILE`    | off          | Also save the raw BLE packet log to `FILE` for replay.  |
| `--duration SECS`      | unlimited    | Stop automatically after `SECS` seconds.                |
| `--session-id NAME`    | `live`       | Used as filename prefix for all outputs.                |

### 3c. Replay a previously-recorded session

```bash
python scripts/run_live.py \
    --source replay \
    --raw-file data/raw/sub01/raw.txt \
    --real-time-factor 2.0 \
    --duration 120
```

`--real-time-factor` lets you speed up or slow down the replay.  Useful
when post-analysing a 30-minute recording.

---

## 4. Collect a labelled calibration session

Training real models requires paired EEG/optical/IMU + subjective fatigue
scores.  The helper script wraps `OpenMuse record` and decodes the raw
packets into Day-2-contract CSVs:

```bash
python scripts/record_calibration_session.py \
    --session-id sub01_baseline_rest \
    --task-type baseline_rest \
    --duration 180 \
    --subjective-fatigue-score 2 \
    --output-dir data/raw/sub01/baseline_rest
```

Running the script:

1. Scans for a Muse if `--address` is omitted.
2. Invokes `OpenMuse record` to capture the raw BLE packet log.
3. Decodes that log into `eeg.csv`, `optical.csv`, `imu.csv`, and
   `metadata.json` — all conforming to `configs/data_contract.json`.
4. Cleans up the raw file unless you pass `--keep-raw`.

Plug the produced folder straight into the Day 2 schema validator:

```bash
python -m neuroguage.cli validate-schema \
    --eeg      data/raw/sub01/baseline_rest/eeg.csv \
    --optical  data/raw/sub01/baseline_rest/optical.csv \
    --imu      data/raw/sub01/baseline_rest/imu.csv \
    --metadata data/raw/sub01/baseline_rest/metadata.json
```

Then feed the full cohort into Day 3 (`feature_builder`), Day 4
(`training`), and the MVP model-selection pipeline to obtain production
artifacts under `artifacts/models/day4/`.

Recommended protocol (≈30 min total, matches the PRD):

| Task               | Duration | `--task-type`          | Fatigue target |
| ------------------ | -------- | ---------------------- | -------------- |
| Baseline rest      | 3 min    | `baseline_rest`        | 1–2            |
| Sustained attention| 5 min    | `sustained_attention`  | 3–5            |
| n-back working mem | 5 min    | `n_back`               | 5–7            |
| Code comprehension | 10 min   | `code_comprehension`   | 6–8            |
| Drowsy re-rest     | 3 min    | `baseline_rest`        | 7–9            |

---

## 5. Troubleshooting

| Symptom                                                   | Fix                                                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `No Muses found`                                          | Make sure the headband LED is solid blue. Disable/enable Bluetooth. Try `OpenMuse find --timeout 30`.  |
| `BleakError: Device not ready` after a minute of streaming | Battery is low. Charge the headband (LED blinks red while charging).                                    |
| `No LSL stream resolved for modality=...`                 | The BLE connection is up but the OpenMuse child process hasn't advertised yet. Wait ~5s or restart.    |
| `stub_models=true` in the summary                          | No trained model under `artifacts/models/day4/`. Run Day 3 + Day 4 pipelines on collected sessions.     |
| Dashboard opens blank / no motion                          | The first cycle takes `step_s` seconds to warm up.  If still blank after 10s, check the log for faults. |
| `latency` is red (>2s)                                     | Close other GPU-heavy apps; verify `--preset` isn't too wide for your BLE link.                        |

Live logs are written to `logs/run_live_workflow_<timestamp>.log`.

---

## 6. Artifacts you should see after a run

Inside `artifacts/reports/live/`:

| File                                     | Meaning                                                            |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `<session>_summary.json`                 | Cycle count, final state, latency stats, source health.            |
| `<session>_updates.json`                 | Every inference cycle (risk, confidence, latency, fault flags).    |
| `<session>_transitions.json`             | Guardrail transitions with rationale (T13.4).                      |
| `<session>_guardrail_summary.json`       | Aggregate state distribution + risk stats.                         |
| `<session>_session_log.csv`              | Flat CSV of all updates for spreadsheet review.                    |
| `run_live_manifest.json`                 | Workflow manifest (useful for reproducibility evidence).           |

Good luck — and keep an eye on the LED battery indicator before the
class demo.
