# NeuroGauge : Real-time Detection of Cognitive Fatigue

NeuroGauge is a real-time multimodal cognitive fatigue risk monitor that
combines EEG, optical (fNIRS-style), and IMU streams from a consumer
wearable (Muse S Athena) with an interpretable symbolic guardrail engine.
The system is delivered in two phases per `docs/PRD.md`:

1. **Phase 1 MVP** — reliable, demonstrable, end-to-end pipeline suitable
   for a 5 to 10 minute class demo (AC-001 through AC-005).
2. **Phase 2 Graduate Enhancement** (optional) — advanced temporal
   multimodal model with strict promotion gates over the MVP.

See `docs/proposal.md`, `docs/PRD.md`, and `docs/TODO.md` for the full
product brief, requirements, and execution plan.

## 1. Repository Layout

```
NeuroGauge/
  configs/                 # Frozen data contract, demo timing envelope
  docs/                    # PRD, proposal, TODO, day-by-day evidence
    day1/checklists/       # MVP + stretch acceptance checklists
    day6/                  # Demo runbook + limitations and risks
  scripts/                 # Entry-point runners for train/eval/live/replay/day-by-day
  src/neuroguage/          # Library modules (ingestion, features, training,
                           #   inference, guardrails, dashboard, demo,
                           #   hardening, release)
  tests/                   # Pytest suite (100+ tests)
  artifacts/               # Generated artifacts (models, features, reports,
                           #   release bundle)
```

## 2. Environment (Required: Conda `neuroguage`)

All scripts, demos, and tests must be executed from the Conda environment
named `neuroguage` (per PRD §7.1 and NFR-004).

### 2.1 One-time setup (Windows / PowerShell)

```powershell
conda env create -f environment.yml
conda activate neuroguage
pip install -e .
```

If you prefer a pip-only setup (no Conda):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2.2 Verify the environment

```powershell
conda activate neuroguage
python scripts/verify_environment.py
```

The verifier prints pinned versions for numpy, pandas, scipy,
scikit-learn, matplotlib, and torch. Mismatches are logged as warnings.

## 3. Core Workflows

Every command below assumes the `neuroguage` Conda environment is active.

### 3.1 Validate a session against the data contract

```powershell
python scripts/validate_schema.py --session <path-to-session.json>
```

The schema validator (Day 1 `T03.4`) rejects malformed sessions early and
writes a validation summary to the session directory.

### 3.2 Train the MVP pipeline end-to-end

```powershell
# Day 2: ingest, align, and build features
python scripts/day2_sync_session.py
python scripts/day2_windowing.py
python scripts/day2_eeg_features.py
python scripts/day2_optical_imu_features.py

# Day 3: unified feature matrix, weak labels, splits, training
python scripts/day3_build_features.py
python scripts/day3_build_labels_and_splits.py
python scripts/day3_train_mvp.py
```

Or use the high-level entry point:

```powershell
python scripts/train.py
```

Artifacts land in:
- `artifacts/features/day3/` — unified feature matrix, labeled frame, splits
- `artifacts/models/day3/` — linear + tree + calibrated late-fusion models
- `artifacts/reports/day3/` — training metadata and split manifests

### 3.3 Evaluate with ablations and pick the production MVP

```powershell
python scripts/day4_evaluate_mvp.py
```

Or:

```powershell
python scripts/evaluate.py
```

Produces session-held-out metrics (AUROC, macro F1, balanced accuracy) for
three variants (`eeg_only`, `eeg_optical`, `eeg_optical_imu_gated`), picks
the production MVP, and writes:

- `artifacts/reports/day4/evaluation_summary.json`
- `artifacts/reports/day4/ablation_report.json`
- `artifacts/reports/day4/model_selection.json`
- `artifacts/reports/day4/evaluation_narrative.md`
- `artifacts/models/day4/ablations/<variant>/`

### 3.4 Run live inference (or replay mode)

Replay a canned Day 4 session (no hardware):

```powershell
python scripts/day4_inference_guardrails.py
```

Live Muse S Athena run (see full walkthrough in
`docs/runbooks/muse_s_athena_live.md`):

```powershell
# Discover your headband (one-time)
python -m neuroguage.cli discover-muse --timeout 10

# Run the live fatigue gauge with dashboard
python scripts/run_live.py --source muse --address <MAC> --session-id classroom_demo_01 --duration 600
```

Synthetic rehearsal (laptop only, no headband):

```powershell
python scripts/run_live.py --source synthetic --duration 60
```

Replay a previously recorded raw session:

```powershell
python scripts/run_live.py --source replay --raw-file data/raw/sub01/session_raw.txt
```

Collect a labelled calibration session (writes contract-shaped CSVs):

```powershell
python scripts/record_calibration_session.py \
    --session-id sub01_baseline_rest \
    --task-type baseline_rest \
    --duration 180 \
    --subjective-fatigue-score 2 \
    --output-dir data/raw/sub01/baseline_rest
```

Live artifacts land in `artifacts/reports/live/<session>_{summary,updates,transitions,session_log,guardrail_summary}.*`.
Day-4 replay artifacts land in `artifacts/reports/day4/`.

### 3.5 Launch the dashboard and class quick demo

```powershell
python scripts/day5_dashboard_demo.py
```

Runs:
- `T14` Dashboard with EEG bands, optical trend, risk score, state
  badge, stream health indicators, and guardrail timeline — validated
  for >= 1280x720 classroom projection.
- `T15` Demo orchestrator with two dry runs + defect log.
- Writes `artifacts/reports/day5/dashboard_snapshot.png`,
  `dashboard_validation.json`, `demo_summary.json`, `dry_run_*_report.json`,
  `defect_log.json`, and `day5_exit_gate.json`.

For a guided interactive demo, follow `docs/day6/demo_runbook.md` which
freezes the 5 to 10 minute class envelope at `configs/demo_timing.json`.

### 3.6 Day 6 hardening and release packaging

```powershell
python scripts/day6_hardening_release.py
```

Runs:
- `T16.1` dry-run defect triage.
- `T16.2` refresh-latency benchmark against NFR-001 (<= 2 s).
- `T16.3` alert-stability stress scenarios (noisy risk, motion spikes,
  sustained ramp).
- `T16.4` AC-001..AC-005 automated verification and checklist update.
- `T17.4` release bundle assembly at `artifacts/release/` with an
  indexed manifest and SHA-256 hashes.

### 3.7 Day 7 stretch enhancement and final packaging

```powershell
python scripts/day7_stretch_and_final.py
```

Runs:
- `T18` trains the BiGRU-attention prototype on the identical held-out
  split, persists `.pt` weights, and benchmarks latency + stability on
  the same harness as the MVP.
- `T19` applies the FR-026 / AC-007 promotion gate (+3 points on
  macro F1 or AUROC, no latency/stability regression) and records the
  decision note in `artifacts/reports/day7/promotion_decision.md`.
- `T20` assembles the release-candidate bundle, runs one full
  class-demo rehearsal under the mandated timing constraints, and
  publishes the final status summary + `day7_exit_gate.json`.

## 4. Test Suite

```powershell
conda activate neuroguage
python -m pytest -q
```

The full suite includes ingestion, windowing, feature extraction,
training, evaluation, live inference, guardrails, dashboard, demo
orchestration, Day 6 hardening, and Day 7 advanced-model + promotion +
finalization tests.

## 5. Reproducibility

- `environment.yml` and `requirements.txt` pin all core dependencies.
- `configs/data_contract.json` freezes the session schema and sampling
  assumptions.
- `configs/demo_timing.json` freezes the 5 to 10 minute demo envelope.
- Every training run writes a manifest with split IDs, dataset version
  tag, hyperparameters, and metric evidence.
- The release bundle (`artifacts/release/manifest.json`) contains
  SHA-256 hashes for every shipped artifact plus an `evidence_index.md`
  human-readable index.  Day 7 extends the bundle with the advanced
  BiGRU-attention model, promotion decision, rehearsal report, and the
  `final_status_summary.*` files under `reports/day7/`.

## 6. Acceptance Criteria

Authoritative checklists:
- MVP: `docs/day1/checklists/mvp_acceptance_ac_001_to_ac_005.md`
- Stretch: `docs/day1/checklists/stretch_acceptance_ac_006_to_ac_009.md`

Auto-generated evidence:
- `artifacts/reports/day6/acceptance_results.json`
- `artifacts/reports/day6/day6_exit_gate.json`
- `artifacts/reports/day7/promotion_decision.json`
- `artifacts/reports/day7/final_status_summary.json`
- `artifacts/reports/day7/day7_exit_gate.json`

Phase-gate rule (`docs/day1/day6_go_no_go_rule.md`): Day 7 stretch work
is allowed only if AC-001 through AC-005 all pass with objective
evidence.  Promotion of the advanced model additionally requires the
FR-026 threshold (+3 points macro F1 or AUROC) and no latency/stability
regression.

## 7. Documentation Index

- `docs/PRD.md` — Product Requirements Document
- `docs/proposal.md` — Original project proposal
- `docs/TODO.md` — 7-day execution plan with daily exit gates
- `docs/runbooks/muse_s_athena_live.md` — Step-by-step Muse S Athena
  live runbook (discover → stream → dashboard → artifacts)
- `docs/day6/demo_runbook.md` — Class quick demo runbook and fallbacks
- `docs/day6/limitations_and_risks.md` — Known limitations, risks, and
  deferred items
- `docs/day7/quick_start.md` — In-bundle quick-start for the final
  release-candidate demo folder
- `docs/day1/governance_and_dependencies.md` — Governance, dependency
  integrity, and no-orphan task policy
- `artifacts/reports/day1/dependency_integrity.md` — Machine-generated
  O-06 dependency-integrity audit (regenerate via
  `python scripts/audit_dependency_integrity.py`)

## 8. License

See `LICENSE`.
