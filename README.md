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

Replay mode — required for day-to-day validation and class dry-runs:

```powershell
python scripts/day4_inference_guardrails.py
```

Dedicated live entry point (Muse stream):

```powershell
python scripts/run_live.py
```

Dedicated replay entry point:

```powershell
python scripts/replay.py
```

Artifacts:

- `artifacts/reports/day4/inference_diagnostics.json` — per-update risk,
  latency, confidence, fault flags
- `artifacts/reports/day4/guardrail_transition_log.json` — timestamped
  GREEN / YELLOW / RED transitions with rationale
- `artifacts/reports/day4/guardrail_summary.json`
- `artifacts/reports/day4/day4_integration_summary.json`

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

## 4. Test Suite

```powershell
conda activate neuroguage
python -m pytest -q
```

The full suite includes ingestion, windowing, feature extraction,
training, evaluation, live inference, guardrails, dashboard, demo
orchestration, and Day 6 hardening tests.

## 5. Reproducibility

- `environment.yml` and `requirements.txt` pin all core dependencies.
- `configs/data_contract.json` freezes the session schema and sampling
  assumptions.
- `configs/demo_timing.json` freezes the 5 to 10 minute demo envelope.
- Every training run writes a manifest with split IDs, dataset version
  tag, hyperparameters, and metric evidence.
- The Day 6 release bundle (`artifacts/release/manifest.json`) contains
  SHA-256 hashes for every shipped artifact plus an `evidence_index.md`
  human-readable index.

## 6. Acceptance Criteria (MVP)

Authoritative checklist:
`docs/day1/checklists/mvp_acceptance_ac_001_to_ac_005.md`

Auto-generated Day 6 evidence:
`artifacts/reports/day6/acceptance_results.json`
`artifacts/reports/day6/day6_exit_gate.json`

Phase-gate rule (`docs/day1/day6_go_no_go_rule.md`): Day 7 stretch work
is allowed only if AC-001 through AC-005 all pass with objective
evidence.

## 7. Documentation Index

- `docs/PRD.md` — Product Requirements Document
- `docs/proposal.md` — Original project proposal
- `docs/TODO.md` — 7-day execution plan with daily exit gates
- `docs/day6/demo_runbook.md` — Class quick demo runbook and fallbacks
- `docs/day6/limitations_and_risks.md` — Known limitations, risks, and
  deferred items
- `docs/day1/governance_and_dependencies.md` — Governance, dependency
  integrity, and no-orphan task policy

## 8. License

See `LICENSE`.
