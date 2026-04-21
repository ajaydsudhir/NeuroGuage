# Day 1 Governance and Dependency Baseline

Version: v1.0.0  
Date: 2026-04-21

## O-01 Phase-Gated Delivery Policy

- Required success target is AC-001 to AC-005 by end of Day 6.
- Stretch target AC-006 to AC-009 is conditional on passing G06.
- If G06 fails, Day 7 is MVP-only and stretch tasks are deferred.

## O-02 Critical Path Lock

- Locked execution sequence: contract -> ingestion -> features -> training -> inference -> guardrails -> dashboard -> demo.
- Dependency integrity rule: every downstream implementation task must reference at least one upstream task from the locked sequence.
- Validation method: dependencies are checked against `docs/TODO.md` before marking a task complete.

## O-03 Risk-First Execution Policy

Priority order for early control:
1. Overfitting controls (session-held-out split, leakage checks, regularization).
2. Artifact sensitivity controls (IMU quality flags, missing-modality fallback).
3. Latency stability controls (bounded refresh path and logging diagnostics).

Daily checkpoint rule:
- Each day exit gate includes a short risk review with updated status and mitigation ownership.

## O-04 Completion Evidence Standard

Task completion evidence rule:
- A task is complete only when at least one concrete artifact exists (script, config, report, or log).

Gate evidence rule:
- Each day gate must have pass/fail evidence links consolidated in a gate evidence index.
- Day 1 evidence index location: `docs/day1/evidence/day1_evidence_index.md`.

## O-07 Runtime Environment Policy

- All project scripts and demos are executed from Conda environment `neuroguage`.
- Preferred command form on Windows for reproducibility:
  - `conda run -n neuroguage python <script>`
- Environment verification artifacts are required for Day 1 and Day 6 evidence.
