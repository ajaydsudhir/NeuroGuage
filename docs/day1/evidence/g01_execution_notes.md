# G01 Execution Notes

Date: 2026-04-21

## Environment Verification

- Command category: runtime environment policy check for `neuroguage`.
- Artifact: `artifacts/reports/day1/environment_verification.json`
- Result: expected environment matched active environment.

## Entrypoint Dry-Run Verification

- Commands executed from `neuroguage`:
  - `scripts/train.py --dry-run`
  - `scripts/evaluate.py --dry-run`
  - `scripts/run_live.py --dry-run`
  - `scripts/replay.py --dry-run`
- Manifest artifacts:
  - `artifacts/reports/day1/entrypoints/train_manifest.json`
  - `artifacts/reports/day1/entrypoints/evaluate_manifest.json`
  - `artifacts/reports/day1/entrypoints/run_live_manifest.json`
  - `artifacts/reports/day1/entrypoints/replay_manifest.json`

## Schema Validator Verification

- Valid-session summary artifact:
  - `artifacts/reports/day1/schema_validation_summary.json`
- Unit test artifact (includes malformed rejection case):
  - `artifacts/reports/day1/schema_validator_tests.xml`
