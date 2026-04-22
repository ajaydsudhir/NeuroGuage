# Advanced vs MVP Promotion Decision (T19.4)

- **Decision**: `KEEP_MVP`
- **Promoted**: `False`
- **Metric gate**: FAIL
- **Latency gate**: PASS
- **Stability gate**: PASS
- **Threshold**: +3 points on macro F1 or AUROC (FR-026)
- **Held-out sessions**: s01

## Rationale

Keep MVP as production demo model. Gate outcome: metric gate FAILED: no +3-point lift in macro F1 or AUROC. Advanced prototype artifacts are retained for future iteration.

## Held-out Metric Deltas (identical sessions)

| Metric | MVP | Advanced | Delta | Meets +3pt |
|--------|-----|----------|-------|------------|
| macro_f1 | 0.4000 | 0.4000 | +0.0000 | no |
| auroc | 1.0000 | 1.0000 | +0.0000 | no |
| balanced_accuracy | 0.5000 | 0.5000 | +0.0000 | no |

## Latency Comparison (NFR-001)

| Stat | MVP | Advanced |
|------|-----|----------|
| max (ms)  | 0.323 | 3.314 |
| p95 (ms)  | 0.145 | 2.644 |
| within budget | True | True |

> advanced_max=3.314ms vs mvp_max=0.323ms (tolerance=50.0ms); advanced_p95=2.644ms vs mvp_p95=0.145ms.

## Alert Stability Comparison (NFR-002)

| Stat | MVP | Advanced |
|------|-----|----------|
| worst transitions/min | 7.400 | 7.400 |
| worst max_consecutive_flips | 2 | 2 |

> worst transitions/min: mvp=7.400, advanced=7.400 (tolerance=1.00); worst max_consecutive_flips: mvp=2, advanced=2.

