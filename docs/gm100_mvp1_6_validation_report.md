# GM-100 MVP1.6 Formal Validation Report

## Summary

This validation runs the strongest MVP1.5 action-sensitivity pilot across three seeds and tests a calibration-preserving variant.

Candidate configs:

- `cf_1p0`: V2/V3 recipe with `counterfactual_weight=1.0`.
- `cf1p0_phi_w20`: same as `cf_1p0`, but `phi_weight=20.0`.

Main conclusions:

- `cf_1p0` validates the MVP1.5 pilot across three seeds: coarse ranking improves by `+0.1054` and all-negative ranking improves by `+0.0920` over V2.
- `cf_1p0` also improves coarse top-1 reranking by `+0.3199` over V2, so the gain is visible in candidate selection, not only pairwise ranking.
- `cf1p0_phi_w20` does not recover a better tradeoff: MAE is slightly worse than `cf_1p0`, and ranking falls back near V2.

## Main Metrics

| family | label | seeds | MAE | RMSE | coarse ranking | all-neg ranking | coarse top-1 | all-neg top-1 | coarse margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `mvp1_v2` | 3 | 0.0158+/-0.0040 | 0.0316+/-0.0025 | 0.7816+/-0.0369 | 0.6881+/-0.0241 | 0.4102+/-0.0512 | 0.1084+/-0.0076 | 0.0076+/-0.0045 |
| formal_validation | `cf_1p0` | 3 | 0.0187+/-0.0027 | 0.0335+/-0.0026 | 0.8870+/-0.0231 | 0.7801+/-0.0132 | 0.7301+/-0.0543 | 0.2423+/-0.0129 | 0.0244+/-0.0019 |
| formal_validation | `cf1p0_phi_w20` | 3 | 0.0200+/-0.0050 | 0.0335+/-0.0023 | 0.7797+/-0.0337 | 0.6850+/-0.0216 | 0.4192+/-0.0522 | 0.1115+/-0.0108 | 0.0077+/-0.0041 |

## Interpretation

- `cf_1p0` coarse-ranking delta over V2: `+0.1054`.
- `cf1p0_phi_w20` MAE delta versus `cf_1p0`: `+0.0013`.
- Top-1 reranking measures whether the positive action is scored above every generated candidate negative for the same state.

## Figures

- `docs/figures/gm100_mvp1_6_validation/mvp1_6_main_metrics.png`
- `docs/figures/gm100_mvp1_6_validation/calibration_vs_coarse_ranking.png`

## Files

- `docs/figures/gm100_mvp1_6_validation/metrics_by_run.csv`
- `docs/figures/gm100_mvp1_6_validation/aggregate_metrics.csv`
