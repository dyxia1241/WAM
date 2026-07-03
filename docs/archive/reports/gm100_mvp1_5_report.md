# GM-100 MVP1.5 Experiment Report

## Summary

MVP1.5 separates label-faithful coarse action counterfactual metrics from temporal diagnostics and tests whether V2 should checkpoint on coarse action sensitivity.

Main conclusions:

- V3 coarse checkpoint selection reproduced the same three-seed aggregate as V2, so checkpoint selection alone is not a new improvement.
- `cf_1p0` is the strongest seed-42 action-sensitivity pilot: coarse ranking reaches 0.9083 versus V2/V3 seed-42 `0.8050`, but MAE is worse than V2/V3.
- `phi_w20` improves calibration on seed 42, with MAE 0.0130, but weakens coarse ranking, so it is not the best critic candidate.
- The temporal diagnostic group stays near chance for V1/V2/V3, which is expected under linearly interpolated primitive-time labels.

Metric policy:

- Primary action metric: `coarse action CF = zero + wrong_arm + scaled_0.25 + scaled_1.75`.
- Diagnostic temporal metric: `temporal diagnostic = reverse + shuffle`.

## Main Metrics

| family | label | seeds | MAE | RMSE | coarse ranking | coarse margin | all-neg ranking | all-neg margin | temporal ranking |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | `mvp1_v1` | 3 | 0.0189+/-0.0027 | 0.0339+/-0.0019 | 0.6004+/-0.0813 | 0.0014+/-0.0013 | 0.5676+/-0.0538 | 0.0009+/-0.0009 | 0.5021+/-0.0046 |
| main | `mvp1_v2` | 3 | 0.0158+/-0.0040 | 0.0316+/-0.0025 | 0.7816+/-0.0369 | 0.0076+/-0.0045 | 0.6881+/-0.0241 | 0.0051+/-0.0030 | 0.5011+/-0.0039 |
| v3_formal | `mvp1_v3_coarse` | 3 | 0.0158+/-0.0040 | 0.0316+/-0.0025 | 0.7816+/-0.0369 | 0.0076+/-0.0045 | 0.6881+/-0.0241 | 0.0051+/-0.0030 | 0.5011+/-0.0039 |

V3 vs V2 coarse-ranking delta: `-0.0000`. This is effectively no change, so V3 should not replace V2 as a new method claim.

## Seed-42 Component Ablation

These runs are pilot ablations, not final statistics. `v2_full` and `v3_coarse` reuse the corresponding remote seed-42 formal outputs.

| label | seeds | MAE | RMSE | coarse ranking | coarse margin | all-neg ranking | all-neg margin | temporal ranking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cf_multi_only` | 1 | 0.0816+/-0.0000 | 0.0924+/-0.0000 | 0.8437+/-0.0000 | 0.0548+/-0.0000 | 0.7272+/-0.0000 | 0.0369+/-0.0000 | 0.4944+/-0.0000 |
| `critic_aux_only` | 1 | 0.0737+/-0.0000 | 0.0773+/-0.0000 | 0.6513+/-0.0000 | 0.0021+/-0.0000 | 0.6022+/-0.0000 | 0.0014+/-0.0000 | 0.5041+/-0.0000 |
| `phi_traj_only` | 1 | 0.0476+/-0.0000 | 0.0522+/-0.0000 | 0.7515+/-0.0000 | 0.0021+/-0.0000 | 0.6678+/-0.0000 | 0.0014+/-0.0000 | 0.5003+/-0.0000 |
| `v2_full` | 1 | 0.0155+/-0.0000 | 0.0336+/-0.0000 | 0.8050+/-0.0000 | 0.0066+/-0.0000 | 0.7022+/-0.0000 | 0.0044+/-0.0000 | 0.4966+/-0.0000 |
| `v3_coarse` | 1 | 0.0155+/-0.0000 | 0.0336+/-0.0000 | 0.8050+/-0.0000 | 0.0066+/-0.0000 | 0.7022+/-0.0000 | 0.0044+/-0.0000 | 0.4966+/-0.0000 |

Ablation takeaway: stronger CF alone gives high ranking but destroys calibration; phi trajectory alone helps ranking but also hurts MAE; critic-flow auxiliary alone is insufficient. The full V2 recipe is the best balanced seed-42 model among these component tests.

## Seed-42 Pilot Sweep

| label | seeds | MAE | RMSE | coarse ranking | coarse margin | all-neg ranking | all-neg margin | temporal ranking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cf_1p0` | 1 | 0.0193+/-0.0000 | 0.0338+/-0.0000 | 0.9083+/-0.0000 | 0.0266+/-0.0000 | 0.7931+/-0.0000 | 0.0192+/-0.0000 | 0.5626+/-0.0000 |
| `critic_w2` | 1 | 0.0134+/-0.0000 | 0.0302+/-0.0000 | 0.7798+/-0.0000 | 0.0046+/-0.0000 | 0.6870+/-0.0000 | 0.0031+/-0.0000 | 0.5015+/-0.0000 |
| `phi_w20` | 1 | 0.0130+/-0.0000 | 0.0297+/-0.0000 | 0.7645+/-0.0000 | 0.0019+/-0.0000 | 0.6715+/-0.0000 | 0.0013+/-0.0000 | 0.4856+/-0.0000 |
| `steps_8` | 1 | 0.0155+/-0.0000 | 0.0322+/-0.0000 | 0.8026+/-0.0000 | 0.0048+/-0.0000 | 0.7008+/-0.0000 | 0.0032+/-0.0000 | 0.4972+/-0.0000 |

Sweep takeaway: `cf_1p0` is the only pilot that clearly improves action sensitivity, reaching the best coarse and all-negative ranking in this report. `phi_w20` and `critic_w2` improve calibration but reduce ranking. `steps_8` gives no meaningful gain over 4-step scoring while adding compute.

## Recommendation

Next run should be a three-seed formal sweep for `cf_1p0`, plus one calibration-preserving variant such as `cf_1p0 + phi_weight=20` or a checkpoint rule that jointly constrains MAE and coarse ranking. Do not promote V3 as a separate model; treat it as a selection-policy check that matched V2.

## Figures

- `docs/figures/gm100_mvp1_5/mvp1_5_main_metrics.png`
- `docs/figures/gm100_mvp1_5/ablation_coarse_ranking.png`
- `docs/figures/gm100_mvp1_5/sweep_coarse_ranking.png`
- `docs/figures/gm100_mvp1_5/calibration_vs_coarse_ranking.png`

## Files

- `docs/figures/gm100_mvp1_5/metrics_by_run.csv`
- `docs/figures/gm100_mvp1_5/aggregate_metrics.csv`
