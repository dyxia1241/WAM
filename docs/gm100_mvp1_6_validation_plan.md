# GM-100 MVP1.6 Validation Plan

Goal: validate whether the MVP1.5 `cf_1p0` pilot is stable across seeds and whether a stronger Phi loss can recover calibration while keeping action sensitivity.

## Formal Runs

Run seeds `42, 43, 44` on 5060 only.

| label | config | purpose |
| --- | --- | --- |
| `cf_1p0` | `mvp0/configs/gm100_mvp1_joint_flow_cf1p0.yaml` | confirm high action sensitivity from the seed-42 pilot |
| `cf1p0_phi_w20` | `mvp0/configs/gm100_mvp1_joint_flow_cf1p0_phi_w20.yaml` | test whether higher Phi weight improves calibration under strong CF |

## Metrics

Primary:

- `DeltaPhi MAE`
- `DeltaPhi RMSE`
- `coarse_action_cf_ranking_acc`
- `all_negatives_tie_aware_ranking_acc`
- `coarse_action_cf_mean_margin`

Reranking demo:

- `coarse_action_cf_top1_acc`: positive action has higher potential than all coarse negatives for the same state.
- `all_negatives_top1_acc`: positive action has higher potential than all generated negatives for the same state.

Temporal diagnostics remain secondary because current potential labels are linear primitive-time interpolation.

## Decision Rule

Promote `cf_1p0` if it improves three-seed coarse ranking and all-negative ranking over V2 without unacceptable calibration loss.

Prefer `cf1p0_phi_w20` if it keeps most of the `cf_1p0` ranking gain while reducing MAE/RMSE toward V2.

## Outputs

Tracked:

- `docs/gm100_mvp1_6_validation_report.md`
- `docs/figures/gm100_mvp1_6_validation/`

Ignored:

- training checkpoints and raw run outputs under `outputs/`
