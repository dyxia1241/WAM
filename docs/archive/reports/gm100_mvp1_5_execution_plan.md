# GM-100 MVP1.5 Execution Plan

Goal: improve MVP1 V2 into a more stable critic-mode-aware joint-flow baseline, while keeping reporting aligned with the current linear-time primitive potential labels.

## Metric Policy

Primary action-sensitivity metric:

```text
coarse action CF = zero + wrong_arm + scaled_0.25 + scaled_1.75
```

Diagnostic metric:

```text
temporal diagnostic = reverse + shuffle
```

Reason: current `DeltaPhi` ground truth is linearly interpolated over primitive time, so reverse/shuffle are not strongly label-supported causal negatives.

## Experiments

### V3 Coarse-Selected Formal

Run seeds `42, 43, 44` with the V2 objective but checkpoint by:

```text
val/coarse_action_cf_ranking_acc
```

Config:

```text
configs/gm100_mvp1_joint_flow_v3_coarse.yaml
```

Target:

```text
coarse CF ranking >= 0.82
all-neg ranking >= 0.72
DeltaPhi MAE <= 0.015
margin >= 0.006
```

### Seed-42 Component Ablation

Run single-seed ablations to estimate which V2 components matter:

| label | change relative to V1 |
| --- | --- |
| `phi_traj_only` | `phi_tokens=8`, trajectory target |
| `critic_aux_only` | explicit action-clamped critic-flow loss |
| `cf_multi_only` | stronger CF loss and multi-negative sampling |
| `v2_full` | existing V2 full recipe |
| `v3_coarse` | V2 full with coarse checkpoint selection |

### Seed-42 Pilot Sweep

Run a small sweep around V3:

| label | change relative to V3 |
| --- | --- |
| `cf_1p0` | `counterfactual_weight=1.0` |
| `phi_w20` | `phi_weight=20` |
| `critic_w2` | `critic_flow_weight=2.0` |
| `steps_8` | `score.denoise_steps=8` |

## Outputs

All training artifacts stay on 5060 under `outputs/`.

Tracked artifacts:

- aggregate CSV/JSON under `docs/figures/gm100_mvp1_5/`
- comparison PNG figures under `docs/figures/gm100_mvp1_5/`
- final report under `docs/gm100_mvp1_5_report.md`

## Interpretation

If V3 improves coarse CF ranking without sacrificing calibration, continue with V3 as the MVP2 base. If pilot sweep improves V3, run the best sweep setting across seeds before promoting it.
