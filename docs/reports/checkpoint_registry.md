# Checkpoint Registry

Checkpoint files are stored on the 5060 only and are intentionally not tracked by git.

5060 repo root:

```text
/data/projects/WAM
```

Current code/report commit:

```text
42a05ba Simplify PP-WAM repo layout
```

## Main Joint-Flow Checkpoints

| model | seed | checkpoint | config | MAE | RMSE | coarse ranking | all-neg ranking | coarse margin |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MVP1 V1 | 42 | `outputs/gm100_mvp1_joint_flow/seed_42/mvp1_joint_flow/best.pt` | `configs/archive/gm100_mvp1_joint_flow_v1.yaml` | 0.0157 | 0.0318 | n/a | 0.5803 | n/a |
| MVP1 V1 | 43 | `outputs/gm100_mvp1_joint_flow/seed_43/mvp1_joint_flow/best.pt` | `configs/archive/gm100_mvp1_joint_flow_v1.yaml` | 0.0203 | 0.0354 | n/a | 0.5086 | n/a |
| MVP1 V1 | 44 | `outputs/gm100_mvp1_joint_flow/seed_44/mvp1_joint_flow/best.pt` | `configs/archive/gm100_mvp1_joint_flow_v1.yaml` | 0.0206 | 0.0345 | n/a | 0.6140 | n/a |
| MVP1 V2 | 42 | `outputs/gm100_mvp1_joint_flow_v2/seed_42/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_v2.yaml` | 0.0155 | 0.0336 | report-only | 0.7022 | report-only |
| MVP1 V2 | 43 | `outputs/gm100_mvp1_joint_flow_v2/seed_43/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_v2.yaml` | 0.0120 | 0.0288 | report-only | 0.6602 | report-only |
| MVP1 V2 | 44 | `outputs/gm100_mvp1_joint_flow_v2/seed_44/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_v2.yaml` | 0.0200 | 0.0324 | report-only | 0.7018 | report-only |
| MVP1 V3 coarse-selected | 42 | `outputs/gm100_mvp1_joint_flow_v3_coarse/seed_42/mvp1_joint_flow/best.pt` | `configs/archive/gm100_mvp1_joint_flow_v3_coarse.yaml` | 0.0155 | 0.0336 | 0.8050 | 0.7022 | 0.0066 |
| MVP1 V3 coarse-selected | 43 | `outputs/gm100_mvp1_joint_flow_v3_coarse/seed_43/mvp1_joint_flow/best.pt` | `configs/archive/gm100_mvp1_joint_flow_v3_coarse.yaml` | 0.0120 | 0.0288 | 0.7390 | 0.6602 | 0.0037 |
| MVP1 V3 coarse-selected | 44 | `outputs/gm100_mvp1_joint_flow_v3_coarse/seed_44/mvp1_joint_flow/best.pt` | `configs/archive/gm100_mvp1_joint_flow_v3_coarse.yaml` | 0.0200 | 0.0324 | 0.8007 | 0.7018 | 0.0125 |
| MVP1.6 `cf_1p0` | 42 | `outputs/gm100_mvp1_6_cf1p0/seed_42/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_cf1p0.yaml` | 0.0193 | 0.0338 | 0.9083 | 0.7931 | 0.0266 |
| MVP1.6 `cf_1p0` | 43 | `outputs/gm100_mvp1_6_cf1p0/seed_43/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_cf1p0.yaml` | 0.0210 | 0.0359 | 0.8624 | 0.7667 | 0.0239 |
| MVP1.6 `cf_1p0` | 44 | `outputs/gm100_mvp1_6_cf1p0/seed_44/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_cf1p0.yaml` | 0.0158 | 0.0308 | 0.8903 | 0.7804 | 0.0228 |
| MVP1.6 `cf1p0_phi_w20` | 42 | `outputs/gm100_mvp1_6_cf1p0_phi_w20/seed_42/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_cf1p0_phi_w20.yaml` | 0.0239 | 0.0362 | 0.7733 | 0.6815 | 0.0076 |
| MVP1.6 `cf1p0_phi_w20` | 43 | `outputs/gm100_mvp1_6_cf1p0_phi_w20/seed_43/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_cf1p0_phi_w20.yaml` | 0.0144 | 0.0321 | 0.7497 | 0.6654 | 0.0036 |
| MVP1.6 `cf1p0_phi_w20` | 44 | `outputs/gm100_mvp1_6_cf1p0_phi_w20/seed_44/mvp1_joint_flow/best.pt` | `configs/gm100/joint_flow_cf1p0_phi_w20.yaml` | 0.0218 | 0.0323 | 0.8162 | 0.7082 | 0.0118 |

`report-only` means the raw checkpoint metrics file predates grouped metric serialization; the aggregate grouped values are available in the current experiment summary.

## Current Aggregate Reference

| model | seeds | MAE | RMSE | coarse ranking | all-neg ranking | coarse top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MVP1 V1 | 3 | 0.0189+/-0.0027 | 0.0339+/-0.0019 | 0.6004+/-0.0813 | 0.5676+/-0.0538 | n/a |
| MVP1 V2 | 3 | 0.0158+/-0.0040 | 0.0316+/-0.0025 | 0.7816+/-0.0369 | 0.6881+/-0.0241 | 0.4102+/-0.0512 |
| MVP1.6 `cf_1p0` | 3 | 0.0187+/-0.0027 | 0.0335+/-0.0026 | 0.8870+/-0.0231 | 0.7801+/-0.0132 | 0.7301+/-0.0543 |
| MVP1.6 `cf1p0_phi_w20` | 3 | 0.0200+/-0.0050 | 0.0335+/-0.0023 | 0.7797+/-0.0337 | 0.6850+/-0.0216 | 0.4192+/-0.0522 |

Primary report:

```text
docs/reports/gm100_ppwam_experiment_summary.md
```

## Other Existing Checkpoint Groups

| group | count | purpose |
| --- | ---: | --- |
| `outputs/gm100_action_fair_s42_44/seed_*/*/best.pt` | 9 | MVP0 action-fair comparison seeds 42/43/44 |
| `outputs/gm100_joint_formal/*/best.pt` | 6 | MVP0 formal discriminative critic baselines |
| `outputs/gm100_prompt_formal/seed_*/*/best.pt` | 9 | prompt-conditioned MVP0 baselines |
| `outputs/gm100_prompt_phi_weight_sweep/delta_w_*/seed_*/obs_action_prompt_cf_multi/best.pt` | 15 | prompt CF delta-weight sweep |
| `outputs/gm100_mvp1_5_ablation/*/seed_42/mvp1_joint_flow/best.pt` | 3 | MVP1.5 component ablation |
| `outputs/gm100_mvp1_5_sweep/*/seed_42/mvp1_joint_flow/best.pt` | 4 | MVP1.5 hyperparameter sweep |
| `outputs/*_smoke/**/best.pt` | 15 | smoke/debug checkpoints, not paper evidence |

## Planned Strong Baseline

The next non-joint-flow baseline config is tracked but not yet trained:

```text
configs/gm100/phi_only_cf1p0.yaml
```

It uses the same prompt, observation-history, proprioception, action chunk, `phi_tokens=8`, and CF supervision as `cf_1p0`, but removes future-observation and action flow modeling. Training this baseline should be confirmed before launching on the 5060.
