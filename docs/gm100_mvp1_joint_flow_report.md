# GM-100 MVP1 Joint Flow Experiment Report

- Machine: 5060 (`dayu@192.168.137.49`)
- Experiment code commit on local and 5060: `1b18043`
- Data: GM-100 50x5 light split, frozen DINOv2 visual features, frozen SigLIP prompt features
- Output root on 5060: `outputs/gm100_mvp1_joint_flow`
- Seeds: `42, 43, 44`

## Architecture / Config

MVP1 uses a lightweight typed-token DiT-style Transformer denoiser.

Condition tokens:

```text
prompt embedding + history observation latents + proprio history
```

Noisy or clamped flow tokens:

```text
future observation latents + future action chunk + DeltaPhi token
```

Model config:

- `history=4`, `horizon=8`, `stride=2`
- `feature_dim=768`, `prompt_dim=768`, `action_dim=14`, `proprio_dim=14`
- `hidden_dim=192`, `transformer_layers=3`, `transformer_heads=4`
- modality embedding + temporal embedding + mask/clamp embedding
- AdaLN-style Transformer blocks with flow timestep conditioning
- modality-specific velocity heads for future obs latent, action, and `DeltaPhi`

Trainable modules:

- all joint-flow projections, embeddings, Transformer attention/MLP blocks, AdaLN timestep MLP, velocity heads

Frozen modules/data and excluded inputs:

- DINOv2 feature extractor, SigLIP text encoder, prompt feature table
- numeric `task_id` and `stage_id` are not used as model embeddings or input tokens

Loss:

```text
L =
  1.0  * MSE(v_obs, v_obs_target)
+ 1.0  * MSE(v_action, v_action_target)
+ 10.0 * MSE(v_phi, v_phi_target)
+ 0.1  * L_cf
```

where `L_cf = -logsigmoid(phi_pos - phi_neg - 0.03)`.

## Results

| seed | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.0157 | 0.0318 | 0.5803 | 0.0008 |
| 43 | 0.0203 | 0.0354 | 0.5086 | 0.0001 |
| 44 | 0.0206 | 0.0345 | 0.6140 | 0.0018 |
| mean+/-std | 0.0189+/-0.0027 | 0.0339+/-0.0019 | 0.5676+/-0.0538 | 0.0009+/-0.0009 |

Per-negative ranking:

| negative | ranking | margin |
| --- | ---: | ---: |
| zero | 0.6439+/-0.1177 | 0.0024+/-0.0021 |
| reverse | 0.5016+/-0.0097 | 0.0000+/-0.0000 |
| shuffle | 0.5027+/-0.0009 | 0.0000+/-0.0001 |
| wrong_arm | 0.5434+/-0.1086 | 0.0009+/-0.0017 |
| scaled_0.25 | 0.6439+/-0.1161 | 0.0018+/-0.0016 |
| scaled_1.75 | 0.5703+/-0.0776 | 0.0004+/-0.0005 |

Flow reconstruction metrics:

| metric | mean+/-std |
| --- | ---: |
| obs_flow_mse | 1.3187+/-0.0177 |
| action_flow_mse | 0.4679+/-0.0522 |
| phi_flow_mse | 0.0656+/-0.0372 |
| future_obs_y0_mse | 0.4297+/-0.0110 |
| action_y0_mse | 0.0887+/-0.0091 |

## Comparison To MVP0

| model | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin |
| --- | ---: | ---: | ---: | ---: |
| MVP0 `stage_action_cf` | 0.0223+/-0.0043 | 0.0360+/-0.0006 | 0.8196+/-0.0272 | 0.0162+/-0.0025 |
| MVP0 `prompt_cf_w10` | 0.0125+/-0.0006 | 0.0330+/-0.0003 | 0.7512+/-0.0215 | 0.0066+/-0.0001 |
| MVP0 `prompt_cf_w20` | 0.0121+/-0.0008 | 0.0321+/-0.0019 | 0.7243+/-0.0226 | 0.0056+/-0.0006 |
| MVP1 joint flow | 0.0189+/-0.0027 | 0.0339+/-0.0019 | 0.5676+/-0.0538 | 0.0009+/-0.0009 |

Comparison figures:

![MVP0 vs MVP1 main metrics](figures/gm100_mvp0_mvp1_comparison/mvp0_mvp1_main_metrics.png)

![MVP0 vs MVP1 calibration vs ranking](figures/gm100_mvp0_mvp1_comparison/mvp0_mvp1_calibration_vs_ranking.png)

![MVP0 vs MVP1 per-negative ranking](figures/gm100_mvp0_mvp1_comparison/mvp0_mvp1_per_negative_ranking.png)

![MVP0 vs MVP1 per-negative margin](figures/gm100_mvp0_mvp1_comparison/mvp0_mvp1_per_negative_margin.png)

## Interpretation

MVP1 closes the engineering loop: prompt-conditioned latent/action/potential joint flow trains, evaluates, scores counterfactual actions, writes checkpoints, writes metrics, and generates figures on 5060.

Scientifically, this first MVP1 is not yet stronger than MVP0. Calibration is acceptable, but action ranking is weak and unstable. It is above random mainly for `zero`, `scaled_0.25`, and sometimes `wrong_arm`; `reverse` and `shuffle` remain effectively random.

The likely bottleneck is critic-mode mismatch: evaluation clamps candidate action at `tau=0` while future obs and `DeltaPhi` start from zeros, but training mostly learns scalar `DeltaPhi` flow and not a robust masked critic trajectory. The seed 42 scatter also shows predicted `DeltaPhi` compressed near low values.

Next changes should focus on:

- train with explicit critic-mode batches more often, not only mixed action clamp;
- use `phi_{t:t+K}` trajectory tokens instead of one scalar `DeltaPhi` token;
- score candidate actions with a short denoising schedule instead of single-step `tau=0`;
- increase or schedule `lambda_cf`, and log CF ranking on validation per epoch;
- keep future obs/action/potential masked-modality training, but make critic mode a first-class training objective.

## Figures / Artifacts

Tracked summary figures:

![MVP1 metric summary](figures/gm100_mvp1_joint_flow/mvp1_metric_summary.png)

![MVP1 ranking by negative](figures/gm100_mvp1_joint_flow/mvp1_ranking_by_negative.png)

Representative seed 42 figures:

- `docs/figures/gm100_mvp1_joint_flow/seed_42/delta_phi_scatter.png`
- `docs/figures/gm100_mvp1_joint_flow/seed_42/per_negative_ranking.png`
- `docs/figures/gm100_mvp1_joint_flow/seed_42/training_curves.png`

5060 output artifacts:

- `outputs/gm100_mvp1_joint_flow/seed_42/mvp1_joint_flow`
- `outputs/gm100_mvp1_joint_flow/seed_43/mvp1_joint_flow`
- `outputs/gm100_mvp1_joint_flow/seed_44/mvp1_joint_flow`
- `outputs/gm100_mvp1_joint_flow/aggregate/metrics_by_seed.csv`
- `outputs/gm100_mvp1_joint_flow/aggregate/aggregate_metrics.json`
- `outputs/gm100_mvp1_joint_flow/aggregate/figures`

Tracked summary data:

- `docs/figures/gm100_mvp1_joint_flow/metrics_by_seed.csv`
- `docs/figures/gm100_mvp1_joint_flow/aggregate_metrics.json`
- `docs/figures/gm100_mvp0_mvp1_comparison/mvp0_mvp1_comparison_metrics.csv`
- `docs/figures/gm100_mvp0_mvp1_comparison/mvp0_mvp1_comparison_metrics.json`

## Verification

- 5060 targeted test: `tests/test_joint_flow.py` -> `4 passed`
- 5060 full test suite: `93 passed, 12 warnings`
- 5060 smoke run: `outputs/gm100_mvp1_joint_flow_smoke/seed_42/mvp1_joint_flow`
- 5060 formal runs: seeds `42`, `43`, `44`
- PNG figures were copied locally and inspected for nonblank rendering.
