# GM-100 PP-WAM Experiment Summary

## Current Status

MVP0 established the motivation: primitive-local potential must be action-conditioned and needs counterfactual supervision. MVP1 moves the method to a lightweight DiT-style joint flow over:

```text
future observation latent + future action chunk + primitive-local potential trajectory
```

The model conditions on frozen visual latents, frozen prompt features, and proprioception. It does not use numeric `task_id` or `stage_id` as model input in prompt/joint-flow experiments.

## Architecture

Current MVP1 backbone:

- typed-token Transformer / lightweight DiT denoiser;
- condition tokens: prompt embedding, observation-history latents, proprio history;
- flow tokens: future observation latents, action tokens, `phi_tokens=8`;
- modality, temporal, mask/clamp, and flow-timestep embeddings;
- trainable projections, attention/MLP blocks, AdaLN-style timestep conditioning, and velocity heads;
- frozen DINOv2 visual feature extractor and frozen SigLIP prompt encoder.

Current main config:

```text
configs/gm100/joint_flow_cf1p0.yaml
```

Key settings:

```text
history=4, horizon=8, stride=2
hidden_dim=192, layers=3, heads=4
phi_tokens=8, phi_target_mode=delta_trajectory
denoise_steps=4, train_denoise_steps=2
obs_weight=1.0, action_weight=1.0, phi_weight=10.0
critic_flow_weight=1.0, counterfactual_weight=1.0, margin=0.03
```

Training loss:

```text
L = MSE(v_obs, v_obs_target)
  + MSE(v_action, v_action_target)
  + 10 * MSE(v_phi, v_phi_target)
  + 1 * L_critic_flow
  + 1 * L_cf
```

where `L_cf = -logsigmoid(phi_pos - phi_neg - margin)`.

## Main Results

| model | seeds | DeltaPhi MAE | DeltaPhi RMSE | coarse ranking | all-neg ranking | coarse top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MVP1 V1 | 3 | 0.0189+/-0.0027 | 0.0339+/-0.0019 | 0.6004+/-0.0813 | 0.5676+/-0.0538 | n/a |
| MVP1 V2 | 3 | 0.0158+/-0.0040 | 0.0316+/-0.0025 | 0.7816+/-0.0369 | 0.6881+/-0.0241 | 0.4102+/-0.0512 |
| MVP1.6 `cf_1p0` | 3 | 0.0187+/-0.0027 | 0.0335+/-0.0026 | 0.8870+/-0.0231 | 0.7801+/-0.0132 | 0.7301+/-0.0543 |
| MVP1.6 `cf1p0_phi_w20` | 3 | 0.0200+/-0.0050 | 0.0335+/-0.0023 | 0.7797+/-0.0337 | 0.6850+/-0.0216 | 0.4192+/-0.0522 |

MVP0 references:

| model | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin |
| --- | ---: | ---: | ---: | ---: |
| MVP0 `stage_action_cf` | 0.0223+/-0.0043 | 0.0360+/-0.0006 | 0.8196+/-0.0272 | 0.0162+/-0.0025 |
| MVP0 `prompt_cf_w10` | 0.0125+/-0.0006 | 0.0330+/-0.0003 | 0.7512+/-0.0215 | 0.0066+/-0.0001 |
| MVP0 `prompt_cf_w20` | 0.0121+/-0.0008 | 0.0321+/-0.0019 | 0.7243+/-0.0226 | 0.0056+/-0.0006 |

## Interpretation

MVP1 V1 is the useful negative result: naive joint flow does not automatically become a critic.

MVP1 V2 shows the mechanism: `phi` trajectory, multi-step scoring, critic-flow auxiliary loss, and stronger CF supervision substantially improve action sensitivity.

MVP1.6 `cf_1p0` is the current best joint-flow critic. It improves over V2 by `+0.1054` coarse ranking, `+0.0920` all-negative ranking, and `+0.3199` coarse top-1. The cost is modest calibration degradation: MAE moves from `0.0158` to `0.0187`.

`cf1p0_phi_w20` did not solve the calibration/ranking tradeoff. The next calibration direction should be constrained checkpointing, post-hoc calibration, or two-term model selection rather than a simple global `phi_weight` increase.

## Figure Pointers

- `docs/figures/current/gm100_mvp1_6_validation/mvp1_6_main_metrics.png`
- `docs/figures/current/gm100_mvp1_6_validation/calibration_vs_coarse_ranking.png`

Older comparison plots are archived under `docs/archive/figures/`.
