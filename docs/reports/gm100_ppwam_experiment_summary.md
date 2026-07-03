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
| phi-only strong baseline `cf1p0` | 3 | 0.0256+/-0.0137 | 0.0366+/-0.0108 | 0.9084+/-0.0186 | 0.7911+/-0.0334 | 0.7790+/-0.0670 |

MVP0 references:

| model | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin |
| --- | ---: | ---: | ---: | ---: |
| MVP0 `stage_action_cf` | 0.0223+/-0.0043 | 0.0360+/-0.0006 | 0.8196+/-0.0272 | 0.0162+/-0.0025 |
| MVP0 `prompt_cf_w10` | 0.0125+/-0.0006 | 0.0330+/-0.0003 | 0.7512+/-0.0215 | 0.0066+/-0.0001 |
| MVP0 `prompt_cf_w20` | 0.0121+/-0.0008 | 0.0321+/-0.0019 | 0.7243+/-0.0226 | 0.0056+/-0.0006 |

## Hard Candidate Reranking

Hard reranking evaluates existing checkpoints without new training. For each test anchor, the logged action is the positive candidate. Four data-bank distractors are selected from the same test split:

- `same_task_phase_wrong`: same task, different stage, closest primitive progress.
- `same_task_far_progress`: same task, farthest primitive progress.
- `cross_task`: different task, preferring the same stage and closest primitive progress.
- `nearest_obs_wrong_action`: nearest observation-history latent with a different task or stage.

This is a logged-action retrieval diagnostic, not a true counterfactual rollout label. It tests whether the model ranks the observed expert action above plausible action-bank distractors.

| model | seeds | anchors/seed | hard pairwise ranking | hard top-1 | tie-aware top-1 | margin to best neg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MVP1.6 `cf_1p0` | 3 | 5552 | 0.8336+/-0.0756 | 0.6391+/-0.1341 | 0.6614+/-0.1420 | 0.0104+/-0.0078 |
| phi-only strong baseline `cf1p0` | 3 | 5552 | 0.8795+/-0.0304 | 0.7248+/-0.0590 | 0.7462+/-0.0669 | 0.0222+/-0.0020 |

Per-candidate ranking:

| model | phase-wrong | far-progress | cross-task | nearest-obs wrong |
| --- | ---: | ---: | ---: | ---: |
| MVP1.6 `cf_1p0` | 0.8415+/-0.0652 | 0.8246+/-0.0590 | 0.8531+/-0.0686 | 0.8151+/-0.1138 |
| phi-only strong baseline `cf1p0` | 0.8850+/-0.0386 | 0.8547+/-0.0068 | 0.8968+/-0.0325 | 0.8813+/-0.0459 |

The first hard reranking pass strengthens the control result: phi-only remains ahead on pairwise ranking, top-1 retrieval, and margin to the best negative. Therefore the current evidence supports `cf_1p0` as the main joint-flow candidate, but not the claim that joint-flow is already a stronger critic than a well-supervised phi-only Transformer.

## Interpretation

MVP1 V1 is the useful negative result: naive joint flow does not automatically become a critic.

MVP1 V2 shows the mechanism: `phi` trajectory, multi-step scoring, critic-flow auxiliary loss, and stronger CF supervision substantially improve action sensitivity.

MVP1.6 `cf_1p0` is the current best joint-flow critic. It improves over V2 by `+0.1054` coarse ranking, `+0.0920` all-negative ranking, and `+0.3199` coarse top-1. The cost is modest calibration degradation: MAE moves from `0.0158` to `0.0187`.

The phi-only strong baseline is an important new control. It uses the same prompt, observation-history, proprioception, action chunk, `phi_tokens=8`, and CF supervision as `cf_1p0`, but removes future-observation and action flow modeling. On current synthetic coarse negatives it reaches higher coarse ranking and coarse top-1 than joint-flow `cf_1p0`, but with substantially worse calibration stability because seed 44 has MAE `0.0415`.

This means current synthetic CF metrics do not prove that joint flow is a stronger critic than a well-supervised phi-only Transformer. The method claim should be tightened: `cf_1p0` remains the main joint-flow candidate, but the next evidence must come from harder candidate sets, downstream reranking, and calibration-aware selection.

`cf1p0_phi_w20` did not solve the calibration/ranking tradeoff. The next calibration direction should be constrained checkpointing, post-hoc calibration, or two-term model selection rather than a simple global `phi_weight` increase.

## Figure Pointers

- `docs/figures/current/gm100_mvp1_6_validation/mvp1_6_main_metrics.png`
- `docs/figures/current/gm100_mvp1_6_validation/calibration_vs_coarse_ranking.png`

Older comparison plots are archived under `docs/archive/figures/`.
