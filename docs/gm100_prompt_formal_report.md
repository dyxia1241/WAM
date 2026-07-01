# GM-100 Prompt-Conditioned Primitive Potential Critic Report

- Git commit: `3efb5546181e9fc0293f708bd5c1601f8b61a948`
- Machine/GPU: `NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, 595.58.03`
- Python: `3.10.20`
- Dataset: `data/prepared/gm100_50x5_light_signal_v1`, DINOv2 visual features, history=4, horizon=8, stride=2
- Prompt artifacts: `data/prompts/gm100_siglip_base`, frozen `google/siglip-base-patch16-224`, 110 prompts, feature dim 768
- Seeds: 42, 43, 44
- Checkpoint selection: `val/delta_phi_mae`

## Model

Token sequence: `[CLS] + obs_tokens + proprio_token + prompt_token (+ action_tokens)`. The prompt token is projected from frozen SigLIP text features and also FiLM-modulates action tokens. `stage_id` and numeric `task_id` are not passed to prompt experiments.

Architecture figure is generated at `outputs/gm100_prompt_formal/figures/prompt_critic_architecture.png` on the 5060 artifact tree.

## Architecture And Config Details

### Inputs

| Input | Shape | Source | Train-time role |
| --- | --- | --- | --- |
| `obs_features` | `[B, 4, C, 768]` | frozen DINOv2 ViT-B/14 feature stores | camera-mean pooled, then projected to obs tokens |
| `proprio` | `[B, 14]` | normalized GM-100 joint/eef state | projected to one proprio token |
| `prompt_features` | `[B, 768]` | frozen `google/siglip-base-patch16-224` text encoder | projected to one prompt token; also drives FiLM |
| `action_chunk` | `[B, 8, 14]` | normalized future action chunk | projected to action tokens for action-conditioned experiments |

The raw string `task_id` is used only to look up `prompt_features` from `data/prompts/gm100_siglip_base/prompt_features.npz`. The model does not receive numeric `task_id` or `stage_id` in prompt experiments.

### Prompt Critic Module

The trained critic is `PromptFiLMTransformerCritic`:

```text
obs_features --mean camera--> Linear(768, 128)              -> obs tokens
proprio ---------------------> Linear(14, 128)              -> proprio token
prompt_features -------------> LayerNorm(768) + MLP         -> prompt token
action_chunk ----------------> Linear(14, 128)              -> action tokens
prompt token ----------------> MLP -> gamma,beta            -> FiLM action tokens
[CLS] + obs + proprio + prompt (+ action) -> TransformerEncoder -> CLS MLP head -> DeltaPhi logit
```

Core dimensions:

```yaml
model:
  name: prompt_film_transformer
  hidden_dim: 128
  transformer_layers: 1
  transformer_heads: 4
  dropout: 0.1
```

The fusion Transformer is a small trainable critic module, not a frozen backbone. It uses one `nn.TransformerEncoderLayer` with self-attention over the fused token sequence:

```text
d_model = 128
nhead = 4
feedforward dim = 512
activation = GELU
norm_first = true
```

Checkpoint inspection confirms trained attention parameters such as:

```text
fusion.layers.0.self_attn.in_proj_weight  (384, 128)
fusion.layers.0.self_attn.out_proj.weight (128, 128)
```

### Frozen Versus Trainable Components

Frozen / precomputed:

- DINOv2 visual encoder: features are read from `data/features/gm100_50x5_light_dinov2_vitb14_224`.
- SigLIP text encoder: prompt embeddings are read from `data/prompts/gm100_siglip_base/prompt_features.npz`.

Trainable in this experiment:

- obs/proprio/prompt/action projection layers;
- prompt FiLM MLP;
- critic fusion Transformer self-attention and feed-forward layers;
- CLS token and final DeltaPhi head.

### Experiment Variants

| Experiment | Token sequence | Training loss | Purpose |
| --- | --- | --- | --- |
| `obs_prompt` | `[CLS] + obs + proprio + prompt` | DeltaPhi SmoothL1 only | action-free prompt-conditioned progress baseline |
| `obs_action_prompt` | `[CLS] + obs + proprio + prompt + action` | DeltaPhi SmoothL1 only | test whether action helps calibrated DeltaPhi regression without ranking supervision |
| `obs_action_prompt_cf_multi` | same as `obs_action_prompt` | DeltaPhi SmoothL1 + `0.1 * L_cf` | test whether multi-negative counterfactual loss creates action sensitivity |

The counterfactual ranking loss is:

```text
L_cf = -logsigmoid(pred_delta_phi_positive - pred_delta_phi_negative - 0.03)
```

Training negative types for `obs_action_prompt_cf_multi`:

```text
zero, reverse, shuffle, wrong_arm, scaled_0.25, scaled_1.75
```

### Run Config

```yaml
data:
  windows_dir: data/prepared/gm100_50x5_light_signal_v1
  episodes_dir: data/episodes/gm100_50x5_light
  features_dir: data/features/gm100_50x5_light_dinov2_vitb14_224
  prompt_features: data/prompts/gm100_siglip_base/prompt_features.npz
  prompt_feature_dim: 768
  norm_stats: data/prepared/gm100_50x5_light_signal_v1/norm_stats.json
  history: 4
  horizon: 8
  stride: 2
  batch_size: 128
  action_dim: 14
  proprio_dim: 14

features:
  encoder: vit_base_patch14_dinov2.lvd142m
  feature_dim: 768

prompts:
  encoder: google/siglip-base-patch16-224
  prompt_table: data/prompts/gm100_siglip_base/prompt_table.jsonl

optim:
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-4
  grad_clip_norm: 1.0

train:
  max_epochs: 10
  save_best_by: val/delta_phi_mae
```

## Prompt Examples

The prompt text is generated per raw string `task_id`. The `task_id` is not included in the model input text; it is only used offline to look up the frozen prompt feature.

Example `task_00001`:

```text
You are evaluating a short robot manipulation segment.

High-level task goal:
Use the gripper to strike the small ball into the tabletop goal.

Canonical primitive chain for this task:
adjust tabletop_goal -> contact ball

Question:
Given the current observation history, robot proprioception, and candidate future action chunk, estimate the progress increment of the current local primitive only. Do not estimate progress toward the whole task. The current primitive label is not provided; infer the active primitive from the observation and action.

Output target:
primitive-local DeltaPhi in [0, 1].
```

Example `task_00059` after fixing the missing metadata row:

```text
You are evaluating a short robot manipulation segment.

High-level task goal:
Insert the tool into the keyhole of the door lock and keep it in a fixed position to facilitate easy operation by personnel.

Canonical primitive chain for this task:
grasp tool -> move tool -> adjust keyhole -> insert tool -> hold tool

Question:
Given the current observation history, robot proprioception, and candidate future action chunk, estimate the progress increment of the current local primitive only. Do not estimate progress toward the whole task. The current primitive label is not provided; infer the active primitive from the observation and action.

Output target:
primitive-local DeltaPhi in [0, 1].
```

## Test Aggregate Metrics

| experiment | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin | zero | shuffle | wrong_arm | scaled_1.75 | reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `obs_prompt` | 0.0120±0.0006 | 0.0321±0.0003 | 0.5000±0.0000 | 0.0000±0.0000 | 0.5000±0.0000 | 0.5000±0.0000 | 0.5000±0.0000 | 0.5000±0.0000 | 0.5000±0.0000 |
| `obs_action_prompt` | 0.0132±0.0011 | 0.0346±0.0024 | 0.4792±0.0335 | 0.0005±0.0002 | 0.5456±0.0855 | 0.5005±0.0060 | 0.3135±0.0215 | 0.5317±0.0957 | 0.5047±0.0042 |
| `obs_action_prompt_cf_multi` | 0.0236±0.0074 | 0.0410±0.0144 | 0.7701±0.0440 | 0.0121±0.0063 | 0.9917±0.0145 | 0.5523±0.0507 | 0.9010±0.0629 | 0.8659±0.0891 | 0.5398±0.0512 |

## Main Findings

- `obs_prompt` gives the best calibrated DeltaPhi regression, but as expected it is action-insensitive: ranking and margins stay exactly at tie/random behavior.
- Adding action without counterfactual loss (`obs_action_prompt`) slightly worsens MAE and yields unstable action ranking: some negatives improve, but wrong-arm ranking is below random.
- Multi-counterfactual training (`obs_action_prompt_cf_multi`) clearly increases action sensitivity, especially for `zero`, `wrong_arm`, and `scaled_1.75`, but it trades off DeltaPhi calibration and remains weak for `shuffle`/`reverse` negatives.
- Stage replacement metrics are intentionally absent for prompt experiments because stage is not a model input.

## Figures

- `outputs/gm100_prompt_formal/figures/test_delta_phi_mae.png`
- `outputs/gm100_prompt_formal/figures/test_delta_phi_rmse.png`
- `outputs/gm100_prompt_formal/figures/test_all_negative_ranking.png`
- `outputs/gm100_prompt_formal/figures/test_all_negative_margin.png`
- `outputs/gm100_prompt_formal/figures/test_per_negative_ranking.png`

## File Layout

```text
outputs/gm100_prompt_formal/
  seed_42|seed_43|seed_44/
    obs_prompt|obs_action_prompt|obs_action_prompt_cf_multi/
      best.pt, metrics.json, manifest.json, eval_test/
  logs/train_seed_*.log
  figures/*.png|*.svg
  summary/*.csv|*.json|experiment_report.md
data/prompts/gm100_siglip_base/
  prompt_table.jsonl, prompt_features.npz, prompt_manifest.json
```

## Caveats

- The `ranking_acc` key in raw eval metrics is the zero-negative ranking from `evaluate_model`; the report table uses `all_negatives_tie_aware_ranking_acc` computed from all requested action sensitivity rows.
- The current counterfactual loss improves action sensitivity but over-optimizes easy negatives. Harder shuffle/reverse sensitivity remains near random, so this is a useful diagnostic rather than a clean final result.