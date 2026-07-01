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