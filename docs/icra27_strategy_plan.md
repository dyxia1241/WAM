# WAM ICRA27 Strategic Plan

## 1. Paper Direction

The internal MVP0 name should not be used as the paper-facing method name. MVP0 is the foundation experiment for a broader paper direction:

```text
Primitive-local progress in world-action models must be action-grounded and process-bounded.
```

The paper-facing line is:

```text
From visually plausible WAMs to process-bounded WAMs.
```

The current method direction is PP-WAM:

```text
Primitive-Potential World Action Model
```

MVP0 should be framed as:

```text
Action-Grounded Primitive Potential Critic
```

Its core equation is:

```text
(o_{\le t}, q_t, c, s_t, a_{t:t+H}) -> Delta phi^(s_t)
```

The MVP0 scientific question is:

```text
Given the same observation/proprioception/stage/task, does the demonstrated action receive higher primitive-local progress than counterfactual actions?
```

## 2. Current Takeaways

The current repo has a working MVP0 engineering loop:

- GM-100 raw subset import.
- Contact-anchored primitive-local labels.
- Prepared train/val/test windows split by episode.
- Frozen DINOv2 visual features.
- Joint action/proprio z-score normalization from train split only.
- Stage-FiLM Transformer critic.
- Counterfactual action evaluation.
- Plot and report generation.

The current result is only a smoke result, not a paper result:

- It trained for only 1 epoch.
- `obs_action_stage_cf` used only a `zero` negative during training.
- Easy negatives were separated, but harder negatives were unstable.
- Ranking loss with weight `0.5` damaged Delta-phi calibration.

The next experiment must produce evidence for the paper claim, not just verify the code path.

## 3. Evidence Chain

MVP0 supports the first paper experiment:

```text
Experiment 1: Is primitive progress action-grounded?
```

Required controls:

- `time_prior`: checks whether the linear potential label is solved by primitive time.
- `obs_stage`: checks whether video/stage-only progress is enough.
- `joint_action_stage`: checks whether joint-state/action alone carries process signal.
- `obs_action_stage`: checks action conditioning without counterfactual loss.
- `obs_action_stage_cf_zero`: checks the previous zero-negative objective.
- `obs_action_stage_cf_multi`: checks the formal multi-negative objective.

The desired result pattern is:

```text
obs_action_stage_cf_multi > obs_action_stage > obs_stage > time_prior
```

The most important paper plot is not only MAE. It is the distribution of:

```text
pred_delta_phi(a+) - pred_delta_phi(a-)
```

split by negative type.

## 4. Immediate 5060 Formal MVP0 Plan

Use the 5060 laptop as the formal pilot machine for this stage because it already has the GM-100 50x5 light subset, prepared windows, DINOv2 features, and normalization stats.

Before running the formal suite, upgrade training from zero-only counterfactuals to multi-negative counterfactuals:

```text
zero
reverse
shuffle
wrong_arm
scaled_0.25
scaled_1.75
```

Formal run defaults:

```text
epochs = 10
batch_size = 128
cf_weight = 0.1
margin = 0.03
seed = 42
```

If runtime is acceptable, add:

```text
seed = 43
seed = 44
```

Formal outputs:

- `metrics.json` and `manifest.json` for each run.
- `predictions.jsonl`.
- `action_sensitivity.csv`.
- `stage_sensitivity.csv` as diagnostic only.
- `summary.csv` and `summary.json`.
- Margin histograms and calibration plots.

## 5. Success Criteria

Minimum success for MVP0:

- Full action-conditioned models beat `obs_stage` on action ranking.
- Multi-negative training improves harder negatives over zero-only training:
  - `shuffle`
  - `reverse`
  - `wrong_arm`
  - `scaled_1.75`
- Delta-phi MAE remains calibrated enough to avoid the degenerate high-margin/high-MAE behavior seen in the smoke run.

Do not treat stage replacement metrics as a main result yet. Current labels are contact-anchored `move` intervals, not reliable semantic approach/grasp/place/release stages.

## 6. Decision Gate After MVP0

If joint-action MVP0 succeeds:

```text
Proceed to MVP1: latent world-action predictor with Delta-phi head.
```

If joint-action MVP0 fails:

```text
Do not start MVP1 yet.
First add EEF/FK action representation and rerun MVP0.
```

MVP1 target:

```text
(o_{\le t}, q_t, c, s_t) -> a_{t:t+H}, z_{t+1:t+H}, Delta phi
```

MVP2 target:

```text
Evaluator mode: (o, q, c, s, a) -> Delta phi
Generator mode: (o, q, c, s, masked action) -> action + future latent + Delta phi
```

Only after MVP0 and MVP1 are validated should the project spend H200 time on larger multi-seed or harder negative experiments.

## 7. Compute Policy

Current stage:

- WSL: code edits, unit tests, docs.
- 5060: formal MVP0 pilot experiments.
- 4090: later data authority and artifact archival.
- H200: later offline multi-seed / larger model experiments.

Large data, features, outputs, and checkpoints stay out of git.
