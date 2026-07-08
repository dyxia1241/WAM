# PP-WAM Three-Source Future/Action Modeling Comparison

Date: 2026-07-08

## Scope

This report isolates the future-observation and future-action modeling metrics
from the seed-42 three-source comparison:

| model | config |
| --- | --- |
| joint-flow | `configs/multisource/joint_flow_3src_equal_cf1p0.yaml` |
| phi-only | `configs/multisource/phi_only_3src_equal_cf1p0.yaml` |

Both models use the same data, prompt features, observation history,
proprioception, action chunks, phi targets, counterfactual negatives, and
checkpoint rule. The difference is that joint-flow trains future observation,
future action, phi, critic-flow, and counterfactual losses, while phi-only trains
only the phi/counterfactual critic path and sets:

```text
obs_weight = 0.0
action_weight = 0.0
critic_flow_weight = 0.0
action_condition_prob = 1.0
```

Therefore future/action metrics should be read as:

```text
Does the joint-flow model learn useful auxiliary world-action structure?
```

not as:

```text
Does joint-flow beat a phi-only model on a task phi-only was designed to solve?
```

## Metrics

Relevant metrics:

| metric | meaning |
| --- | --- |
| `obs_flow_mse` | velocity MSE for future observation latent flow |
| `action_flow_mse` | velocity MSE for future action flow |
| `future_obs_y0_mse` | denoised future observation latent reconstruction MSE |
| `action_y0_mse` | denoised future action reconstruction MSE |
| `phi_flow_mse` | velocity MSE for primitive-local potential tokens |

Lower is better for all metrics in this report.

## Validation Results

| metric | joint-flow | phi-only | joint - phi | relative reduction vs phi-only |
| --- | ---: | ---: | ---: | ---: |
| `obs_flow_mse` | 1.044827 | 1.650341 | -0.605514 | 36.69% |
| `action_flow_mse` | 0.396360 | 2.049308 | -1.652948 | 80.66% |
| `future_obs_y0_mse` | 0.327219 | 0.548508 | -0.221289 | 40.34% |
| `action_y0_mse` | 0.074584 | 0.681668 | -0.607083 | 89.06% |
| `phi_flow_mse` | 0.036222 | 0.042977 | -0.006755 | 15.72% |

## Test Results

| metric | joint-flow | phi-only | joint - phi | relative reduction vs phi-only |
| --- | ---: | ---: | ---: | ---: |
| `obs_flow_mse` | 1.046300 | 1.650472 | -0.604172 | 36.61% |
| `action_flow_mse` | 0.464454 | 1.966498 | -1.502044 | 76.38% |
| `future_obs_y0_mse` | 0.328719 | 0.552404 | -0.223685 | 40.49% |
| `action_y0_mse` | 0.095637 | 0.654869 | -0.559232 | 85.40% |
| `phi_flow_mse` | 0.034487 | 0.039612 | -0.005125 | 12.94% |

## Coupled Phi / Ranking Context

The future/action advantage does not translate into better critic ranking in
this run:

| test metric | joint-flow | phi-only | joint - phi |
| --- | ---: | ---: | ---: |
| DeltaPhi MAE | 0.056219 | 0.058691 | -0.002471 |
| DeltaPhi RMSE | 0.107403 | 0.111219 | -0.003815 |
| coarse ranking | 0.856556 | 0.906200 | -0.049644 |
| all-negative ranking | 0.724360 | 0.803813 | -0.079453 |

Interpretation:

- Joint-flow is better on DeltaPhi calibration and on future/action modeling.
- Phi-only is better on synthetic counterfactual ranking and margin.
- The extra world-action modeling has not yet become a stronger critic score.

## What The Metrics Mean

The strongest positive signal for joint-flow is action reconstruction:

```text
test action_y0_mse reduction = 85.40%
test action_flow_mse reduction = 76.38%
```

This indicates that the typed joint-flow backbone learns future action dynamics
from the same conditioning context. It is not merely learning phi tokens while
ignoring action flow.

The future observation signal is also real:

```text
test future_obs_y0_mse reduction = 40.49%
test obs_flow_mse reduction = 36.61%
```

This supports the world-action-potential formulation: the model can represent
future latent consequences in the same denoising process as future actions and
process potential.

The phi-flow metric improves more modestly:

```text
test phi_flow_mse reduction = 12.94%
```

That aligns with the aggregate MAE/RMSE result: joint-flow is a little better
calibrated, but this calibration advantage is not enough to beat phi-only on
ranking.

## Paper Implication

These metrics are useful for the paper, but only if framed correctly.

Do not claim:

```text
Future/action modeling makes joint-flow a better critic on current synthetic
counterfactual ranking.
```

The current data says the opposite: phi-only is the stronger critic.

Do claim:

```text
Joint-flow learns future action and future observation latent structure that
phi-only intentionally does not model, while retaining slightly better
DeltaPhi calibration.
```

The paper value of joint-flow must therefore come from using that structure:

1. consistency-aware candidate selection;
2. semantic/base-policy candidate reranking;
3. action-unknown predictor mode;
4. ARX suboptimal-process pairs where future consequences matter.

## Next Diagnostic

The next useful diagnostic is not a generic ablation. It is a selection rule
that combines:

```text
score(candidate) = DeltaPhi score
                 + future-latent consistency term
                 + action smoothness / plausibility term
```

Then evaluate whether this combined score beats phi-only on harder candidate
sets. If it does not, joint-flow's future/action advantage remains a modeling
property rather than an action-selection contribution.
