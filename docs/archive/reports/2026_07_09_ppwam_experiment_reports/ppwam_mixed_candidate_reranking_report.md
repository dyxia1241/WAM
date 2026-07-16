# PP-WAM Mixed Candidate Reranking Report

Date: 2026-07-09

## Summary

This experiment moves PP-WAM beyond self-sampled imagined futures. It evaluates
whether PP-WAM can score a realistic mixed candidate set:

```text
logged action
semantic action-bank negatives
same-task same-stage nearest action as a base-policy proxy
smooth action perturbations
WAM-sampled action candidates
```

This is a stronger paper diagnostic than pure synthetic counterfactual ranking
because the candidate pool contains both clearly wrong semantic distractors and
plausible alternatives.

## Candidate Set

Each anchor has 12 candidates:

```text
logged
same_task_phase_wrong
same_task_far_progress
cross_task
nearest_obs_wrong_action
same_task_stage_nearest
smooth_perturb_0.10
smooth_perturb_0.25
wam_sample_0
wam_sample_1
wam_sample_2
wam_sample_3
```

Only the four action-bank distractors are treated as strict semantic negatives:

```text
same_task_phase_wrong
same_task_far_progress
cross_task
nearest_obs_wrong_action
```

The same-task same-stage nearest action is a base-policy proxy, not a guaranteed
negative. Smooth perturbations and WAM samples are also not treated as strict
negatives because they may sometimes be reasonable alternatives.

## Runs

Executed on the 5060:

```text
num_anchors = 3072
num_wam_samples = 4
rescore_batch_size = 512
```

Output paths:

```text
outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/candidate_rerank_test_mixed_n4_a3072
outputs/rh20t_phi_only_cf1p0/seed_42/mvp1_joint_flow/candidate_rerank_test_mixed_n4_a3072
outputs/reassemble_cf1p0/seed_42/mvp1_joint_flow/candidate_rerank_test_mixed_n4_a3072
outputs/reassemble_phi_only_cf1p0/seed_42/mvp1_joint_flow/candidate_rerank_test_mixed_n4_a3072
```

For phi-only runs, the WAM samples and proposal potential come from the matched
joint-flow checkpoint. This keeps the candidate pool comparable. The pure
phi-only scorer result is the `model_phi` metric.

## Main Results

Strict semantic-negative retrieval:

| source | scorer | model phi pairwise | model phi logged top-1 | margin to best neg |
| --- | --- | ---: | ---: | ---: |
| RH20T | joint-flow | 0.9128 | 0.7233 | 0.0494 |
| RH20T | phi-only | 0.5718 | 0.2262 | -0.0351 |
| REASSEMBLE | joint-flow | 0.9433 | 0.8317 | 0.0581 |
| REASSEMBLE | phi-only | 0.9934 | 0.9730 | 0.7359 |

Full mixed-pool selection:

| source | scorer | selector | selected model phi | selected - logged model phi | selected logged rate | selected WAM rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| RH20T | joint-flow | max model phi | 0.1768 | 0.0078 | 0.0749 | 0.5039 |
| RH20T | joint-flow | calibrated smooth | 0.1745 | 0.0054 | 0.1953 | 0.1104 |
| RH20T | phi-only | max model phi | 0.1815 | 0.0478 | 0.0104 | 0.2109 |
| RH20T | phi-only | calibrated smooth | 0.1556 | 0.0220 | 0.0485 | 0.4245 |
| REASSEMBLE | joint-flow | max model phi | 0.1413 | 0.0232 | 0.0433 | 0.7835 |
| REASSEMBLE | joint-flow | calibrated smooth | 0.1268 | 0.0087 | 0.2910 | 0.1185 |
| REASSEMBLE | phi-only | max model phi | 0.8896 | 0.0571 | 0.3239 | 0.4206 |
| REASSEMBLE | phi-only | calibrated smooth | 0.6191 | -0.2134 | 0.1901 | 0.4209 |

## Interpretation

RH20T is a strong positive result for joint-flow:

```text
joint-flow model_phi strict pairwise = 0.9128
phi-only model_phi strict pairwise = 0.5718
```

This is the first realistic-candidate result where joint future
action/observation/potential modeling clearly outperforms the matched phi-only
critic on the main scoring metric.

Calibrated smooth selection also changes behavior in the desired direction. It
reduces blind WAM-sample selection and increases logged / same-task-stage
selection:

```text
RH20T joint-flow max_model_phi WAM rate = 0.5039
RH20T joint-flow calibrated_smooth WAM rate = 0.1104
RH20T joint-flow calibrated_smooth logged rate = 0.1953
```

REASSEMBLE remains a calibration stress case. Phi-only has excellent strict
ranking, but its score scale is extreme:

```text
REASSEMBLE phi-only max_model_phi selected model phi = 0.8896
REASSEMBLE joint-flow max_model_phi selected model phi = 0.1413
```

This is consistent with the earlier REASSEMBLE phi-only result: ranking can look
very strong while DeltaPhi calibration is unusable. REASSEMBLE should therefore
be used to argue that PP-WAM evaluation needs calibration metrics, not ranking
alone.

## Paper Takeaway

The paper can now claim a stronger intermediate result:

```text
On RH20T mixed candidate reranking, joint PP-WAM substantially outperforms the
matched phi-only critic in identifying logged actions over semantic negatives.
```

But the paper should not yet claim real policy improvement:

```text
This is offline candidate reranking, not closed-loop execution.
```

The next necessary evidence is ARX-SubSuccess better/worse successful segment
ranking, where candidates differ in process quality rather than only semantic
wrongness.
