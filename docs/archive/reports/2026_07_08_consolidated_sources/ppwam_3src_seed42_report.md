# PP-WAM Three-Source Seed-42 Consolidated Report

Date: 2026-07-08

## 0. What Was Reviewed

This report consolidates the current active reports:

- `docs/reports/ppwam_paper_master_plan.md`
- `docs/reports/gm100_ppwam_experiment_summary.md`
- `docs/reports/checkpoint_registry.md`
- `docs/reports/ppwam_roadmap.md`
- `docs/reports/ppwam_next_round_3src_plan.md`
- the seed-42 three-source joint-flow / phi-only outputs on the 5060

The shared conclusion across the older reports was already conservative:

```text
MVP1.6 joint-flow is viable, but the strong phi-only critic is stronger on
GM100 synthetic counterfactual ranking and first-pass hard reranking.
```

This three-source round tests whether that conclusion changes when the same
matched comparison is trained and evaluated across GM100, RH20T, and
REASSEMBLE.

## 1. Question

The narrow scientific question for this round was:

```text
Does joint future-observation/action/potential flow provide useful evidence
beyond a strong phi-only critic when trained and evaluated across three sources?
```

The answer is:

```text
For critic-style synthetic counterfactual ranking: no.
For calibration and world-action modeling signals: yes, but not yet enough
to justify claiming joint-flow is the better action selector.
```

## 2. Runs

Both runs used seed 42 and checkpoint selection by
`val/coarse_action_cf_ranking_acc`.

| model | config | output |
| --- | --- | --- |
| 3-source joint-flow `cf1p0` | `configs/multisource/joint_flow_3src_equal_cf1p0.yaml` | `outputs/ppwam_3src_equal_cf1p0/seed_42/mvp1_joint_flow` |
| 3-source phi-only `cf1p0` | `configs/multisource/phi_only_3src_equal_cf1p0.yaml` | `outputs/ppwam_3src_equal_phi_only_cf1p0/seed_42/mvp1_joint_flow` |

The remote execution machine was the 5060:

```text
repo: /data/projects/WAM
commit: 1e3ee94 Improve multisource prompt generation
```

Both runs completed successfully and wrote:

```text
best.pt
metrics.json
history.jsonl
manifest.json
eval_test/metrics.json
eval_test/manifest.json
experiment_report.md
figures/
```

## 3. Data Audit

Three-source prepared data:

```text
data/prepared/ppwam_3src_equal_signal_v1
```

Prompt features:

```text
data/prompts/ppwam_3src_equal_siglip/prompt_features.npz
data/prompts/ppwam_3src_equal_siglip/prompt_table.jsonl
```

Audit:

| item | value |
| --- | ---: |
| total windows | 75000 |
| train windows | 20000/source |
| val windows | 2500/source |
| test windows | 2500/source |
| canonical action dim | 14 |
| canonical proprio dim | 14 |
| canonical cameras | 3 |
| prompt features | 222 x 768 |

Source ids:

```text
gm100: 0
rh20t: 1
reassemble: 2
```

The prompt table ids matched the prompt feature ids before training.

## 4. Aggregate Results

### Validation

| model | DeltaPhi MAE | DeltaPhi RMSE | coarse ranking | all-neg ranking | all-neg margin | future obs y0 MSE | action y0 MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint-flow | 0.057294 | 0.105559 | 0.860689 | 0.725467 | 0.012317 | 0.327219 | 0.074584 |
| phi-only | 0.060256 | 0.109712 | 0.909400 | 0.795093 | 0.048828 | 0.548508 | 0.681668 |
| joint - phi | -0.002962 | -0.004153 | -0.048711 | -0.069626 | -0.036511 | -0.221289 | -0.607084 |

### Test

| model | DeltaPhi MAE | DeltaPhi RMSE | coarse ranking | all-neg ranking | all-neg margin | future obs y0 MSE | action y0 MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint-flow | 0.056219 | 0.107403 | 0.856556 | 0.724360 | 0.011644 | 0.328719 | 0.095637 |
| phi-only | 0.058691 | 0.111219 | 0.906200 | 0.803813 | 0.048634 | 0.552404 | 0.654869 |
| joint - phi | -0.002472 | -0.003816 | -0.049644 | -0.079453 | -0.036990 | -0.223685 | -0.559232 |

Interpretation:

- Phi-only is clearly better as a synthetic counterfactual critic.
- Joint-flow is slightly better calibrated on DeltaPhi MAE/RMSE.
- Joint-flow has much better future-observation/action reconstruction losses,
  but those losses are expected to favor joint-flow because phi-only is not
  trained to model future obs/action. These are world-action modeling signals,
  not direct evidence that joint-flow is a stronger critic.

## 5. Source-Stratified Test Results

### DeltaPhi Calibration

| source | joint-flow MAE | phi-only MAE | joint - phi |
| --- | ---: | ---: | ---: |
| GM100 | 0.016817 | 0.015685 | 0.001132 |
| RH20T | 0.115826 | 0.120378 | -0.004552 |
| REASSEMBLE | 0.036014 | 0.040008 | -0.003994 |

Joint-flow has better MAE on RH20T and REASSEMBLE, but not GM100.

### Coarse Counterfactual Ranking

| source | joint-flow | phi-only | joint - phi |
| --- | ---: | ---: | ---: |
| GM100 | 0.871400 | 0.906867 | -0.035467 |
| RH20T | 0.875067 | 0.906933 | -0.031867 |
| REASSEMBLE | 0.823200 | 0.904800 | -0.081600 |

Phi-only wins coarse ranking on all three sources.

### All-Negative Tie-Aware Ranking

| source | joint-flow | phi-only | joint - phi |
| --- | ---: | ---: | ---: |
| GM100 | 0.725640 | 0.770280 | -0.044640 |
| RH20T | 0.741120 | 0.827440 | -0.086320 |
| REASSEMBLE | 0.706320 | 0.813720 | -0.107400 |

Phi-only also wins all-negative ranking on all three sources. The largest
ranking gap is on REASSEMBLE.

## 6. Gate Decision

The strict three-source gate from `ppwam_next_round_3src_plan.md` says Outcome A
requires:

```text
3src joint-flow >= 3src phi-only on overall coarse ranking
3src joint-flow >= 3src phi-only on at least 2 of 3 source-specific coarse metrics
```

This run does not satisfy Outcome A.

The best label is:

```text
Outcome B, but only through calibration/world-action signals.
```

Reason:

- Phi-only wins overall ranking.
- Phi-only wins source-specific coarse ranking on every source.
- Joint-flow wins overall MAE/RMSE and wins MAE on RH20T and REASSEMBLE.
- Joint-flow models future obs/action substantially better, but that must be
  turned into a usable selection signal before it supports the paper claim.

If we judge only critic/reranking metrics, the result behaves like Outcome C.
If we include calibration and future/action modeling, it is Outcome B.

## 7. Relation To Earlier GM100 Evidence

The current result agrees with the GM100 report and checkpoint registry:

| setting | joint-flow conclusion | phi-only conclusion |
| --- | --- | --- |
| GM100 synthetic CF | viable, but lower ranking than phi-only | strongest current critic |
| GM100 hard reranking | lower pairwise/top-1 than phi-only | stronger logged-action retrieval |
| 3-source synthetic CF | lower ranking than phi-only on every source | strongest current three-source critic |

The new evidence is not that joint-flow suddenly becomes a better critic under
multi-source training. It does not.

The new evidence is that the three-source pipeline works and gives source-
stratified diagnostics, and that joint-flow retains calibration/world-action
signals even while losing critic ranking.

## 8. Meaning For The Paper Mainline

This round narrows the paper story.

The paper should not claim:

```text
PP-WAM joint-flow is already a stronger critic than a matched phi-only model.
```

The paper can still claim, with appropriate evidence:

```text
PP-WAM is a joint world-action-potential formulation whose extra modeled
future obs/action variables may be useful for predictor mode, consistency-aware
selection, semantic/base-policy candidate reranking, or real suboptimal-process
data.
```

The strong phi-only result is not a problem to hide. It is now one of the most
important controls in the paper:

```text
Any joint-flow advantage must survive a strong action-conditioned process
potential critic baseline.
```

For ICRA, this pushes the story away from "we beat the critic on synthetic
negatives" and toward:

```text
Can process-potential modeling improve action selection in realistic
suboptimal-yet-successful robot execution?
```

That aligns with the ARX-SubSuccess plan: the dataset and downstream evaluation
must expose process-quality distinctions that simple synthetic negatives do not.

## 9. Immediate Next Plan

Do not run ablations yet.

The next required experiments are source controls:

```bash
cd /data/projects/WAM
.conda/wam/bin/python -m ppwam.joint_flow --config configs/rh20t/joint_flow_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/rh20t/phi_only_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/reassemble/joint_flow_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/reassemble/phi_only_cf1p0.yaml
```

The next report should answer:

1. Does three-source training help or hurt RH20T relative to RH20T-only?
2. Does three-source training help or hurt REASSEMBLE relative to REASSEMBLE-only?
3. Is joint-flow's MAE advantage on RH20T/REASSEMBLE real, or just a
   multi-source artifact?
4. Is REASSEMBLE's larger ranking gap caused by data normalization, action
   padding, prompt quality, camera padding, or source difficulty?

Run seeds 43 and 44 only after the source-control table is readable. Good
reasons to expand are source-specific transfer effects or a result that
contradicts the current GM100 prior. This seed-42 result alone is not a good
reason to scale the same three-source pair blindly.

## 10. Technical Follow-Ups

After source controls, prioritize diagnostics that can actually justify the
joint-flow complexity:

1. Semantic/base-policy candidate reranking, not only synthetic zero/reverse/
   shuffle/scaled negatives.
2. Future-latent consistency as an action-selection term:
   score candidate actions by both DeltaPhi and predicted future-observation
   consistency.
3. Predictor/action-unknown mode:
   test whether generated action/future latent trajectories have utility beyond
   critic scoring.
4. ARX-SubSuccess pilot ingestion:
   build optimal-vs-suboptimal pairs inside successful trajectories.
5. Only then run ablations or Base-scale configs.

## 11. Bottom Line

Current experimental conclusion:

```text
Phi-only is the stronger critic. Joint-flow is not yet justified as a better
reranker, but it remains justified as a world-action-potential model candidate
because it carries calibration and future/action modeling signals that the
phi-only critic intentionally lacks.
```

Paper implication:

```text
The main paper must be honest about phi-only. PP-WAM's distinctive value has to
come from dual-mode inference, future-latent/action consistency, semantic or
base-policy candidate selection, and real suboptimal-process data, not from
simple synthetic counterfactual ranking.
```
