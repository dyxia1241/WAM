# PP-WAM Current Consolidated Report

Date: 2026-07-08

## 0. Executive Conclusion

The current PP-WAM evidence supports a conservative paper direction:

```text
Primitive-local process potential must be action-conditioned and trained with
counterfactual supervision. Joint-flow is a viable world-action-potential
formulation, but it is not yet a stronger critic than a matched phi-only model.
```

The strongest current critic/reranker is the phi-only strong baseline. The
joint-flow model remains scientifically useful because it models future
observation latents and future actions jointly with process potential. That
extra structure has to be used through predictor mode, future-latent/action
consistency, semantic/base-policy candidate reranking, or real
suboptimal-process data. It should not be sold as already winning on simple
synthetic counterfactual ranking.

## 1. Paper Mainline

The paper should be framed around:

```text
Learning action-conditioned primitive-local process potential from successful
but process-variable robot trajectories, then using that potential for action
selection and process-quality improvement.
```

Target venue remains ICRA / ICLR 2027, with ICRA as the safer main route because
the strongest story is robot execution quality, candidate reranking, and an
ARX-SubSuccess dataset rather than only a model benchmark.

The final contribution stack should be:

1. PP-WAM: typed-token joint flow over future observation latents, future action
   chunks, and primitive-local process potential.
2. Dual-mode inference: action-known critic/reranker mode and action-unknown
   predictor/policy mode.
3. ARX-SubSuccess: real dual-arm successful trajectories with intentionally
   suboptimal execution processes.
4. Action-selection utility: offline candidate reranking and real robot evidence
   that process potential reduces suboptimal behavior or improves efficiency.

### Dual-Mode Definition

The current implementation should be described as two inference modes sharing
one typed-token denoiser:

```text
Predictor mode:
  context -> future observation latent, future action chunk, process potential
  action tokens are noisy/generated variables

Critic mode:
  context + clamped candidate action chunk -> future observation latent,
                                             process potential
  action tokens are known conditions, not generated variables
```

This means the current `action_is_condition=True` training behavior is
semantically correct: the action chunk is copied from the clean target,
represented with a clamped token state, and the action loss is zero. In critic
mode the action head output is ignored; optimizing it would blur the distinction
between "given candidate action" and "generated future action."

The next evaluation metrics should therefore distinguish:

```text
predictor_* metrics: action is noisy/generated
critic_* metrics: positive/executed action is clamped and only future
                  observation + potential consequences are evaluated
```

## 2. Strategic Differentiation

PP-WAM should not be positioned as a smaller generic video-action world model.
The safer and sharper positioning is:

```text
tau0-WM-style systems: large-scale video-action-task-progress WAMs
PP-WAM: primitive-local, suboptimal-sensitive, action-clamped
        process-potential WAM
```

The risky generic claim to avoid is:

```text
We jointly model future observation latents, action chunks, and task progress.
```

That phrasing overlaps too directly with large WAM work that already combines
future visual/action prediction, candidate action evaluation, and dense
task-progress scoring. PP-WAM's distinct claim should instead be:

```text
PP-WAM evaluates whether a candidate action chunk advances the currently active
manipulation primitive by predicting its action-clamped future consequence and
primitive-local process potential.
```

Short version:

```text
PP-WAM turns WAM from generic future prediction into primitive-local action
consequence evaluation.
```

### Differentiation Axes

| axis | large video-action WAM route | PP-WAM route |
| --- | --- | --- |
| main goal | unified future prediction, policy learning, action evaluation | primitive-local process-quality action evaluation |
| scale | large shared video/action backbone | compact diagnostic typed-token DiT, later scalable |
| progress signal | dense task or subtask progress | active-primitive-local potential trajectory |
| critic mode | candidate scoring through rollout/value heads | action-clamped consequence flow |
| data route | broad multi-source demonstrations and rollouts | suboptimal-yet-success dual-arm process segments |
| paper value | foundation WAM capability | process-sensitive action reranking interface |

This means the paper should not compete on:

```text
larger backbone
more general WAM
generic task-progress prediction
```

It should compete on:

```text
primitive-local potential
action-clamped consequence flow
counterfactual process supervision
suboptimal-yet-success action consequences
process-quality candidate reranking
```

### ARX-SubSuccess Role

ARX-SubSuccess should be framed as a process-quality dataset, not a scale
dataset:

```text
successful trajectories with local process-quality variation:
hesitation, overshoot, detour, regrasp, wrong-arm attempt, late coordination,
unnecessary contact, unstable contact, and overcorrection.
```

The useful training/evaluation unit is:

```text
context c
executed action chunk a
future observation consequence z_future
primitive-local potential phi
suboptimality tag or better/worse segment relation
```

This separates PP-WAM from generic success imitation: the question is not only
whether the final task succeeds, but whether the candidate action advances the
current primitive in a high-quality way.

### Required Stand-Out Experiments

The next evidence must move beyond simple synthetic counterfactual ranking.
Prioritize:

1. Primitive-local ambiguity test:
   the same action can be high or low potential depending on the active
   primitive.
2. Suboptimal-success ranking:
   direct/smooth/correct segments should rank above hesitation, detour,
   wrong-arm, regrasp, overshoot-and-correct, or unstable-contact segments,
   while both trajectories still end in success.
3. Action-clamped consequence consistency:
   given an executed action, predicted future latent consequences should match
   the observed consequence, with phi reflecting process quality.

Current interpretation:

```text
Synthetic ranking says phi-only is the stronger current critic.
Critic consequence metrics say joint-flow models action-clamped future
consequences better.
The paper must turn that consequence signal into primitive-local process-quality
selection evidence.
```

## 3. Current Evidence Stack

### MVP0

MVP0 established the motivation:

```text
primitive-local potential must be action-conditioned and needs counterfactual
supervision.
```

Representative MVP0 references:

| model | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin |
| --- | ---: | ---: | ---: | ---: |
| MVP0 `stage_action_cf` | 0.0223+/-0.0043 | 0.0360+/-0.0006 | 0.8196+/-0.0272 | 0.0162+/-0.0025 |
| MVP0 `prompt_cf_w10` | 0.0125+/-0.0006 | 0.0330+/-0.0003 | 0.7512+/-0.0215 | 0.0066+/-0.0001 |
| MVP0 `prompt_cf_w20` | 0.0121+/-0.0008 | 0.0321+/-0.0019 | 0.7243+/-0.0226 | 0.0056+/-0.0006 |

### GM100 Joint-Flow Progression

MVP1 moved the method to a lightweight DiT-style typed-token joint flow over:

```text
future observation latent + future action chunk + primitive-local potential trajectory
```

Current MVP1.6 settings:

```text
history=4, horizon=8, stride=2
hidden_dim=192, layers=3, heads=4
phi_tokens=8
denoise_steps=4, train_denoise_steps=2
obs_weight=1.0, action_weight=1.0, phi_weight=10.0
critic_flow_weight=1.0, counterfactual_weight=1.0
```

GM100 aggregate results:

| model | seeds | DeltaPhi MAE | DeltaPhi RMSE | coarse ranking | all-neg ranking | coarse top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MVP1 V1 | 3 | 0.0189+/-0.0027 | 0.0339+/-0.0019 | 0.6004+/-0.0813 | 0.5676+/-0.0538 | n/a |
| MVP1 V2 | 3 | 0.0158+/-0.0040 | 0.0316+/-0.0025 | 0.7816+/-0.0369 | 0.6881+/-0.0241 | 0.4102+/-0.0512 |
| MVP1.6 `cf_1p0` | 3 | 0.0187+/-0.0027 | 0.0335+/-0.0026 | 0.8870+/-0.0231 | 0.7801+/-0.0132 | 0.7301+/-0.0543 |
| phi-only `cf1p0` | 3 | 0.0256+/-0.0137 | 0.0366+/-0.0108 | 0.9084+/-0.0186 | 0.7911+/-0.0334 | 0.7790+/-0.0670 |

Interpretation:

- MVP1.6 `cf_1p0` is the best current joint-flow critic.
- Phi-only is stronger on synthetic coarse ranking and top-1.
- Joint-flow should not be claimed as the stronger critic on GM100.

### GM100 Hard Reranking

Hard reranking uses each logged test action as the positive candidate and four
action-bank distractors:

```text
same_task_phase_wrong
same_task_far_progress
cross_task
nearest_obs_wrong_action
```

Results:

| model | seeds | hard pairwise ranking | hard top-1 | margin to best neg |
| --- | ---: | ---: | ---: | ---: |
| MVP1.6 `cf_1p0` | 3 | 0.8336+/-0.0756 | 0.6391+/-0.1341 | 0.0104+/-0.0078 |
| phi-only `cf1p0` | 3 | 0.8795+/-0.0304 | 0.7248+/-0.0590 | 0.0222+/-0.0020 |

This reinforces the same conclusion: phi-only remains the stronger current
critic/reranker.

## 4. Three-Source Seed-42 Round

The newest round trained matched seed-42 models on an equal three-source dataset
spanning GM100, RH20T, and REASSEMBLE.

Data audit:

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

Runs:

| model | config | output |
| --- | --- | --- |
| 3-source joint-flow `cf1p0` | `configs/multisource/joint_flow_3src_equal_cf1p0.yaml` | `outputs/ppwam_3src_equal_cf1p0/seed_42/mvp1_joint_flow` |
| 3-source phi-only `cf1p0` | `configs/multisource/phi_only_3src_equal_cf1p0.yaml` | `outputs/ppwam_3src_equal_phi_only_cf1p0/seed_42/mvp1_joint_flow` |

Both ran on the 5060 at commit `1e3ee94` and completed successfully.

### Aggregate Metrics

Validation:

| model | DeltaPhi MAE | DeltaPhi RMSE | coarse ranking | all-neg ranking | all-neg margin | future obs y0 MSE | action y0 MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint-flow | 0.057294 | 0.105559 | 0.860689 | 0.725467 | 0.012317 | 0.327219 | 0.074584 |
| phi-only | 0.060256 | 0.109712 | 0.909400 | 0.795093 | 0.048828 | 0.548508 | 0.681668 |

Test:

| model | DeltaPhi MAE | DeltaPhi RMSE | coarse ranking | all-neg ranking | all-neg margin | future obs y0 MSE | action y0 MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint-flow | 0.056219 | 0.107403 | 0.856556 | 0.724147 | 0.011588 | 0.326976 | 0.095084 |
| phi-only | 0.058691 | 0.111219 | 0.906200 | 0.802987 | 0.048546 | 0.546581 | 0.656054 |

### Source-Stratified Test Metrics

Coarse ranking:

| source | joint-flow | phi-only | joint - phi |
| --- | ---: | ---: | ---: |
| GM100 | 0.871400 | 0.906867 | -0.035467 |
| RH20T | 0.875067 | 0.906933 | -0.031867 |
| REASSEMBLE | 0.823200 | 0.904800 | -0.081600 |

All-negative ranking:

| source | joint-flow | phi-only | joint - phi |
| --- | ---: | ---: | ---: |
| GM100 | 0.727320 | 0.771120 | -0.043800 |
| RH20T | 0.740080 | 0.826240 | -0.086160 |
| REASSEMBLE | 0.705040 | 0.811600 | -0.106560 |

DeltaPhi MAE:

| source | joint-flow | phi-only | joint - phi |
| --- | ---: | ---: | ---: |
| GM100 | 0.016817 | 0.015685 | 0.001132 |
| RH20T | 0.115826 | 0.120378 | -0.004552 |
| REASSEMBLE | 0.036014 | 0.040008 | -0.003994 |

Decision:

```text
Phi-only wins overall ranking and source-specific ranking on every source.
Joint-flow wins overall MAE/RMSE and source MAE on RH20T and REASSEMBLE.
```

If judged only by critic/reranking metrics, the result behaves like Outcome C.
If calibration and world-action modeling are included, it is Outcome B:
phi-only wins overall, but joint-flow retains source-specific and modeling
signals.

## 5. Future/Action Modeling Result

The future/action metrics isolate whether joint-flow learned the extra structure
that phi-only intentionally does not model.

Important interpretation update:

```text
future_obs_y0_mse and action_y0_mse are predictor-mode metrics from the
action-noisy flow path. They show that joint-flow learns future observation and
action structure, but they are not yet critic consequence metrics.
```

Test metrics:

| metric | joint-flow | phi-only | joint - phi | relative reduction vs phi-only |
| --- | ---: | ---: | ---: | ---: |
| `predictor_obs_flow_mse` | 1.046480 | 1.650599 | -0.604119 | 36.60% |
| `predictor_action_flow_mse` | 0.467582 | 1.973803 | -1.506222 | 76.31% |
| `predictor_future_obs_y0_mse` | 0.326976 | 0.546581 | -0.219605 | 40.18% |
| `predictor_action_y0_mse` | 0.095084 | 0.656054 | -0.560971 | 85.51% |
| `predictor_phi_flow_mse` | 0.035200 | 0.040033 | -0.004834 | 12.07% |

Interpretation:

- Joint-flow is much better at reconstructing future action and future
  observation latents.
- This confirms that the joint-flow backbone is learning non-trivial
  world-action structure.
- This still does not prove better action selection, because phi-only wins the
  ranking metrics.

These metrics are valuable only if the next evaluation uses them:

```text
candidate score = DeltaPhi score
                + future-latent consistency
                + action smoothness / plausibility
```

The immediate metric gap was the action-clamped critic consequence path:

```text
critic_future_obs_y0_mse
critic_obs_flow_mse
critic_phi_flow_mse
critic_delta_phi_mae
```

These have now been computed from existing checkpoints on positive/executed
actions. They do not require changing the model architecture or retraining,
because the training path already supports clamped action tokens and already
sets action loss to zero in that mode.

Critic consequence test metrics:

| metric | joint-flow | phi-only | joint - phi | relative reduction vs phi-only |
| --- | ---: | ---: | ---: | ---: |
| `critic_future_obs_y0_mse` | 0.334467 | 0.560828 | -0.226361 | 40.36% |
| `critic_obs_flow_mse` | 1.045216 | 1.650572 | -0.605356 | 36.68% |
| `critic_phi_flow_mse` | 0.032087 | 0.035134 | -0.003047 | 8.67% |
| `critic_delta_phi_mae` | 0.056219 | 0.058691 | -0.002471 | 4.21% |
| `critic_delta_phi_rmse` | 0.107403 | 0.111219 | -0.003815 | 3.43% |

Interpretation:

- In action-clamped critic mode, joint-flow predicts executed-action future
  observation consequences substantially better than phi-only.
- The same critic mode still loses synthetic counterfactual ranking to
  phi-only, so consequence modeling is a real signal but not yet a better
  action-selection rule.

## 6. Current Checkpoint Registry

Important checkpoint groups on the 5060:

| group | purpose |
| --- | --- |
| `outputs/gm100_mvp1_6_cf1p0/seed_*/mvp1_joint_flow/best.pt` | current GM100 joint-flow candidate |
| `outputs/gm100_phi_only_cf1p0/seed_*/mvp1_joint_flow/best.pt` | current GM100 strong phi-only baseline |
| `outputs/gm100_hard_rerank/cf_1p0/seed_*/hard_reranking_metrics.json` | hard reranking for joint-flow |
| `outputs/gm100_hard_rerank/phi_only_cf1p0/seed_*/hard_reranking_metrics.json` | hard reranking for phi-only |
| `outputs/ppwam_3src_equal_cf1p0/seed_42/mvp1_joint_flow/best.pt` | three-source joint-flow seed 42 |
| `outputs/ppwam_3src_equal_phi_only_cf1p0/seed_42/mvp1_joint_flow/best.pt` | three-source phi-only seed 42 |

Checkpoint files remain on the 5060 and are intentionally not tracked by git.

## 7. Paper Implication

The paper should not claim:

```text
PP-WAM joint-flow is already a stronger critic/reranker than a matched
phi-only model.
```

The paper can claim:

```text
PP-WAM joint-flow is a viable world-action-potential formulation. Its future
action and future observation modeling are real, but their value must be
demonstrated through selection or prediction tasks that actually use them.
```

The strong phi-only baseline is now central to the paper:

```text
Any claimed joint-flow advantage must survive a strong action-conditioned
process-potential critic baseline.
```

This shifts the paper away from simple synthetic counterfactual wins and toward
realistic process-quality evaluation:

```text
Can process-potential modeling improve action selection in realistic
suboptimal-yet-successful robot execution?
```

That question aligns with the ARX-SubSuccess dataset plan.

## 8. Current Roadmap

### Immediate Experiments

Do not run generic ablations yet.

Run source controls first:

```bash
cd /data/projects/WAM
.conda/wam/bin/python -m ppwam.joint_flow --config configs/rh20t/joint_flow_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/rh20t/phi_only_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/reassemble/joint_flow_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/reassemble/phi_only_cf1p0.yaml
```

These controls should answer:

1. Does three-source training help or hurt RH20T relative to RH20T-only?
2. Does three-source training help or hurt REASSEMBLE relative to
   REASSEMBLE-only?
3. Is joint-flow's MAE advantage on RH20T/REASSEMBLE real?
4. Is REASSEMBLE's larger ranking gap caused by source normalization, action
   padding, prompt quality, camera padding, or source difficulty?

### Next Diagnostics

After source controls, prioritize:

1. semantic/base-policy candidate reranking;
2. future-latent consistency scoring;
3. action-unknown predictor mode;
4. ARX-SubSuccess pilot ingestion;
5. Base-scale configs only after the above diagnostics show why scale matters.

### ARX-SubSuccess

The ARX dataset should focus on:

```text
successful trajectories with distinguishable process quality
```

Target first full version:

```text
8-12 tasks
300-500 successful trajectories
2-3 camera views
dual-arm action/proprio
primitive boundaries
suboptimality tags
```

Pilot:

```text
2-3 tasks
50-80 successful trajectories
validate sync, tags, and optimal-vs-suboptimal pair construction
```

## 9. Execution Rules

Formal training runs happen on the 5060:

```text
ssh: /mnt/c/WINDOWS/System32/OpenSSH/ssh.exe
repo: /data/projects/WAM
python: .conda/wam/bin/python
```

Local WSL is for:

```text
code edits
unit tests
documentation
git operations
syncing reports/scripts
```

Ignored artifacts:

```text
data/
outputs/
checkpoints/
*.pt
*.npz
*.npy
```

## 10. Bottom Line

Current technical conclusion:

```text
Phi-only is the stronger current critic. Joint-flow is not yet justified as a
better reranker, but it remains justified as a world-action-potential model
candidate because it learns future action and future observation structure that
phi-only intentionally lacks.
```

Current paper conclusion:

```text
The paper must be honest about phi-only. PP-WAM's distinctive value has to come
from dual-mode inference, future-latent/action consistency, semantic or
base-policy candidate selection, and real suboptimal-process data.
```
