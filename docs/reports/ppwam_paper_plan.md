# PP-WAM Paper Plan

Date: 2026-07-16

This is the compact active paper plan. The full 2026-07-15 planning draft is
archived at:

```text
docs/archive/reports/2026_07_16_docs_cleanup/ppwam_paper_plan_full_2026_07_15.md
```

## Thesis

PP-WAM studies in-domain perturbation recovery for manipulation policies:

```text
Given a nominal policy that can solve a task, can a compact WAM select an action
future that recovers healthy process progress after hesitation, detour, or
overshoot?
```

The main model is a potential-guided joint WAM:

```text
p_theta(a_future, z_future, phi_future | language, obs_history, proprio_history)
```

where:

```text
a_future: future action chunk
z_future: future observation latent
phi_future: future process potential / progress signature
```

The policy-facing score should use action-conditioned gain:

```text
DeltaPhi(c, a) = Phi_{t+K} - Phi_t
```

## Positioning

What to claim:

```text
process-sensitive action reranking
signed progress / stagnation / regression supervision
joint action-observation-potential future modeling
controlled sim perturbations plus real-process validation
```

What not to claim:

```text
generic new-task planning
large-scale video-action foundation modeling
success prediction as the central contribution
RoboTwin as the final dataset contribution
```

## Contributions

1. PP-WAM: a compact typed-token joint WAM over action chunks, future observation
   latents, and process potential.
2. Signed process-gain supervision: `phi_t`, `phi_future`, and
   `delta_phi_raw`, including stagnation and regression segments.
3. Sim-SubSuccess: a RoboTwin controlled perturbation factory with hesitation,
   detour, and overshoot variants.
4. Action reranking diagnostics: imagined-future selection, candidate reranking,
   and expert-vs-perturb action preference.

## Architecture Roadmap

The backbone plan is deliberately staged. V1 is the current repo direction; V2
and V3 are upgrades after the signed-gain and pairwise-ranking evidence is
stable.

### V1: Compact PP-WAM DiT

V1 uses the current architecture:

```text
Frozen encoders / feature extractors:
  DINOv2 visual features
  SigLIP or mock prompt features

Trainable PP-WAM core:
  compact typed-token DiT / Transformer
  self-attention over condition, future action, future observation latent,
  and potential tokens

Outputs:
  action chunk
  future observation latent
  potential curve / signed gain
```

V1 does not separately replace the visual encoder with a video VAE. The point of
V1 is to prove the method with the existing feature stack:

```text
frozen representation extraction
+ our compact joint future module
+ signed process-gain supervision
+ expert-vs-perturb ranking
```

This is the main near-term paper baseline. It should beat late-fusion value
heads and action-only / action-potential-only variants before we invest in a
larger backbone.

### V2: Action-DiT Shared Backbone

V2 tests whether a pretrained or larger action-DiT-style backbone is a better
shared future module than the compact V1 DiT.

Target structure:

```text
Frozen / partially adapted action DiT backbone:
  context tokens
  action tokens
  optional future-observation tokens
  optional potential tokens

Trainable PP-WAM adapters / heads:
  input projections for PP-WAM token streams
  action head
  future observation latent head
  potential / signed-gain head
```

Training should start conservatively:

```text
1. freeze action-DiT backbone
2. train projections and heads
3. add LoRA / adapters if needed
4. unfreeze only the last few blocks as a final ablation
```

V2 is not the first implementation target because action-space, horizon,
embodiment, and token-distribution mismatch can dominate the experiment. Its
purpose is to compare:

```text
compact PP-WAM DiT
vs.
adapted action-DiT shared backbone
```

### V3: Potential-Query Refiner

V3 adds an explicit potential-centric routing module after the shared backbone:

```text
potential queries
  attend to context + action hidden + future-observation hidden
  -> refined potential hidden
  -> potential curve / signed gain head
```

This module should be gated or zero-initialized so it does not destabilize the
shared backbone at initialization. It is used to test the stronger claim:

```text
process potential is read from candidate action and predicted consequence,
not produced by a late-fusion scalar head.
```

Required V3 ablations:

```text
V2 shared backbone without potential-query refiner
V3 full gradient coupling
V3 detached action K/V
V3 detached action+future-obs K/V
late-fusion value/gain head
```

The V3 module should become a main architectural claim only if it improves
signed gain prediction and expert-vs-perturb action preference beyond V1/V2.

## Data Roles

RoboTwin:

```text
controlled, scalable, explainable perturbation signatures
use for method bring-up, ablations, pretraining, and figures
```

ARX / real robot / policy rollout:

```text
real process-quality validation
recovery-pair evidence
stronger paper-facing dataset claim
```

Training usage:

```text
expert / direct:
  action BC = yes
  value / gain = yes
  ranking role = positive

suboptimal action:
  action BC = no
  value / gain = yes
  ranking role = negative
  consequence learning = yes

recovery action:
  action BC = yes
  value / gain = yes
  ranking role = positive
```

## Current Experiment Stack

Completed:

```text
RoboTwin 20-task 2x controlled perturbation import
signed delta_phi_raw prepared windows
DINOv2 feature extraction
potential gain audit
signed joint-flow smoke
expert-vs-perturb paired ranking evaluator
click_bell DP 50-demo epoch-300 eval: 5/10 success
V1 compact typed-token DiT implementation
```

Still weak:

```text
expert-vs-perturb model preference is not reliably learned yet
detour is easier than hesitation / overshoot
current evidence is mostly offline reranking and controlled sim
```

Next decisive experiment:

```text
Finish V1 direct-vs-perturb pairwise training and show that expert/recovery
chunks rank above suboptimal chunks under matched progress context.
```

## Paper Narrative

A concise abstract-level story:

```text
Nominal robot policies can succeed while exhibiting poor local process quality,
or can enter recoverable in-domain perturbations. PP-WAM learns a compact
action-conditioned world model whose imagined futures include both consequences
and process potential. By supervising signed potential gain from successful but
process-variable trajectories, PP-WAM can rerank action futures toward recovery
instead of merely imitating every observed action.
```

## Success Criteria

Minimum publishable evidence:

```text
1. controlled perturbation data with clear signed process-gain labels
2. pairwise action preference over direct/recovery vs suboptimal chunks
3. advantage over BC / DP candidate selection on recovery-relevant diagnostics
4. real ARX or policy rollout validation beyond hand-edited RoboTwin
```
