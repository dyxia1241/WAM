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
```

Still weak:

```text
expert-vs-perturb model preference is not reliably learned yet
detour is easier than hesitation / overshoot
current evidence is mostly offline reranking and controlled sim
```

Next decisive experiment:

```text
Train with direct-vs-perturb pairwise loss and show that expert/recovery chunks
rank above suboptimal chunks under matched progress context.
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
