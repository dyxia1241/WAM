# PP-WAM Roadmap

## Paper Direction

Target venue: ICRA / ICLR 2027.

Core claim:

```text
Primitive-local process potential should be modeled jointly with action and future observation latent dynamics, not attached as a standalone score after action generation.
```

## MVP Roles

- MVP0: motivation and discriminative critic baseline.
- MVP1: latent-action-potential joint flow with critic-mode supervision.
- MVP2: masked predictor/critic training, where the same model supports action-known critic mode and action-unknown predictor/policy mode.
- MVP3: test-time flow refinement, using previous-action initialization, adaptive denoising steps, and candidate reranking.
- Final system: downstream action selection with sampled or base-policy candidate chunks.

## Near-Term Plan

1. Keep `cf_1p0` as the current main joint-flow candidate, but do not claim it beats a strong phi-only critic on current synthetic coarse negatives.
2. Treat the first hard candidate-reranking pass as a control result: phi-only still beats `cf_1p0` on logged-action retrieval.
3. Add semantic/base-policy candidate reranking beyond action-bank distractors.
4. Add data-driven hard negatives to training only after the evaluation candidate sets are stable.
5. Add calibration-aware selection: coarse/hard ranking first, MAE bounded second.
6. Make masked critic/predictor mode first-class in training only after hard reranking or predictor utility clarifies where joint-flow has a practical advantage.
7. Improve labels only if temporal-order metrics become a main claim.

## Current Tactical Result

The hard reranking evaluator is implemented in `ppwam.joint_flow_rerank`. It ranks the logged test action against four action-bank distractors per anchor: same-task phase-wrong, same-task far-progress, cross-task, and nearest-observation wrong-action.

First-pass GM-100 hard reranking still favors the phi-only strong baseline:

| model | hard pairwise ranking | hard top-1 |
| --- | ---: | ---: |
| MVP1.6 `cf_1p0` | 0.8336+/-0.0756 | 0.6391+/-0.1341 |
| phi-only strong baseline `cf1p0` | 0.8795+/-0.0304 | 0.7248+/-0.0590 |

This means the paper should not claim that joint-flow is already a stronger critic. The next tactical objective is to test whether joint-flow earns its complexity through semantic candidate sets, base-policy candidate reranking, future-latent consistency, or action-unknown predictor mode.

## Backbone Choice

Use the current lightweight DiT for MVP1/MVP2:

- small enough for GM-100 experiments on the 5060;
- supports typed observation/action/potential tokens;
- keeps attention as the coupling mechanism across modalities;
- leaves room for larger DiT or pretrained backbones only after the scientific signal is stable.
