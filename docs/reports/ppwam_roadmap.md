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

1. Keep `cf_1p0` as the current main MVP1 critic config.
2. Run the phi-only strong baseline in `configs/gm100/phi_only_cf1p0.yaml` after explicit training confirmation.
3. Add downstream candidate-reranking evaluation with positive action plus sampled/base-policy negatives.
4. Add calibration-aware selection: coarse ranking first, MAE bounded second.
5. Make masked critic/predictor mode first-class in training.
6. Improve labels only if temporal-order metrics become a main claim.

## Backbone Choice

Use the current lightweight DiT for MVP1/MVP2:

- small enough for GM-100 experiments on the 5060;
- supports typed observation/action/potential tokens;
- keeps attention as the coupling mechanism across modalities;
- leaves room for larger DiT or pretrained backbones only after the scientific signal is stable.
