# PP-WAM Potential-Guided Joint WAM Plan

Date: 2026-07-09

## Narrative

The main PP-WAM story should be a potential-guided joint WAM, not a dual-mode
predictor/critic system.

Core model:

```text
p_theta(a_{t:t+K-1}, z_{t+1:t+K}, phi_{t:t+K-1} | language, obs_history, proprio_history)
```

PP-WAM extends joint future-state/action WAMs to joint
future-state/action/process-potential modeling. At inference time, it samples
multiple imagined futures and executes the action prefix from the future with
the best calibrated primitive-local potential.

Main test-time loop:

```text
context c_t
  -> sample N imagined futures (a_i, z_i, phi_i)
  -> score each future by calibrated primitive-local potential
  -> execute the first h steps of the selected action chunk
  -> re-observe and replan
```

The action-conditioned scorer remains useful, but it should be described as an
auxiliary evaluator/diagnostic rather than the main model definition.

## Why This Fits the Current Evidence

The 2026-07-09 source controls support this route:

```text
RH20T-only joint-flow beats RH20T-only phi-only on ranking and top-1 metrics.
REASSEMBLE-only exposes that ranking alone can be misleading when DeltaPhi
calibration collapses.
```

This means the paper should not say "joint-flow is always a stronger critic."
It should say:

```text
Joint future action / future observation / process-potential modeling can
improve action selection when the evaluation uses imagined futures, calibrated
potential, and consequence consistency.
```

## Minimal Experiment

Run a first offline imagined-future selection diagnostic on the RH20T-only
joint-flow checkpoint:

```bash
python -m ppwam.joint_flow_sample_select \
  --checkpoint outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/best.pt \
  --num-samples 16 \
  --max-batches 32
```

The diagnostic samples N futures per context, ranks them by generated potential,
and compares the selected future against:

```text
random generated sample
logged action rescored by action-clamped PP-WAM
selected generated action rescored by action-clamped PP-WAM
```

Success for this minimal version is modest:

```text
selected futures should beat random generated futures on rescored potential;
selected actions should remain finite and not obviously implausible;
generated potential and action-clamped rescore should not be completely
decorrelated.
```

This is not yet a real robot or policy result. It is the smallest check that the
new potential-guided WAM story can be operationalized by the existing model.

### First Result

The first RH20T-only run completed on 2026-07-09:

```text
num_samples = 16
num_examples = 3072
selected generated phi = 0.4202
random generated phi = 0.1934
selected rescored phi = 0.1360
random rescored phi = 0.1321
logged rescored phi = 0.1691
generated/rescored phi abs gap = 0.2842
```

This validates the mechanics of imagined-future sampling, but it does not yet
validate naive policy selection. The selected futures have much higher generated
phi, but the selected actions do not meaningfully beat random after
action-clamped rescoring.

The next selector should therefore be calibrated:

```text
score = generated_phi
      + alpha * action_clamped_rescore
      - beta * |generated_phi - action_clamped_rescore|
      - gamma * action_smoothness_or_implausibility
```

## Next Decisions

If the minimal diagnostic is healthy:

```text
add future-latent consistency and action smoothness penalties;
repeat on REASSEMBLE joint-flow;
then build a semantic/base-policy candidate comparison.
```

If it is unhealthy:

```text
do not expand seeds;
inspect source mixing, phi scale, generated action distribution, and whether
training needs explicit potential-conditioned generation.
```
