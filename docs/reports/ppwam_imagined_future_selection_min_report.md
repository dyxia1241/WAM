# PP-WAM Imagined-Future Selection Minimal Report

Date: 2026-07-09

## Run

Minimal potential-guided WAM diagnostic on the RH20T-only joint-flow checkpoint:

```bash
cd /data/projects/WAM
.conda/wam/bin/python -m ppwam.joint_flow_sample_select \
  --checkpoint outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/best.pt \
  --num-samples 16 \
  --max-batches 32 \
  --output-dir outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/sample_select_test_n16_b32
```

Output:

```text
outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/sample_select_test_n16_b32/metrics.json
outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/sample_select_test_n16_b32/selection_rows.csv
outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/sample_select_test_n16_b32/manifest.json
```

The run evaluated 3072 test examples.

## Metrics

| metric | value |
| --- | ---: |
| selected generated phi mean | 0.4202 |
| random generated phi mean | 0.1934 |
| selected - random generated phi | 0.2268 |
| selected rescored phi mean | 0.1360 |
| random rescored phi mean | 0.1321 |
| logged rescored phi mean | 0.1691 |
| selected - random rescored phi | 0.0039 |
| selected - logged rescored phi | -0.0331 |
| selected beats random rescore rate | 0.4658 |
| selected beats logged rescore rate | 0.3304 |
| generated/rescored phi abs gap | 0.2842 |
| selected action MSE to logged | 0.5085 |
| random action MSE to logged | 0.5042 |

## Interpretation

The minimal sampler successfully operationalizes the new potential-guided WAM
story:

```text
context -> N generated futures -> choose highest generated potential
```

But naive generated-potential selection is not yet a healthy action-selection
mechanism.

Positive signal:

```text
selected generated phi is much higher than random generated phi
```

Failure mode:

```text
the selected action barely improves action-clamped rescore over random,
loses to the logged action, and has a large generated-vs-rescored phi gap.
```

This means the current joint-flow model can generate futures with higher
internal phi, but the generated phi is not yet calibrated enough to select
actions by itself.

## Next Step

Do not claim policy improvement from naive self-sampling.

The next implementation should add a calibrated selection score:

```text
score = generated_phi
      + alpha * action_clamped_rescore
      - beta * |generated_phi - action_clamped_rescore|
      - gamma * action_smoothness_or_implausibility
```

Then compare:

```text
random generated sample
max generated_phi sample
max calibrated score sample
logged action
```

If calibrated selection beats random and approaches or exceeds logged-action
rescore without implausible action drift, the potential-guided WAM story becomes
much stronger.
