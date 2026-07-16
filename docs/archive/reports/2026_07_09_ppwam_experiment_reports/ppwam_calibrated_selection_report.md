# PP-WAM Calibrated Imagined-Future Selection Report

Date: 2026-07-09

## Summary

The calibrated selector turns the minimal imagined-future sampler from a
mechanics check into a positive action-selection diagnostic.

The previous naive selector chose the candidate with maximum generated
potential:

```text
argmax generated_phi
```

That raised internal generated phi but did not improve action-clamped rescore.
The new diagnostic scores every generated candidate with the action-clamped
evaluator, then compares:

```text
random sample
max generated_phi
max rescored_phi
calibrated_gap
calibrated_smooth
logged action reference
```

The calibrated score is:

```text
z(generated_phi) + z(rescored_phi)
- z(abs(generated_phi - rescored_phi))
- 0.1 * z(action_smoothness)
```

This score is a test-time selection utility over imagined futures, not a
training loss or ground-truth reward.

## Runs

Executed on the 5060:

```text
host repo: /data/projects/WAM
python: .conda/wam/bin/python
checkpoint: outputs/{source}_cf1p0/seed_42/mvp1_joint_flow/best.pt
rescore_batch_size: 512
max_batches: 32
num_examples: 3072 per run
```

Output paths:

```text
outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/sample_select_test_n16_b32_calibrated
outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow/sample_select_test_n32_b32_calibrated
outputs/reassemble_cf1p0/seed_42/mvp1_joint_flow/sample_select_test_n16_b32_calibrated
outputs/reassemble_cf1p0/seed_42/mvp1_joint_flow/sample_select_test_n32_b32_calibrated
```

## Main Results

| source | N | random rescore | max generated rescore | calibrated gap rescore | calibrated smooth rescore | max rescore | logged rescore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RH20T | 16 | 0.1321 | 0.1360 | 0.1741 | 0.1737 | 0.1770 | 0.1691 |
| RH20T | 32 | 0.1327 | 0.1333 | 0.1763 | 0.1757 | 0.1790 | 0.1691 |
| REASSEMBLE | 16 | 0.1239 | 0.1248 | 0.1300 | 0.1299 | 0.1482 | 0.1181 |
| REASSEMBLE | 32 | 0.1240 | 0.1252 | 0.1318 | 0.1318 | 0.1526 | 0.1181 |

Consistency diagnostics:

| source | N | max-generated gap | calibrated-smooth gap | generated/rescore corr |
| --- | ---: | ---: | ---: | ---: |
| RH20T | 16 | 0.2842 | 0.0605 | 0.2758 |
| RH20T | 32 | 0.3561 | 0.0752 | 0.2698 |
| REASSEMBLE | 16 | 0.0483 | 0.0358 | 0.4379 |
| REASSEMBLE | 32 | 0.0516 | 0.0315 | 0.4356 |

## Interpretation

RH20T is now the cleanest positive result for the potential-guided WAM story:

```text
max-generated selection fails:
  N=16 rescore 0.1360, below logged 0.1691
  N=32 rescore 0.1333, below logged 0.1691

calibrated selection succeeds:
  N=16 calibrated_smooth rescore 0.1737, above logged 0.1691
  N=32 calibrated_smooth rescore 0.1757, above logged 0.1691
```

Increasing N makes naive max-generated selection worse by increasing the
generated/rescored gap, while calibrated selection benefits from more
candidates. This is exactly the failure mode and fix that the paper story needs:

```text
generated phi alone is not enough;
process-potential-guided WAM needs calibration through action-clamped rescore
and consistency penalties.
```

REASSEMBLE remains a calibration stress test. Random generated actions already
score above the logged reference under the current evaluator, so it should not
be used as a standalone policy-improvement claim. It is still useful because
calibrated selection improves over naive max-generated selection and reduces
the generated/rescored gap.

## Paper Takeaway

The action-selection evidence should now be stated as:

```text
PP-WAM samples multiple imagined action/observation/potential futures. Selecting
by generated potential alone is not reliable, but a calibrated score combining
generated potential, action-clamped rescore, and consistency penalties selects
futures with higher evaluator-approved primitive-local process potential.
```

This supports the main narrative:

```text
PP-WAM is not merely an imitation model that emits plausible actions. It exposes
an imagined-future interface where action candidates can be selected by
process-potential consistency.
```

## Next Steps

1. Add a semantic/base-policy candidate set so candidates are not only sampled
   from the WAM prior.
2. Add source-mixing diagnostics before expanding three-source seeds.
3. Build ARX-SubSuccess better/worse successful segment pairs, where the same
   task succeeds but process quality differs.
4. Move to Base scale only after the semantic/base-policy and ARX pilot metrics
   confirm that calibrated selection beats phi-only under realistic candidates.
