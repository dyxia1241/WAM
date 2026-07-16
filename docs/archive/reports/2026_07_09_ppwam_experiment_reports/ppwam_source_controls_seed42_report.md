# PP-WAM Source Controls Seed-42 Report

Date: 2026-07-09

## Summary

The RH20T and REASSEMBLE single-source controls are complete on the 5060.

Main result:

```text
Single-source training substantially improves RH20T and REASSEMBLE ranking
relative to the three-source seed-42 models.
```

This means the next paper step should not be blind seed expansion. The
three-source pipeline is useful, but its source mixing currently changes the
ranking/calibration tradeoff enough that source-specific diagnostics are now
the highest-value work.

## Runs

Executed on:

```text
host: dayu-TX-Gaming-FA608UM-FA608UM
repo: /data/projects/WAM
HEAD: 57e6ba8
python: .conda/wam/bin/python
```

Commands:

```bash
.conda/wam/bin/python -m ppwam.joint_flow --config configs/rh20t/joint_flow_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/rh20t/phi_only_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/reassemble/joint_flow_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/reassemble/phi_only_cf1p0.yaml
```

All four runs wrote:

```text
best.pt
metrics.json
history.jsonl
eval_test/metrics.json
experiment_report.md
```

## Main Comparison

Source-specific metrics for the three-source rows are taken from the three-source
seed-42 eval outputs. Single-source rows are the newly completed controls.

| source | training | model | coarse ranking | all-neg ranking | DeltaPhi MAE |
| --- | --- | --- | ---: | ---: | ---: |
| RH20T | 3-source | joint-flow | 0.8751 | 0.7401 | 0.1158 |
| RH20T | 3-source | phi-only | 0.9069 | 0.8262 | 0.1204 |
| RH20T | RH20T-only | joint-flow | 0.9785 | 0.9099 | 0.1181 |
| RH20T | RH20T-only | phi-only | 0.9444 | 0.7921 | 0.1135 |
| REASSEMBLE | 3-source | joint-flow | 0.8232 | 0.7050 | 0.0360 |
| REASSEMBLE | 3-source | phi-only | 0.9048 | 0.8116 | 0.0400 |
| REASSEMBLE | REASSEMBLE-only | joint-flow | 0.9806 | 0.9291 | 0.0664 |
| REASSEMBLE | REASSEMBLE-only | phi-only | 0.9986 | 0.9744 | 0.7542 |

Top-1 and consequence metrics for the four single-source controls:

| source | model | coarse top-1 | all-neg top-1 | critic future obs y0 MSE | predictor action y0 MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| RH20T | joint-flow | 0.9402 | 0.6329 | 0.4347 | 0.1600 |
| RH20T | phi-only | 0.8352 | 0.2891 | 1.2030 | 0.6965 |
| REASSEMBLE | joint-flow | 0.9652 | 0.7262 | 0.4480 | 0.0406 |
| REASSEMBLE | phi-only | 0.9964 | 0.8846 | 1.1751 | 0.5698 |

## Interpretation

RH20T-only is the cleanest positive signal for joint-flow:

```text
joint-flow beats phi-only on coarse ranking, all-negative ranking, top-1
ranking, and action/future-observation consequence metrics.
```

REASSEMBLE-only is more mixed:

```text
phi-only nearly saturates synthetic ranking, but its DeltaPhi MAE is extremely
large. Joint-flow is much better calibrated and much better on consequence
metrics, but still loses simple ranking to phi-only.
```

The three-source result is therefore not simply "phi-only wins everywhere." A
more accurate reading is:

```text
single-source joint-flow can be a strong critic on RH20T;
single-source REASSEMBLE exposes a ranking-vs-calibration failure mode;
three-source mixing substantially hurts ranking on RH20T and REASSEMBLE.
```

## Next Step

Do not expand 3-source seeds yet.

The highest-value next step is a source-mixing diagnostic:

1. compare per-source DeltaPhi distributions, action/proprio z-score ranges,
   action padding, camera padding, and prompt feature mappings;
2. inspect why REASSEMBLE phi-only can rank nearly perfectly while producing
   unusable DeltaPhi scale;
3. add calibration-gated ranking or future-latent consistency scoring before
   claiming reranking utility;
4. only expand seeds after the source-mixing failure mode is understood.
