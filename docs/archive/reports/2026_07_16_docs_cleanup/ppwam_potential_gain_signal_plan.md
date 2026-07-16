# PP-WAM Potential/Gain Training Signal Plan

Date: 2026-07-09

## Summary

The proposed shift is correct:

```text
do not train only an isolated image/action score;
learn a comparable state potential Phi first;
train action-conditioned gain DeltaPhi(c, a) as the policy-facing signal.
```

This fits the current PP-WAM story. The WAM still generates or scores:

```text
future action chunk
future observation latent
future potential trajectory
```

But the paper explanation should explicitly separate:

```text
Phi_t: where the process currently is
Phi_{t+K}: where the candidate future lands
DeltaPhi = Phi_{t+K} - Phi_t: whether the action advances, stalls, or regresses
```

The reranker should ultimately optimize action-conditioned `DeltaPhi`, not a
free-floating absolute action score.

## Current Repo Upgrade

The prepared window schema is now upgraded in a backward-compatible way.

Existing training fields remain:

```text
primitive_time
delta_phi
```

New explicit potential/gain fields:

```text
phi_t
phi_future
delta_phi_raw
```

Episode labels can now optionally provide a frame-level potential array:

```json
{
  "primitive_boundaries": [...],
  "potential": [0.0, 0.01, 0.02, ...]
}
```

`phi` is accepted as an alias for `potential`. When this array is present,
window preparation uses it directly:

```text
phi_t = potential[t]
phi_future = potential[t + K]
delta_phi_raw = phi_future - phi_t
delta_phi = max(0, delta_phi_raw)  # legacy compatibility field
```

This means Sim-SubSuccess and ARX-SubSuccess can represent real regression:

```text
delta_phi_raw < 0
```

For current boundary-derived datasets, `delta_phi_raw` is still mostly
non-negative because frame time moves forward. For Sim-SubSuccess and
ARX-SubSuccess, these fields should come from privileged-state or annotated
potential arrays so that `delta_phi_raw` can represent:

```text
progress: DeltaPhi > 0
stagnation: DeltaPhi ~= 0
regression: DeltaPhi < 0
```

An audit tool is available:

```bash
python -m ppwam.potential_gain_audit \
  --windows-dir data/prepared/rh20t_signal_v1 \
  --output-dir outputs/audits/rh20t_potential_gain
```

It checks label presence, `Phi_future - Phi_t == DeltaPhi`, gain distribution,
positive/stagnation/regression rates, and per-episode monotonicity violations.

First audit on existing prepared datasets:

| dataset | windows | has explicit Phi | raw gain mean | negative rate | stagnation rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| RH20T | 56,709 | no | 0.1805 | 0.0000 | 0.0277 |
| REASSEMBLE | 121,206 | no | 0.0658 | 0.0000 | 0.0089 |
| GM100 | 54,909 | no | 0.0262 | 0.0000 | 0.0000 |
| 3-source | 75,000 | no | 0.0912 | 0.0000 | 0.0126 |

Interpretation:

```text
existing prepared datasets still use boundary/primitive-time labels;
they do not yet contain true explicit Phi arrays or negative gain.
```

This confirms that Sim-SubSuccess/ARX import needs to provide explicit
frame-level potential arrays before changing the training objective.

## Data Strategy

### Layer 1: Sim-SubSuccess

Use RoboTwin or a RoboTwin-like simulator for rule-controlled dual-arm
suboptimal-yet-success data.

Start with only three robust patterns:

```text
hesitation / stagnation
detour / inefficient path
overshoot-and-correct
```

Avoid difficult contact-heavy patterns in v0:

```text
regrasp
unnecessary contact
near collision
wrong-arm attempt
```

V0 target:

```text
4 dual-arm tasks
200 expert trajectories per task
600 suboptimal-success trajectories per task
3200 total trajectories
```

Required labels:

```text
Phi_t
DeltaPhi_{t,K}
primitive_id
suboptimal_type
success flag
future observation/action/proprio
```

### Layer 2: ARX-SubSuccess

Use real ARX data as high-quality validation/adaptation, not as the first scale
source.

Pilot target:

```text
2-3 tasks
50-80 successful trajectories
smooth and suboptimal-success variants
manual primitive boundaries
suboptimal segment tags
```

Full target:

```text
6 tasks
240-300 real episodes
```

### Layer 3: Policy Rollouts

Only after the monitor works offline:

```text
base policy rollouts
base policy + PP-WAM reranking
failure/recovery cases
```

Metrics should focus on process quality, not only success:

```text
completion time
regression area
stagnation ratio
suboptimal behavior count
manual intervention count
success rate not degraded
```

## Training Signal

Final target loss:

```text
L = L_abs_potential
  + lambda_delta * L_delta_gain
  + lambda_rank * L_pairwise_gain_ranking
  + lambda_obs * L_future_obs_consequence
```

Minimum next training change:

```text
continue training current DeltaPhi objective for compatibility;
add optional absolute Phi head/loss once prepared datasets include stable phi_t.
```

Do not immediately replace all old metrics. Keep the existing reranking and
calibrated selection metrics as baselines while adding:

```text
Phi MAE
DeltaPhi raw MAE
progress/stagnation/regression classification
macro consistency: sum local gains ~= long-horizon gain
```

## Immediate Next Step

Before touching model architecture again:

1. Run `potential_gain_audit` on RH20T, REASSEMBLE, GM100, and 3-source prepared
   windows. Done for the current prepared datasets.
2. Add a Sim-SubSuccess import adapter only after deciding the concrete
   RoboTwin export format.
3. Build a tiny synthetic Sim-SubSuccess fixture with negative `DeltaPhi` to
   test the full training/evaluation path.
4. Add optional `phi_t` supervision to joint-flow only after the audit confirms
   label scale and split consistency.
