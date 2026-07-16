# PP-WAM Current Status

Date: 2026-07-16

This is the active project status page. Detailed historical reports are archived
under `docs/archive/reports/`.

## Main Direction

PP-WAM is currently framed as:

```text
Compact process-potential WAM for in-domain perturbation recovery.
```

The model jointly represents:

```text
future action chunk
future observation latent
future process potential / gain
```

The intended test-time use is not generic task planning. It is reranking or
selecting action futures when a nominal manipulation policy enters an in-domain
suboptimal process such as hesitation, detour, overshoot, or recovery.

## Current Evidence

RoboTwin Sim-SubSuccess is the active controlled data factory:

```text
20 tasks x 3 perturbations = 60 2x suboptimal-success episodes
prepared windows = 7494
vision features = 60 DINOv2 files
potential audit:
  delta_phi_raw_min = -0.04663
  delta_phi_raw_mean = 0.03021
  negative_rate ~= 14.96%
  stagnation_rate ~= 15.08%
```

The signed-gain data path is implemented and tested:

```text
RoboTwin HDF5 import
frame-level potential sidecars
phi_t / phi_future / signed delta_phi_raw window labels
signed target support in joint_flow
potential gain and variant audits
expert-vs-perturb pair ranking evaluator
```

The current 2-task task-specific rule-v2 pilot is generated and audited on
5060:

```text
tasks: beat_block_hammer, click_bell
episodes: 8 total
  expert/direct: 2
  controlled perturbations: hesitation, detour, overshoot for each task
prepared windows: 660
delta_phi_raw_min = -0.05960
delta_phi_raw_mean = 0.04673
negative_rate ~= 17.88%
stagnation_rate ~= 25.61%
phi consistency max error = 0.0
```

Rule-v2 labels are task-specific and active-arm aware:

```text
beat_block_hammer: right-arm EEF rule
click_bell:        left-arm EEF rule
```

Current architecture version:

```text
V1 compact PP-WAM DiT
  frozen DINOv2 visual features
  frozen / mock SigLIP prompt features
  trainable typed-token Transformer
  future obs + action + potential flow heads
```

Planned upgrades:

```text
V2 action-DiT shared backbone adaptation
V3 potential-query refiner after the shared backbone
```

Earlier RH20T diagnostics support the potential-guided imagined-future story:

```text
calibrated imagined-future selection beats random and logged-action rescore on RH20T
mixed candidate reranking: RH20T joint-flow strict pairwise = 0.9128
```

RoboTwin DP baseline status:

```text
task: click_bell
data: 50 clean expert demonstrations
checkpoint evaluated: demo_clean epoch 300
rollouts: 10
success: 5 / 10
local videos: outputs/robotwin_policy_rollouts/dp_click_bell_demo_clean_50_epoch300_20260715_171744/
```

The attempted 600-epoch DP watcher did not produce a completed 600 checkpoint
for evaluation during the last run.

Official RoboTwin2.0 ACT checkpoints are the current task-specific policy
source for autonomous rollouts beyond local `click_bell` DP:

```text
5060 target:
  /data/projects/RoboTwin/policy/ACT/checkpoints/official_demo_clean_50

expected files:
  beat_block_hammer/demo_clean-50/policy_last.ckpt   335,907,442 bytes
  beat_block_hammer/demo_clean-50/dataset_stats.pkl       7,752 bytes
  click_bell/demo_clean-50/policy_last.ckpt          335,907,442 bytes
  click_bell/demo_clean-50/dataset_stats.pkl              6,184 bytes

total expected size: 671,828,820 bytes ~= 640.71 MiB
```

## Key Paths

5060:

```text
ssh dayu@10.1.233.65
WAM:      /data/projects/WAM
RoboTwin: /data/projects/RoboTwin
python:   /data/projects/WAM/.conda/wam/bin/python
```

RoboTwin 20-task WAM artifacts on 5060:

```text
data/episodes/robotwin_20task_2x_rule_subsuccess_v1
data/prepared/robotwin_20task_2x_rule_subsuccess_v1
data/features/robotwin_20task_2x_rule_subsuccess_v1_dinov2_vitb14_224
outputs/audits/robotwin_20task_2x_rule_subsuccess_v1_gain
```

RoboTwin 2-task task-specific rule-v2 artifacts on 5060:

```text
data/episodes/robotwin_2task_rule_v2
data/prepared/robotwin_2task_rule_v2
outputs/audits/robotwin_2task_rule_v2_gain
docs/figures/current/robotwin_2task_rule_v2_potential/
```

Current figures:

```text
docs/figures/current/robotwin_20task_2x_rule_potential/
docs/figures/current/robotwin_2x_rule_potential/
docs/figures/current/robotwin_click_bell_2x_rule_potential/
docs/figures/current/robotwin_2task_rule_v2_potential/
docs/figures/current/gm100_mvp1_6_validation/
```

## Next Work

Immediate technical next steps stay within V1:

1. Add a direct-vs-perturb pairwise training loss using the existing expert-pair
   rows.
2. Re-evaluate expert-vs-perturb ranking on `beat_block_hammer` and `click_bell`
   before expanding pairwise training to all 20 tasks.
3. Keep RoboTwin as controlled Sim-SubSuccess; use ARX / policy rollouts for
   real process validation.
4. Do not put suboptimal action chunks into ordinary BC loss; use them for
   potential/gain, ranking, and consequence learning.

Backbone roadmap:

```text
V1 now:
  current compact typed-token PP-WAM DiT.
  Do not separately replace encoders yet.

V2 next:
  adapt an action-DiT-style shared backbone with PP-WAM projections and heads.
  Start frozen, then LoRA/adapters, then last-block unfreeze if needed.

V3 after V2:
  add a gated potential-query module where phi tokens attend to context,
  action hidden, and future-observation hidden.
```

## Active Docs

```text
docs/reports/ppwam_status.md       current status and evidence
docs/reports/ppwam_paper_plan.md   compact paper plan
docs/5060_ssh.md                   machine access notes
```

Historical details were moved to:

```text
docs/archive/reports/2026_07_16_docs_cleanup/
docs/archive/reports/2026_07_09_ppwam_experiment_reports/
docs/archive/reports/2026_07_14_robotwin_1x_smoke/
```
