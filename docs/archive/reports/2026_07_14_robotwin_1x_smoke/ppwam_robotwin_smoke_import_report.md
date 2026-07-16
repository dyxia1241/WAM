# PP-WAM RoboTwin Smoke Import Report

Date: 2026-07-13

## Summary

RoboTwin is now usable on the 5060 as a Sim-SubSuccess upstream smoke source.
The first WAM-side adapter is implemented and verified:

```text
RoboTwin HDF5 -> WAM episode layout -> prepare_windows -> potential/gain audit
```

The adapter is intentionally conservative. It does not claim real
suboptimal-success data yet; it proves the HDF5 export format can feed WAM and
that external potential sidecars can express negative gain.

This report is an execution log. The scaling protocol, task plan, trajectory
modification taxonomy, Phi annotation rules, and downstream signed-gain training
plan are tracked separately in:

```text
docs/archive/reports/2026_07_16_docs_cleanup/ppwam_robotwin_subsuccess_data_plan.md
```

## 5060 State

Use:

```bash
ssh dayu@10.1.233.65
```

Relevant paths:

```text
RoboTwin repo: /data/projects/RoboTwin
RoboTwin env:  /home/dayu/anaconda3/envs/RoboTwin
WAM repo:      /data/projects/WAM
WAM env:       /data/projects/WAM/.conda/wam
```

Do not use the WAM env for RoboTwin runtime work. Do not clean the WAM dirty
tree.

## RoboTwin Runtime Verification

RoboTwin environment on the 5060:

```text
torch 2.7.1+cu128
torch CUDA 12.8
RTX 5060 Laptop GPU available
sapien / mplib / open3d / curobo / warp imports OK
pip check OK
```

Render smoke:

```bash
cd /data/projects/RoboTwin
/home/dayu/anaconda3/envs/RoboTwin/bin/python script/test_render.py
```

Result:

```text
Render Well
```

## RoboTwin Smoke Data

Collected smoke task:

```text
task:   beat_block_hammer
config: ppwam_smoke_clean
```

Artifacts:

```text
/data/projects/RoboTwin/data/beat_block_hammer/ppwam_smoke_clean/data/episode0.hdf5
/data/projects/RoboTwin/data/beat_block_hammer/ppwam_smoke_clean/video/episode0.mp4
/data/projects/RoboTwin/data/beat_block_hammer/ppwam_smoke_clean/instructions/episode0.json
/data/projects/RoboTwin/data/beat_block_hammer/ppwam_smoke_clean/scene_info.json
```

HDF5 shape summary:

```text
frames: 117
joint_action/vector: (117, 14)
left/right endpose:  (117, 7)
left/right gripper:  (117,)
cameras: front_camera, head_camera, left_camera, right_camera
```

## WAM Adapter

New code:

```text
ppwam/import_robotwin.py
scripts/import_robotwin.py
tests/test_import_robotwin.py
```

Default mapping:

```text
proprio = joint_action/vector
        + left_endpose + left_gripper
        + right_endpose + right_gripper

action[t] = proprio[t + 1]
action[-1] = proprio[-1]
```

Default labels are smoke-only:

```text
primitive_boundaries: even approach/grasp/move/release split
potential: linear 0 -> 1
```

For Sim-SubSuccess, pass an explicit sidecar:

```bash
python -m ppwam.import_robotwin \
  --hdf5 /path/to/episode0.hdf5 \
  --output data/episodes/robotwin_variant_v0 \
  --label-sidecar data/robotwin_sidecars/example.json \
  --instructions /path/to/instructions/episode0.json \
  --scene-info /path/to/scene_info.json \
  --overwrite
```

Sidecar fields:

```json
{
  "label_source": "robotwin_subsuccess_rule_v0",
  "success": true,
  "suboptimal_type": "overshoot_and_correct",
  "primitive_boundaries": [
    {"stage": "approach", "start": 0, "end": 29},
    {"stage": "grasp", "start": 30, "end": 58},
    {"stage": "move", "start": 59, "end": 87},
    {"stage": "release", "start": 88, "end": 116}
  ],
  "potential": [0.0, 0.01, 0.02]
}
```

`phi` is accepted as an alias for `potential` through the existing WAM label
reader.

## Verified WAM Outputs

Expert-style smoke import:

```text
/data/projects/WAM/data/episodes/robotwin_smoke_v0
/data/projects/WAM/data/prepared/robotwin_smoke_v0
/data/projects/WAM/outputs/audits/robotwin_smoke_v0_potential_gain
```

Result:

```text
episode frames: 117
proprio/action dim: 30
prepared windows: 54
delta_phi_raw consistency: true
negative gain rate: 0.0
```

Overshoot sidecar smoke:

```text
/data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_overshoot_sidecar.json
/data/projects/WAM/data/episodes/robotwin_smoke_overshoot_v0
/data/projects/WAM/data/prepared/robotwin_smoke_overshoot_v0
/data/projects/WAM/outputs/audits/robotwin_smoke_overshoot_v0_potential_gain
```

Result:

```text
prepared windows: 54
delta_phi_raw min: -0.2200
delta_phi_raw max: 0.1168
negative gain rate: 0.0741
stagnation rate: 0.0185
delta_phi_raw == phi_future - phi_t: true
```

Simulator-produced hesitation replay:

```text
RoboTwin config: /data/projects/RoboTwin/task_config/ppwam_hesitation_replay.yml
RoboTwin data:   /data/projects/RoboTwin/data/beat_block_hammer/ppwam_hesitation_replay
WAM sidecar:     /data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_hesitation_replay_sidecar.json
WAM episode:     /data/projects/WAM/data/episodes/robotwin_hesitation_replay_v0
WAM windows:     /data/projects/WAM/data/prepared/robotwin_hesitation_replay_v0
WAM audit:       /data/projects/WAM/outputs/audits/robotwin_hesitation_replay_v0_potential_gain
```

Generation method:

```text
copy the successful ppwam_smoke_clean seed and _traj_data
set use_seed: true
insert 180 repeated joint rows with zero velocity into right_joint_path[2]
run RoboTwin collect_data.sh to replay through SAPIEN and export a new HDF5/video
```

This is not only a label-sidecar transformation. The new HDF5/video were
produced by RoboTwin replay:

```text
expert smoke frames:     117
hesitation replay frames: 129
right-arm low-motion run: frames 49-73
four cameras imported: 129 JPEGs per camera
```

WAM audit:

```text
prepared windows: 60
delta_phi_raw min: 0.0000
delta_phi_raw max: 0.0859
negative gain rate: 0.0
stagnation rate: 0.15
delta_phi_raw == phi_future - phi_t: true
```

Simulator-produced detour replay:

```text
RoboTwin config: /data/projects/RoboTwin/task_config/ppwam_detour_replay.yml
RoboTwin data:   /data/projects/RoboTwin/data/beat_block_hammer/ppwam_detour_replay
WAM sidecar:     /data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_detour_replay_sidecar.json
WAM episode:     /data/projects/WAM/data/episodes/robotwin_detour_replay_v0
WAM windows:     /data/projects/WAM/data/prepared/robotwin_detour_replay_v0
WAM audit:       /data/projects/WAM/outputs/audits/robotwin_detour_replay_v0_potential_gain
```

Generation method:

```text
copy the successful ppwam_smoke_clean seed and _traj_data
set use_seed: true
insert a 180-row smooth joint-space loop into right_joint_path[0]
run RoboTwin collect_data.sh to replay through SAPIEN and export a new HDF5/video
```

This replay preserves the original endpoint but adds an inefficient approach
detour before returning to the expert path:

```text
expert smoke frames:       117
detour replay frames:      129
edited segment rows:       509 -> 689
edited segment path ratio: 1.069
right-arm HDF5 path ratio: 1.034
detour frame range:        frames 17-28
four cameras imported:     129 JPEGs per camera
```

WAM audit:

```text
prepared windows: 60
delta_phi_raw min: 0.0000
delta_phi_raw max: 0.0701
negative gain rate: 0.0
stagnation rate: 0.05
delta_phi_raw == phi_future - phi_t: true
```

Simulator-produced overshoot-and-correct replay:

```text
RoboTwin config: /data/projects/RoboTwin/task_config/ppwam_overshoot_replay.yml
RoboTwin data:   /data/projects/RoboTwin/data/beat_block_hammer/ppwam_overshoot_replay
WAM sidecar:     /data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_overshoot_replay_sidecar.json
WAM episode:     /data/projects/WAM/data/episodes/robotwin_overshoot_replay_v0
WAM windows:     /data/projects/WAM/data/prepared/robotwin_overshoot_replay_v0
WAM audit:       /data/projects/WAM/outputs/audits/robotwin_overshoot_replay_v0_potential_gain
```

Generation method:

```text
copy the successful ppwam_smoke_clean seed and _traj_data
set use_seed: true
append a 180-row out-and-back joint-space loop to right_joint_path[4]
run RoboTwin collect_data.sh to replay through SAPIEN and export a new HDF5/video
```

This replay overshoots the final release/contact pose and corrects back to the
expert final joint pose:

```text
expert smoke frames:        117
overshoot replay frames:    129
edited segment rows:        158 -> 338
edited segment path ratio:  1.681
right-arm HDF5 path ratio:  1.031
right-EE xyz path ratio:    1.015
potential dip frame range:  frames 111-119
four cameras imported:      129 JPEGs per camera
```

WAM audit:

```text
prepared windows: 60
delta_phi_raw min: -0.2000
delta_phi_raw max: 0.2650
negative gain rate: 0.0833
stagnation rate: 0.0
delta_phi_raw == phi_future - phi_t: true
phi_t monotonic violation rate: 0.0847
```

Combined RoboTwin Sim-SubSuccess smoke source:

```text
WAM episode root: /data/projects/WAM/data/episodes/robotwin_subsuccess_smoke_v0
WAM windows:      /data/projects/WAM/data/prepared/robotwin_subsuccess_smoke_v0
WAM audit:        /data/projects/WAM/outputs/audits/robotwin_subsuccess_smoke_v0_potential_gain
```

Included episodes:

```text
expert_beat_block_hammer_episode0:     117 frames, 4 cameras
hesitation_beat_block_hammer_episode0: 129 frames, 4 cameras
detour_beat_block_hammer_episode0:     129 frames, 4 cameras
overshoot_beat_block_hammer_episode0:  129 frames, 4 cameras
```

Prepared-window split:

```text
num windows: 234
train: overshoot_beat_block_hammer_episode0, hesitation_beat_block_hammer_episode0
val:   expert_beat_block_hammer_episode0
test:  detour_beat_block_hammer_episode0
```

Combined audit:

```text
delta_phi_raw min: -0.2000
delta_phi_raw max: 0.2650
negative gain rate: 0.0214
stagnation rate: 0.0513
delta_phi_raw == phi_future - phi_t: true
phi_t monotonic violation rate: 0.0217
```

## Second Task Smoke: Click Bell

The same RoboTwin `ppwam_smoke_clean` config was also run for a second task:

```text
task:   click_bell
config: ppwam_smoke_clean
seed:   0
frames: 81
```

RoboTwin artifacts:

```text
/data/projects/RoboTwin/data/click_bell/ppwam_smoke_clean/data/episode0.hdf5
/data/projects/RoboTwin/data/click_bell/ppwam_smoke_clean/video/episode0.mp4
/data/projects/RoboTwin/data/click_bell/ppwam_smoke_clean/instructions/episode0.json
/data/projects/RoboTwin/data/click_bell/ppwam_smoke_clean/scene_info.json
```

HDF5 shape summary:

```text
frames: 81
joint_action/vector: (81, 14)
left/right endpose:  (81, 7)
left/right gripper:  (81,)
cameras: front_camera, head_camera, left_camera, right_camera
language: Push the white dome bell's <top center> on the table
```

WAM outputs:

```text
episode root: /data/projects/WAM/data/episodes/robotwin_click_bell_smoke_v0
windows:      /data/projects/WAM/data/prepared/robotwin_click_bell_smoke_v0
features:     /data/projects/WAM/data/features/robotwin_click_bell_smoke_dinov2_vitb14_224
prompts:      /data/projects/WAM/data/prompts/robotwin_click_bell_smoke_siglip
audit:        /data/projects/WAM/outputs/audits/robotwin_click_bell_smoke_v0_potential_gain
```

WAM audit:

```text
prepared windows: 36
delta_phi_raw min: 0.0875
delta_phi_raw max: 0.1000
negative gain rate: 0.0
stagnation rate: 0.0
delta_phi_raw == phi_future - phi_t: true
phi_t monotonic violation rate: 0.0
```

Feature and prompt stores:

```text
click_bell_episode0.npz: 81 frames x 4 cameras x 768
SigLIP prompt shape:    1 task x 768
SigLIP task_ids:        click_bell
norm stats:             1 episode, 81 frames
```

Train-batch sanity passed for `JointFlowPreparedWindowDataset`:

```text
obs_features:        [16, 4, 4, 768]
future_obs_features: [16, 8, 4, 768]
proprio:             [16, 30]
proprio_history:     [16, 4, 30]
action_chunk:        [16, 8, 30]
prompt_features:     [16, 768]
model outputs:       v_obs [16, 8, 768], v_action [16, 8, 30], v_phi [16, 5]
single forward loss: 9.4528
```

Simulator-produced `click_bell` hesitation replay:

```text
RoboTwin config: /data/projects/RoboTwin/task_config/ppwam_hesitation_replay.yml
RoboTwin data:   /data/projects/RoboTwin/data/click_bell/ppwam_hesitation_replay
WAM sidecar:     /data/projects/WAM/data/robotwin_sidecars/click_bell_hesitation_replay_sidecar.json
WAM episode:     /data/projects/WAM/data/episodes/robotwin_click_bell_hesitation_replay_v0
WAM windows:     /data/projects/WAM/data/prepared/robotwin_click_bell_hesitation_replay_v0
WAM audit:       /data/projects/WAM/outputs/audits/robotwin_click_bell_hesitation_replay_v0_potential_gain
```

Generation method:

```text
copy the successful click_bell ppwam_smoke_clean seed and _traj_data
set use_seed: true
insert 180 repeated joint rows with zero velocity into left_joint_path[1]
run RoboTwin collect_data.sh to replay through SAPIEN and export a new HDF5/video
```

This replay adds a simulator-observed low-motion segment after the original
press/contact run:

```text
expert click_bell frames:      81
hesitation replay frames:      93
edited segment rows:           136 -> 316
inserted pause rows:           180
original low-motion run:       frames 33-57
inserted low-motion run:       frames 64-74
four cameras imported:         93 JPEGs per camera
```

WAM audit:

```text
prepared windows: 42
delta_phi_raw min: 0.0000
delta_phi_raw max: 0.1432
negative gain rate: 0.0
stagnation rate: 0.0476
delta_phi_raw == phi_future - phi_t: true
phi_t monotonic violation rate: 0.0
```

Feature and loader sanity:

```text
click_bell_episode0.npz: 93 frames x 4 cameras x 768
norm stats:             1 episode, 93 frames
prompt_features:        reused click_bell SigLIP [1, 768]
train windows:          42
prompt_features batch:  [16, 768]
model outputs:          v_obs [16, 8, 768], v_action [16, 8, 30], v_phi [16, 5]
single forward loss:    9.4908
```

Simulator-produced `click_bell` detour replay:

```text
RoboTwin data: /data/projects/RoboTwin/data/click_bell/ppwam_detour_replay
WAM sidecar:   /data/projects/WAM/data/robotwin_sidecars/click_bell_detour_replay_sidecar.json
WAM episode:   /data/projects/WAM/data/episodes/robotwin_click_bell_detour_replay_v0
WAM windows:   /data/projects/WAM/data/prepared/robotwin_click_bell_detour_replay_v0
WAM audit:     /data/projects/WAM/outputs/audits/robotwin_click_bell_detour_replay_v0_potential_gain
```

Generation method:

```text
copy the successful click_bell ppwam_smoke_clean seed and _traj_data
set use_seed: true
insert a 180-row smooth out-and-back loop into left_joint_path[0]
run RoboTwin collect_data.sh to replay through SAPIEN and export a new HDF5/video
```

This replay preserves the original endpoint while adding an inefficient approach
detour:

```text
expert click_bell frames: 81
detour replay frames:     93
edited segment rows:      502 -> 682
edited segment path ratio: 1.044
left-arm HDF5 path ratio: 1.034
detour frame range:       frames 18-29
four cameras imported:    93 JPEGs per camera
```

WAM audit:

```text
prepared windows: 42
delta_phi_raw min: 0.0000
delta_phi_raw max: 0.1038
negative gain rate: 0.0
stagnation rate: 0.0476
delta_phi_raw == phi_future - phi_t: true
phi_t monotonic violation rate: 0.0
```

Combined `click_bell` Sim-SubSuccess smoke source:

```text
WAM episode root: /data/projects/WAM/data/episodes/robotwin_click_bell_subsuccess_smoke_v0
WAM windows:      /data/projects/WAM/data/prepared/robotwin_click_bell_subsuccess_smoke_v0
WAM features:     /data/projects/WAM/data/features/robotwin_click_bell_subsuccess_smoke_dinov2_vitb14_224
WAM audit:        /data/projects/WAM/outputs/audits/robotwin_click_bell_subsuccess_smoke_v0_potential_gain
```

Included episodes:

```text
expert_click_bell_episode0:     81 frames, 4 cameras
hesitation_click_bell_episode0: 93 frames, 4 cameras
detour_click_bell_episode0:     93 frames, 4 cameras
```

Prepared-window split:

```text
num windows: 120
train: expert_click_bell_episode0
val:   hesitation_click_bell_episode0
test:  detour_click_bell_episode0
seed:  5
```

Combined audit:

```text
delta_phi_raw min: 0.0000
delta_phi_raw max: 0.1432
negative gain rate: 0.0
stagnation rate: 0.0333
delta_phi_raw == phi_future - phi_t: true
phi_t monotonic violation rate: 0.0
```

Combined loader and one-epoch joint-flow smoke:

```text
config:
  configs/robotwin/joint_flow_click_bell_subsuccess_siglip_smoke.yaml
output:
  /data/projects/WAM/outputs/robotwin_click_bell_subsuccess_siglip_smoke_joint_flow/mvp1_joint_flow
loader sanity:
  train n=36, val n=42, test n=42
  obs_features [8, 4, 4, 768], future_obs_features [8, 8, 4, 768]
  action_chunk [8, 8, 30], prompt_features [8, 768]
test:
  delta_phi_mae 0.2761
  all-neg ranking 0.5190
  coarse ranking 0.6429
```

## Feature And Training-Input Smoke

Real frozen DINOv2 features were extracted on the 5060:

```text
features: /data/projects/WAM/data/features/robotwin_subsuccess_smoke_dinov2_vitb14_224
model:    vit_base_patch14_dinov2.lvd142m
device:   cuda
dim:      768
episodes: 4
```

Feature stores:

```text
expert_beat_block_hammer_episode0.npz:     117 frames x 4 cameras x 768
hesitation_beat_block_hammer_episode0.npz: 129 frames x 4 cameras x 768
detour_beat_block_hammer_episode0.npz:     129 frames x 4 cameras x 768
overshoot_beat_block_hammer_episode0.npz:  129 frames x 4 cameras x 768
```

Additional training-input artifacts:

```text
mock prompt features: /data/projects/WAM/data/prompts/robotwin_subsuccess_smoke_mock32/prompt_features.npz
mock prompt dim:      32
SigLIP prompt features:
  /data/projects/WAM/data/prompts/robotwin_subsuccess_smoke_siglip/prompt_features.npz
SigLIP model:         google/siglip-base-patch16-224
SigLIP prompt shape:  1 task x 768
SigLIP task_ids:      beat_block_hammer
norm stats:           /data/projects/WAM/data/prepared/robotwin_subsuccess_smoke_v0/norm_stats.json
train stats:          2 episodes, 258 frames
```

Batch shape checks passed for `PreparedWindowDataset` and
`JointFlowPreparedWindowDataset` on train/val/test:

```text
obs_features:        [B, 4, 4, 768]
future_obs_features: [B, 8, 4, 768]
proprio:             [B, 30]
proprio_history:     [B, 4, 30]
action_chunk:        [B, 8, 30]
prompt_features:     [B, 32]
```

The same loader and model-forward sanity passed with real SigLIP text features:

```text
config:              configs/robotwin/joint_flow_subsuccess_siglip_smoke.yaml
obs_features:        [16, 4, 4, 768]
future_obs_features: [16, 8, 4, 768]
proprio:             [16, 30]
proprio_history:     [16, 4, 30]
action_chunk:        [16, 8, 30]
prompt_features:     [16, 768]
model outputs:       v_obs [16, 8, 768], v_action [16, 8, 30], v_phi [16, 5]
single forward loss: 10.2000
```

One-epoch smoke runs:

```text
critic output:     /data/projects/WAM/outputs/robotwin_subsuccess_smoke_critic_input_check
critic checkpoint: best.pt written
critic metrics:    delta_phi_mae 0.1001, ranking_acc 1.0

joint-flow output:     /data/projects/WAM/outputs/robotwin_subsuccess_smoke_joint_flow_input_check
joint-flow checkpoint: best.pt written
joint-flow test:       delta_phi_mae 0.1078, all-neg ranking 0.3967

SigLIP joint-flow output:
  /data/projects/WAM/outputs/robotwin_subsuccess_siglip_smoke_joint_flow/mvp1_joint_flow
SigLIP joint-flow checkpoint: best.pt written
SigLIP joint-flow val:
  delta_phi_mae 0.1685, all-neg ranking 0.5407, coarse ranking 0.6296
SigLIP joint-flow test:
  delta_phi_mae 0.1752, all-neg ranking 0.6067, coarse ranking 0.7056
```

Reusable configs:

```text
configs/robotwin/joint_flow_subsuccess_smoke.yaml
configs/robotwin/joint_flow_subsuccess_siglip_smoke.yaml
configs/robotwin/joint_flow_click_bell_subsuccess_siglip_smoke.yaml
```

Config sanity:

```text
train batch obs_features:        [16, 4, 4, 768]
train batch future_obs_features: [16, 8, 4, 768]
train batch action_chunk:        [16, 8, 30]
train batch prompt_features:     [16, 32]
model outputs: v_obs [16, 8, 768], v_action [16, 8, 30], v_phi [16, 5]
single forward loss: 9.1837
```

## Tests

Executed on the 5060 WAM env:

```bash
/data/projects/WAM/.conda/wam/bin/python -m pytest \
  tests/test_import_robotwin.py \
  tests/test_potential_gain_audit.py \
  -q
```

Result:

```text
4 passed
```

Also executed:

```bash
/data/projects/WAM/.conda/wam/bin/python -m pytest \
  tests/test_prepare_windows_io.py \
  tests/test_labels.py \
  tests/test_features.py \
  -q
```

Result:

```text
22 passed, 1 warning
```

## Next Step

The current RoboTwin scaling plan is maintained in:

```text
docs/archive/reports/2026_07_16_docs_cleanup/ppwam_robotwin_subsuccess_data_plan.md
```

The first three simulator-produced Sim-SubSuccess variants now exist:

```text
hesitation / stagnation
detour / inefficient path
overshoot-and-correct
```

These are also combined into a small WAM prepared source:

```text
robotwin_subsuccess_smoke_v0
```

The current simulator-produced variants use:

```text
1. modified trajectory execution plus HDF5/video export
2. matched sidecar potential/boundary labels for WAM potential-gain training
```

Feature extraction and training-input checks now pass. Next work should decide
whether to keep this as a smoke-only source or scale the same replay pattern.
The smoke source now has real frozen SigLIP prompt features, and a second task
(`click_bell`) has expert, hesitation, and detour paths combined into its own
train/val/test source. The next scale-up question is coverage: more tasks,
seeds, and simulator-produced sub-success variants. A two-task training source
is now mechanically reachable with `merge_prepared_sources`, because both
`beat_block_hammer` and `click_bell` have non-empty train/val/test prepared
splits.
