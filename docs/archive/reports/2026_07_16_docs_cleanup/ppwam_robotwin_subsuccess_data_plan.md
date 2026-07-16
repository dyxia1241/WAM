# PP-WAM RoboTwin Sim-SubSuccess 2x 数据计划

Date: 2026-07-14

## 0. 当前决策

后续 RoboTwin 数据主线统一使用 `2x` replay 版本。早期 `1x` replay 只保留为
bring-up / smoke 历史记录，不再作为当前训练和论文展示的主数据。

归档记录：

```text
docs/archive/reports/2026_07_14_robotwin_1x_smoke/ppwam_robotwin_smoke_import_report.md
/data/projects/RoboTwin/data_archive/2026_07_14_robotwin_1x_smoke/
/data/projects/WAM/data/archive/2026_07_14_robotwin_1x_smoke/
```

当前主数据方向：

```text
RoboTwin scripted expert + planner seed
  -> 修改 _traj_data joint path
  -> SAPIEN replay 生成 2x HDF5 / video
  -> 外置 label JSON 标注全局 process Phi
  -> WAM import / prepare_windows / audit / training
```

## 1. 论文定位

RoboTwin 不是最终真实数据集贡献，而是 Sim-SubSuccess 数据工厂。它负责快速、
可控地产出：

```text
成功但过程质量不同的轨迹
可解释的 stagnation / detour / regression 区间
signed DeltaPhi_raw 训练标签
```

这服务 PP-WAM 的论文主线：

```text
学习一个可比较的 process potential Phi，
再学习 action-conditioned transition gain DeltaPhi(c, a)，
用 predicted gain 做 action selection / reranking。
```

真实机器人贡献仍应落在 ARX-SubSuccess。RoboTwin 负责预训练、消融、标注协议
验证和可视化案例。

## 1.1 对时悦 critique 的战术回应

时悦的几个担心应该吸收进 proposal，但不能把当前执行主线打散。战术判断是：

```text
1. 可以利用主流 frozen backbone / representation，
   但不能把方法退化成大模型后接一个 late-fusion potential head。

2. 人工扰动不是为了覆盖真实错误全集，
   而是作为 controllable process signatures:
     hesitation -> stagnation / plateau
     detour -> inefficient path / mild regression
     overshoot -> local regression + recovery

3. 坏动作不能进入普通 BC imitation loss。
   它们用于 Phi / DeltaPhi_raw / ranking / consequence learning。

4. Recovery action 和 suboptimal action 必须分开。
   overshoot 动作本身是 negative / low-gain；
   overshoot 后 correction 才是 positive recovery。

5. Potential 应是 imagined future 的 first-class token，
   和 future action / future observation latent 联合建模，
   不是 action 生成后的后验 scalar 打分。
```

对应训练协议：

```text
expert / high-quality segment:
  use_for_action_bc = true
  use_for_value = true
  use_for_ranking_positive = true

suboptimal segment:
  use_for_action_bc = false
  use_for_value = true
  use_for_ranking_negative = true
  use_for_consequence = true

recovery segment:
  use_for_action_bc = true
  use_for_value = true
  use_for_ranking_positive = true
```

当前 RoboTwin controlled perturbation 线只兑现第一阶段：

```text
Controlled Sim-SubSuccess
  -> rule-based Phi sidecar
  -> signed DeltaPhi_raw
  -> dataloader / joint-flow signed target
  -> audit / ranking 验证 bad action 是低 gain 或 negative gain
```

Policy rollout 和 recovery pairs 是第二阶段：

```text
Policy Rollout Mixed-Quality
  base policy success / suboptimal success / recoverable failure / terminal failure
  bad state -> recovery action pairs
```

因此，短期不切换到大 VLA 主线，不大规模 rollout；先把 controlled
perturbation 到 signed process gain 的闭环打通。

## 2. 不再强调语义阶段

当前阶段不需要把每条轨迹强行拆成 `approach / grasp / move / release` 等语义
阶段。我们先采用更稳的定义：

```text
expert trajectory:
  近似单调的全局 process Phi: 0 -> 1

perturbed trajectory:
  在干扰区间改变 Phi 曲线形状
```

也就是说，当前主监督来自 frame-level `potential`，不是来自 stage label。

实现上，WAM schema 仍要求 `primitive_boundaries`，所以 2x sidecar 暂时使用单一
全局 boundary：

```json
[
  {"stage": "move", "start": 0, "end": 236}
]
```

这个 boundary 只是格式占位；真正的训练信号是 `potential[t]`。

## 3. 三类 2x 改造

### 3.1 Hesitation

改造方式：

```text
插入长时间重复 joint rows，velocity 接近 0。
```

Phi 标注：

```text
干扰前正常增长
停顿区间 plateau
停顿后继续增长到 1.0
```

语义：

```text
没有回退，但过程停滞。
DeltaPhi_raw ~= 0。
```

### 3.2 Detour

改造方式：

```text
插入 out-and-back joint-space / EEF-space loop。
```

Phi 标注：

```text
进入 detour 前正常增长
detour 中先小幅下降
detour 后恢复增长
```

语义：

```text
低效路径，可能短时远离当前 subgoal，但最终回到 expert path。
DeltaPhi_raw 可以轻微为负。
```

### 3.3 Overshoot And Correct

改造方式：

```text
在接近目标或关键位姿后越过，再修正回来。
```

Phi 标注：

```text
overshoot 前增长到较高 Phi
overshoot 中出现 regression dip
correction 后恢复并到达 1.0
```

语义：

```text
真实回退再恢复。
DeltaPhi_raw < 0 应在 audit 中可见。
```

## 4. Phi / DeltaPhi 训练定义

状态 potential：

```text
Phi_t in [0, 1]
```

Action-conditioned gain：

```text
DeltaPhi_raw = Phi_{t+K} - Phi_t
```

关键规则：

```text
Phi 可以 clamp 到 [0, 1]。
DeltaPhi_raw 不应 clip 到非负。
```

解释：

```text
DeltaPhi_raw > 0: progress
DeltaPhi_raw = 0: stagnation
DeltaPhi_raw < 0: regression
```

当前代码里的：

```text
delta_phi = max(0, delta_phi_raw)
```

只应继续作为 legacy compatibility 字段。Sim-SubSuccess 主训练目标应迁移到 signed
`delta_phi_raw`。

## 5. 当前 2x beat_block_hammer 标注

当前已完成三条 2x replay：

```text
ppwam_hesitation_2x_v1: 237 frames
ppwam_detour_2x_v1:     237 frames
ppwam_overshoot_2x_v1:  237 frames
```

对应 sidecar：

```text
/data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_hesitation_2x_v1_sidecar.json
/data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_detour_2x_v1_sidecar.json
/data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_overshoot_2x_v1_sidecar.json
```

当前 Phi anchors：

```text
hesitation:
  (0, 0.00), (78, 0.34), (197, 0.34), (236, 1.00)

detour:
  (0, 0.00), (17, 0.14), (80, 0.09), (137, 0.28), (236, 1.00)

overshoot:
  (0, 0.00), (138, 0.72), (180, 0.56), (228, 0.86), (236, 1.00)
```

图像：

```text
docs/figures/current/robotwin_2x_potential/beat_block_hammer_hesitation_2x_v1_potential.png
docs/figures/current/robotwin_2x_potential/beat_block_hammer_detour_2x_v1_potential.png
docs/figures/current/robotwin_2x_potential/beat_block_hammer_overshoot_2x_v1_potential.png
docs/figures/current/robotwin_2x_potential/beat_block_hammer_2x_variants_potential_combined.png
```

## 5.1 Rule-based Phi 标注版本

2026-07-14 额外生成了一套不覆盖原手工 anchor 的 rule-based sidecar：

```text
/data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_hesitation_2x_v1_rule_v1_sidecar.json
/data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_detour_2x_v1_rule_v1_sidecar.json
/data/projects/WAM/data/robotwin_sidecars/beat_block_hammer_overshoot_2x_v1_rule_v1_sidecar.json
```

规则输入：

```text
RoboTwin _traj_data/episode0.pkl dense planner joint path
HDF5 joint_action/right_arm replay path
reconstructed expert path from shortest common segment
```

规则逻辑：

```text
1. 用最短共同 segment 重建 expert joint path。
2. 将 replay joint path 投影到 expert path，得到 nearest expert progress 和 deviation。
3. 用 dense path segment length excess 自动识别被扰动的 segment。
4. hesitation: 用 nearest progress 的最长平台定位 stagnation interval。
5. detour: 用 off-expert-path deviation 的 union interval 标注先降后升。
6. overshoot: 用后段 off-expert-path deviation peak 标注 regression dip 和恢复。
```

当前 rule anchors：

```text
hesitation:
  (0, 0.00), (80, 0.339), (198, 0.339), (236, 1.00)

detour:
  (0, 0.00), (28, 0.119), (98, 0.029), (128, 0.262), (236, 1.00)

overshoot:
  (0, 0.00), (138, 0.585), (191, 0.424), (228, 0.886), (236, 1.00)
```

对比图：

```text
docs/figures/current/robotwin_2x_rule_potential/beat_block_hammer_hesitation_2x_v1_manual_vs_rule_phi.png
docs/figures/current/robotwin_2x_rule_potential/beat_block_hammer_detour_2x_v1_manual_vs_rule_phi.png
docs/figures/current/robotwin_2x_rule_potential/beat_block_hammer_overshoot_2x_v1_manual_vs_rule_phi.png
docs/figures/current/robotwin_2x_rule_potential/beat_block_hammer_2x_manual_vs_rule_phi_combined.png
```

## 5.2 当前 2x click_bell 标注

2026-07-14 已完成 `click_bell` 三条 2x replay：

```text
ppwam_hesitation_2x_v1: 161 frames
ppwam_detour_2x_v1:     161 frames
ppwam_overshoot_2x_v1:  161 frames
```

对应 rule-based sidecar：

```text
/data/projects/WAM/data/robotwin_sidecars/click_bell_hesitation_2x_v1_rule_v1_sidecar.json
/data/projects/WAM/data/robotwin_sidecars/click_bell_detour_2x_v1_rule_v1_sidecar.json
/data/projects/WAM/data/robotwin_sidecars/click_bell_overshoot_2x_v1_rule_v1_sidecar.json
```

生成方式：

```text
source: /data/projects/RoboTwin/data_archive/2026_07_14_robotwin_1x_smoke/click_bell/ppwam_smoke_clean
2x active:
  /data/projects/RoboTwin/data/click_bell/ppwam_hesitation_2x_v1
  /data/projects/RoboTwin/data/click_bell/ppwam_detour_2x_v1
  /data/projects/RoboTwin/data/click_bell/ppwam_overshoot_2x_v1
```

扰动方式：

```text
hesitation:
  在 left_joint_path[1] press segment 插入 1200 个重复 joint rows。

detour:
  在 left_joint_path[0] approach segment 插入 1200 个 out-and-back joint-space loop rows。

overshoot:
  在 left_joint_path[1] press segment 后段插入 1200 个沿 press direction 的 overshoot-and-correct rows。
```

rule inputs:

```text
RoboTwin _traj_data/episode0.pkl dense left-arm planner joint path
HDF5 joint_action/left_arm replay path
reconstructed expert path from shortest common segment
```

当前 rule anchors：

```text
hesitation:
  (0, 0.00), (66, 0.413), (144, 0.413), (160, 1.00)

detour:
  (0, 0.00), (22, 0.138), (55, 0.061), (88, 0.330), (160, 1.00)

overshoot:
  (0, 0.00), (78, 0.488), (106, 0.388), (134, 0.778), (160, 1.00)
```

图像：

```text
docs/figures/current/robotwin_click_bell_2x_rule_potential/click_bell_hesitation_2x_v1_rule_phi.png
docs/figures/current/robotwin_click_bell_2x_rule_potential/click_bell_detour_2x_v1_rule_phi.png
docs/figures/current/robotwin_click_bell_2x_rule_potential/click_bell_overshoot_2x_v1_rule_phi.png
docs/figures/current/robotwin_click_bell_2x_rule_potential/click_bell_2x_rule_phi_combined.png
```

旧 1x active 产物已归档：

```text
/data/projects/RoboTwin/data_archive/2026_07_14_robotwin_1x_smoke/click_bell/
/data/projects/WAM/data/archive/2026_07_14_robotwin_1x_smoke/click_bell_1x_archive_manifest.txt
/data/projects/WAM/outputs/archive/2026_07_14_robotwin_1x_smoke/audits/
```

## 6. 相机和视频

RoboTwin 默认 `episode0.mp4` 是 head-camera preview。HDF5 中实际有多路相机：

```text
front_camera
head_camera
left_camera
right_camera
```

WAM importer 会读取 HDF5 中所有带 `rgb` 的 camera group。训练特征不是单 head
camera。后续如果要人工检查视频，应生成 2x2 multi-camera grid mp4。

## 7. 下游训练计划

### 7.1 Dataloader

已让 dataloader 返回：

```text
phi_t
phi_future
delta_phi_raw
delta_phi  # legacy clipped field
```

### 7.2 Joint-flow target

已给 joint-flow 增加 target 配置：

```yaml
model:
  phi_target_kind: delta_phi_raw
```

同时保留旧配置：

```yaml
model:
  phi_target_kind: legacy_delta_phi
```

### 7.3 Loss

推荐目标：

```text
L = L_abs_phi
  + lambda_gain * L_signed_delta_phi_raw
  + lambda_rank * L_pairwise_gain_ranking
  + lambda_obs * L_future_obs_consequence
  + lambda_action * L_future_action
```

优先级：

```text
1. 先跑 signed delta_phi_raw smoke。
2. 再加 Phi_t / Phi_future auxiliary loss。
3. 再加 segment-level pairwise ranking。
```

## 8. 任务扩展

当前可用：

```text
beat_block_hammer 2x: hesitation + detour + overshoot
click_bell 2x: hesitation + detour + overshoot
```

下一步优先：

```text
1. 补 direct / expert paired replay，给 ranking 提供同 context good action。
2. 扩展到 4-6 个稳定任务。
```

候选 Tier 1：

```text
press_stapler
stack_blocks_two 或 stack_blocks_three
place_object_basket 或 place_object_scale
handover_block 或 handover_mic
```

暂缓：

```text
wrong-arm attempt
late coordination
regrasp
unnecessary contact
near collision
```

## 9. 近期执行清单

1. 用 2x sidecar 导入：

```text
robotwin_2task_2x_rule_subsuccess_v1
```

2. 跑：

```text
prepare_windows
potential_gain_audit
extract_vision_features
norm_stats
loader / forward sanity
```

3. 修改 WAM dataset / joint-flow：

```text
支持 signed delta_phi_raw target
保留 legacy delta_phi target
```

4. 跑对比：

```text
legacy_delta_phi vs signed_delta_phi_raw
```

5. 设计 policy rollout / recovery pair 数据格式。

## 10. v1 成功标准

```text
1. 至少 2 个 task 有 hesitation + detour + overshoot 2x replay。
2. 每条 replay 有 HDF5、head-camera preview、多相机 image import。
3. Phi audit 能看到 stagnation / mild negative / regression 分布。
4. joint-flow 能训练 signed delta_phi_raw。
5. counterfactual ranking 能区分 good action chunk 和明显坏 action chunk。
6. direct / expert paired ranking 作为 v2 成功标准，不混入当前 v1。
```

## 11. 2026-07-14 执行状态

已完成 2 task x 3 perturbation 的 2x rule 数据闭环：

```text
episodes:
  /data/projects/WAM/data/episodes/robotwin_2task_2x_rule_subsuccess_v1

prepared windows:
  /data/projects/WAM/data/prepared/robotwin_2task_2x_rule_subsuccess_v1
  570 windows, history=4, horizon=8, stride=2

real vision features:
  /data/projects/WAM/data/features/robotwin_2task_2x_rule_subsuccess_v1_dinov2_vitb14_224

variant audit:
  /data/projects/WAM/outputs/audits/robotwin_2task_2x_rule_subsuccess_v1_variant_gain

signed-gain joint-flow smoke:
  /data/projects/WAM/outputs/robotwin_2task_2x_rule_dinov2_signed_unclamped_smoke_joint_flow_20ep/mvp1_joint_flow
```

关键 audit 结果：

```text
overall:
  delta_phi_raw_min = -0.02857
  delta_phi_raw_mean = 0.03931
  negative_rate = 14.74%
  stagnation_rate = 15.79%

hesitation:
  beat_block_hammer stagnation_rate = 48.25%
  click_bell stagnation_rate = 46.05%

detour / overshoot:
  beat_block_hammer detour negative_rate = 28.95%
  beat_block_hammer overshoot negative_rate = 21.93%
  click_bell detour negative_rate = 18.42%
  click_bell overshoot negative_rate = 15.79%
```

关键模型 smoke 结果：

```text
config:
  model.phi_target_kind = delta_phi_raw
  score.clamp defaults to false for signed delta_phi_raw

test:
  delta_phi_mae = 0.02848
  all_negatives_tie_aware_ranking_acc = 0.78596
  coarse_action_cf_ranking_acc = 0.95322
  zero_ranking_acc = 0.96491
```

注意：这个结果证明 signed-gain + real visual feature 的训练/评估链路已经打通，并且
能识别合成 counterfactual bad chunks。它还不能替代 direct expert vs perturbed action
的同 context ranking；后者需要补 direct/expert paired replay 或 policy candidate pairs。

## 12. Policy Rollout / Recovery Pair 下一阶段

下一阶段不是把所有 rollout 都拿来 BC，而是把 rollout 切成三类用途：

```text
expert / direct:
  use_for_action_bc = true
  use_for_value = true
  ranking_role = positive

suboptimal action:
  use_for_action_bc = false
  use_for_value = true
  ranking_role = negative

recovery action:
  use_for_action_bc = true
  use_for_value = true
  ranking_role = positive_recovery
```

建议先做最小闭环：

```text
1. 对每个 task 保留当前 2x hesitation / detour / overshoot。
2. 为同一 seed/context 额外导入 direct/expert replay。
3. 从相同 t 或相邻 progress bin 构造 pair:
     direct action > hesitation action
     direct action > detour action
     recovery action > overshoot action
4. policy rollout 只先收集 1 个 base policy 的 success / suboptimal success / failure。
5. 失败后的 recovery segment 单独标注，不把 failure action 送入普通 BC。
```

数据字段建议：

```json
{
  "segment_quality": "suboptimal | recovery | expert",
  "use_for_action_bc": false,
  "use_for_value": true,
  "ranking_group_id": "same_task_same_progress_bin",
  "ranking_role": "negative",
  "source_policy": "scripted | base_policy | ppwam_rerank",
  "potential_label_source": "rule_privileged_state"
}
```

## 13. 2026-07-14 Expert Replay 补充

已补齐同 seed direct expert replay。注意：expert/direct 轨迹不强行拉成 2x；2x 是为了让
扰动过程肉眼可见，direct expert 应保留高效轨迹长度，再通过 progress bin / nearest
`Phi_t` 做配对。

来源：

```text
beat_block_hammer:
  /data/projects/RoboTwin/data_archive/2026_07_14_robotwin_1x_smoke/beat_block_hammer/ppwam_smoke_clean
  seed = 1
  frames = 117

click_bell:
  /data/projects/RoboTwin/data_archive/2026_07_14_robotwin_1x_smoke/click_bell/ppwam_smoke_clean
  seed = 0
  frames = 81
```

WAM 导入：

```text
expert only:
  /data/projects/WAM/data/episodes/robotwin_2task_expert_direct_v1

expert + perturb combined:
  /data/projects/WAM/data/episodes/robotwin_2task_expert_plus_2x_rule_v1
  /data/projects/WAM/data/prepared/robotwin_2task_expert_plus_2x_rule_v1_expert_train_split
  /data/projects/WAM/data/features/robotwin_2task_expert_plus_2x_rule_v1_dinov2_vitb14_224
```

Expert sidecar：

```text
data/robotwin_sidecars/beat_block_hammer_expert_direct_v1_sidecar.json
data/robotwin_sidecars/click_bell_expert_direct_v1_sidecar.json
```

Expert potential 使用 linear monotonic `Phi: 0 -> 1`。Combined audit：

```text
beat_block_hammer expert:
  windows = 54
  delta_phi_raw_mean = 0.06881
  negative_rate = 0
  stagnation_rate = 0

click_bell expert:
  windows = 36
  delta_phi_raw_mean = 0.09965
  negative_rate = 0
  stagnation_rate = 0
```

对比 perturb 均值：

```text
beat_block_hammer perturb mean gain:
  detour     0.03366
  hesitation 0.03282
  overshoot  0.03307

click_bell perturb mean gain:
  detour     0.04956
  hesitation 0.04631
  overshoot  0.04964
```

也就是说，标签层面 expert direct 已经明显高于扰动版本。

## 14. Expert-Perturb Paired Ranking 现状

新增 evaluator：

```text
ppwam/robotwin_expert_pair_ranking.py
```

配对方式：

```text
对每个 perturb window:
  找 same task 下 nearest Phi_t 的 expert window
  比较 expert action chunk 和 perturb logged action chunk
```

已跑两个 context：

```text
perturb context:
  outputs/robotwin_2task_expert_plus_2x_rule_signed_joint_flow_20ep/mvp1_joint_flow/expert_pair_ranking

expert context:
  outputs/robotwin_2task_expert_plus_2x_rule_signed_joint_flow_20ep/mvp1_joint_flow/expert_pair_ranking_expert_context
```

标签层面：

```text
overall_gt_pairwise_acc = 0.90
overall_gt_delta_margin_mean = 0.04193
```

当前模型层面仍不稳定：

```text
perturb context:
  overall_model_pairwise_acc = 0.32105

expert context:
  overall_model_pairwise_acc = 0.32982
```

局部能分对：

```text
beat_block_hammer detour:
  expert context model_pairwise_acc = 0.70614

click_bell detour:
  expert context model_pairwise_acc = 0.75658
```

但 hesitation / overshoot 的 paired preference 还没有学起来。结论是：

```text
数据和标签已经支持 direct > perturbed；
当前 joint-flow smoke 的 counterfactual ranking loss 还不足以学会 expert-vs-perturb paired preference。
```

下一步应把 expert-pair rows 变成训练 loss，而不是只做 eval：

```text
L_pair_direct =
  -log sigmoid(score(c, expert_action) - score(c, perturb_action) - margin)

其中 c 可以分别尝试:
  expert context
  perturb context
  shared nearest-progress context / latent interpolation
```

## 15. No-Extra-Loss Mixed Training Check

按“只把 expert 和 suboptimal 放在一起训练，不加奇怪 loss”的设置跑了一版：

```text
checkpoint:
  /data/projects/WAM/outputs/robotwin_2task_expert_plus_2x_rule_signed_joint_flow_no_extra_loss_50ep/mvp1_joint_flow/best.pt

data:
  expert + 2x perturb combined

loss:
  obs_weight = 1.0
  action_weight = 1.0
  phi_weight = 5.0
  critic_flow_weight = 0
  counterfactual_weight = 0

train:
  max_epochs = 50
  action_condition_prob = 0.75
```

也就是说，这版没有 counterfactual ranking loss，也没有 direct-vs-perturb pairwise loss；
只用原本 joint-flow 的 obs/action/phi flow loss。

Expert-pair ranking:

```text
perturb context:
  overall_num_pairs = 570
  overall_gt_pairwise_acc = 0.90
  overall_model_pairwise_acc = 0.42105
  overall_model_margin_mean = -0.00102

expert context:
  overall_num_pairs = 570
  overall_gt_pairwise_acc = 0.90
  overall_model_pairwise_acc = 0.42281
  overall_model_margin_mean = -0.00093
```

局部分项：

```text
detour 比较容易:
  beat_block_hammer detour pairwise_acc ~= 0.59
  click_bell detour pairwise_acc ~= 0.95

hesitation / overshoot 仍然没有学起来:
  hesitation pairwise_acc 很低
  overshoot 只有 beat_block_hammer 稍高于随机，click_bell 仍低
```

结论：

```text
只混合 expert + suboptimal 数据做普通 flow training，
不会自然学出 direct action > perturb action 的 pairwise preference。

这不是标签问题；标签层面 gt_pairwise_acc = 0.90。
问题在训练目标没有直接约束同 progress bin 下的相对 action gain。
```

## 16. 20-task 2x Rule Expansion

2026-07-14 已把 RoboTwin Sim-SubSuccess 从 2 个任务扩到 20 个任务。每个
task 都有三类 2x perturbation：

```text
hesitation
detour
overshoot
```

任务列表：

```text
beat_block_hammer
click_bell
click_alarmclock
press_stapler
turn_switch
stamp_seal
open_laptop
open_microwave
move_pillbottle_pad
move_stapler_pad
move_can_pot
place_empty_cup
place_shoe
place_container_plate
place_object_scale
place_object_stand
place_phone_stand
stack_blocks_two
stack_bowls_two
handover_block
```

RoboTwin 侧产物：

```text
/data/projects/RoboTwin/data/<task>/ppwam_hesitation_2x_v1/
/data/projects/RoboTwin/data/<task>/ppwam_detour_2x_v1/
/data/projects/RoboTwin/data/<task>/ppwam_overshoot_2x_v1/
```

每条 variant 均验证有：

```text
data/episode0.hdf5
video/episode0.mp4
_traj_data/episode0.pkl
*_edit_summary.json
```

WAM 侧 rule sidecar：

```text
/data/projects/WAM/data/robotwin_sidecars/<task>_<kind>_2x_v1_rule_v1_sidecar.json
```

WAM 统一数据目录：

```text
episodes:
  /data/projects/WAM/data/episodes/robotwin_20task_2x_rule_subsuccess_v1

prepared windows:
  /data/projects/WAM/data/prepared/robotwin_20task_2x_rule_subsuccess_v1

real DINOv2 features:
  /data/projects/WAM/data/features/robotwin_20task_2x_rule_subsuccess_v1_dinov2_vitb14_224

audit:
  /data/projects/WAM/outputs/audits/robotwin_20task_2x_rule_subsuccess_v1_gain
```

批处理工具：

```text
tools/robotwin_make_2x_variants.py
tools/robotwin_rule_sidecars.py
tools/robotwin_import_20task_2x.py
```

执行摘要：

```text
RoboTwin variants:
  20 tasks x 3 perturbations = 60 episodes

WAM import:
  num_ok = 60 / 60

prepared windows:
  7494 windows

DINOv2 feature files:
  60 / 60

HDF5 / MP4 / sidecar / Phi length validation:
  60 / 60 pass
```

Potential audit：

```text
delta_phi_raw_min = -0.04663
delta_phi_raw_max = 0.68605
delta_phi_raw_mean = 0.03021
delta_phi_raw_negative_rate = 14.96%
delta_phi_raw_stagnation_rate = 15.08%
delta_phi_raw_positive_rate = 69.96%
phi_consistency_abs_error_max = 0
```

唯一需要特殊处理的是 `open_microwave / hesitation`。默认 hesitation 插在较早
motion segment 时 replay 会保存 HDF5/MP4，但最终 `check_success` assert 失败。
已用更保守的 `final_segment` hesitation 策略重跑该 variant，返回码为 0，并重新
生成对应 rule sidecar 和 DINOv2 feature。

当前这批 20-task 数据仍沿用 rule-v1 generic potential：

```text
expert-like global trajectory: near-monotonic 0 -> 1
hesitation: inserted interval becomes plateau / low-slope stagnation
detour: inserted interval has local dip then recovery
overshoot: inserted interval has stronger local regression then recovery
```

注意：这批扩展完成的是 controlled perturbation 数据工厂，不代表真实 policy
failure distribution 已覆盖。后续用于论文主线时，应继续保留定位：

```text
Sim-SubSuccess:
  scalable controllable process signatures

ARX / policy rollout:
  real process validation and compositional error distribution
```
