# PP-WAM Paper Plan

Date: 2026-07-15

## 0. Main Decision

当前论文主线收窄为：

```text
Compact Process-Potential WAM for In-Domain Perturbation Recovery
```

一句话故事：

```text
当 nominal manipulation policy 在已知任务中出现 hesitation / detour / overshoot
这类非终止过程扰动时，PP-WAM 联合想象 future action、future observation latent
和 potential curve，并在测试时选择 predicted process recovery 最健康的 action。
```

这不是通用大规模 WAM 论文，也不是单纯 process scorer。主张应落在：

```text
process recovery
action-conditioned potential gain
tri-coupled imagined future
test-time reranking
controlled Sim-SubSuccess + real ARX validation
```

## 1. Problem Definition

论文问题定义：

```text
Given a nominal manipulation policy and an in-domain execution perturbation,
can a compact WAM predict which imagined future restores primitive progress
and select the corresponding action at test time?
```

中文表述：

```text
给定一个本来能完成任务的策略，当执行中发生同域过程扰动时，
能否用一个紧凑 WAM 判断哪条 action-conditioned future 能恢复任务进程？
```

必须避免的泛化承诺：

```text
不要声称解决通用新任务泛化。
不要声称训练通用 video-action foundation model。
不要把目标写成简单 task success prediction。
```

核心对象：

```text
F_i = (a^i_{t:t+K-1}, z^i_{t+1:t+K}, phi^i_{t:t+K})

a: future action chunk
z: future observation latent
phi: process potential curve
```

潜势不是后验 scalar head，而是 imagined future 的同步 token stream：

```text
Potential is the progress signature of an imagined action-conditioned future.
```

## 2. Contribution Stack

### C1. Controlled Sim-SubSuccess Data

RoboTwin 用作可控数据工厂，不作为最终真实贡献本身。当前已完成：

```text
20 tasks x 3 perturbations = 60 2x suboptimal-success episodes
rule_v1 potential sidecars
7494 prepared windows
60 DINOv2 feature files
```

权威数据细节维护在：

```text
docs/reports/ppwam_robotwin_subsuccess_data_plan.md
```

论文里只需要抽象为：

```text
expert replay
-> joint-path perturbation injection
-> SAPIEN replay
-> success filter
-> rule-based Phi / DeltaPhi_raw labels
```

三类 process signatures：

```text
hesitation: plateau / low slope
detour: inefficient path / mild local regression
overshoot: local regression then recovery
```

### C2. Tri-Coupled PP-WAM

模型不是大 backbone 后接 value head，而是 typed-token future model：

```text
condition:
  language / prompt
  observation history latent
  proprioception history

future tokens:
  action chunk
  future observation latent
  potential curve
```

当前 MVP 是小型 JointFlowDiT。投稿主模型不应停留在 smoke 规模，建议：

```text
PP-WAM-Base:
  hidden_dim = 384 or 512
  layers = 6 or 8
  heads = 8
  parameters ~= 15M-50M
```

Frozen encoder 路线：

```text
near-term:
  DINOv2 pooled baseline
  DINOv2 multi-view / patch-token ablation

next:
  video foundation latent, e.g. Wan VAE / video latent

not first step:
  full Wan-DiT LoRA / adapter
```

理由：

```text
大视频 DiT 可以提供视觉时序先验；
但当前核心瓶颈仍是 action-conditioned gain / ranking supervision，
不是简单把视觉 backbone 放大。
```

### C3. Process-Curve Reranking

推理时采样或接收多个 candidate futures：

```text
context c_t
  -> sample / query N futures F_i = (a_i, z_i, phi_i)
  -> score each future by curve health
  -> execute first h steps of selected action
  -> re-observe and replan
```

推荐 score：

```text
score(F_i) =
  DeltaPhi_i
  - lambda_r * regression_area_i
  - lambda_s * stagnation_duration_i
  - lambda_a * action_cost_i
  - lambda_u * uncertainty_i
```

`action_is_condition=True` 这条路径保留为：

```text
external candidate scoring
expert-vs-perturb ranking
generated future consistency rescore
```

但论文主叙事应是：

```text
joint future model with optional action-conditioned queries
```

而不是“两个 mode 学两个分布”。

## 3. Training Recipe

数据必须按用途分三类：

```text
expert / good:
  action loss high
  future obs loss high
  potential loss high
  ranking positive

suboptimal:
  action loss zero or very low
  future obs loss high if executed
  potential loss high
  ranking negative / hard negative

recovery:
  action loss high
  potential loss high
  ranking positive recovery
```

核心原则：

```text
坏动作教模型识别低 gain / regression future；
坏动作不能作为普通 BC 正样本被无条件模仿。
```

推荐 loss：

```text
L =
  L_action_good_or_recovery
  + lambda_z L_future_obs
  + lambda_phi L_potential_curve
  + lambda_delta L_delta_phi_raw
  + lambda_rank L_pairwise
```

pairwise 目标：

```text
DeltaPhi(c, a_positive) > DeltaPhi(c, a_negative) + margin
```

当前已知事实：

```text
只把 expert + suboptimal 混合做普通 flow loss，不会自然学出
direct > perturb preference。
```

因此 direct-vs-perturb / recovery-vs-bad ranking loss 是主线必需项，不是可选装饰。

## 4. Experiments

### Q1. Potential Curve Prediction

证明模型真的学到 process potential：

```text
Phi MAE
DeltaPhi_raw MAE
curve correlation
slope sign accuracy
regression detection F1
stagnation detection F1
```

按类型报告：

```text
expert
hesitation
detour
overshoot
policy rollout error
```

### Q2. Coupling Ablation

必须证明 tri-coupled WAM 不是 late-fusion value head：

```text
LateFusion-Value:
  encoder feature + scalar value / DeltaPhi head

Action-only:
  context -> action

ObsAction-WAM:
  context -> action + future obs latent

ActionPotential:
  context -> action + potential curve

Full PP-WAM:
  context -> action + future obs latent + potential curve

Full + potential-centric attention:
  phi tokens query action + future obs + context
```

关键比较：

```text
Full PP-WAM > LateFusion-Value
Full PP-WAM > ActionPotential
Full PP-WAM > ObsAction-WAM
```

如果 future observation latent 对 reranking 没贡献，WAM 叙事要降级为 process-potential
critic，不能硬讲 world model。

### Q3. Test-Time Reranking

候选动作来源：

```text
base-policy samples
expert-like candidate
hesitation-like candidate
detour-like candidate
overshoot-like candidate
recovery candidate
```

指标：

```text
top-1 high-DeltaPhi selection accuracy
selected candidate DeltaPhi_raw
suboptimal candidate selection rate
regression area reduction
stagnation ratio reduction
completion time
final success rate
```

### Q4. Real ARX Recovery

ICRA 版本需要真实闭环或至少强真实验证：

```text
minimum:
  3 tasks
  each with hesitation / detour / overshoot
  base policy vs PP-WAM rerank
  >= 100 real trials if possible

fallback:
  offline real ranking + 1-2 online recovery demos
```

真实评估指标：

```text
success rate
completion time
number of corrections
regression area
stagnation duration
manual intervention count
```

## 5. Baselines

必须有强 baseline，否则方法容易被认为只是工程组合：

```text
Time / monotonic prior
LateFusion scalar potential head
Phi-only action-conditioned critic
ObsAction WAM without potential curve
ActionPotential without future obs
RoVer-style scalar PRM verifier
WVM-style trajectory value / task progression head
Base policy without reranking
```

其中 `Phi-only critic` 是最重要 baseline：

```text
任何 PP-WAM 优势都必须超过强 action-conditioned process critic，
否则 joint future obs/action/potential 的复杂度没有论文价值。
```

## 6. Venue Strategy

### ICRA-First

ICRA 更现实，因为主线是机器人执行过程恢复：

```text
RoboTwin controlled perturbation data
compact PP-WAM
test-time reranking
real ARX recovery
videos and failure analysis
```

最低可投线：

```text
RoboTwin:
  >= 20 tasks controlled perturbation data is now available
  held-out perturbation location / task / visual condition still needed

ARX:
  >= 3 tasks
  base vs PP-WAM rerank
  clear improvement in recovery metric or manual intervention

Model:
  Full PP-WAM beats LateFusion and Phi-only on recovery/reranking metrics
```

### ICLR-Compatible

ICLR 需要更强方法普适性：

```text
held-out task / perturbation / visual domain
strong scalar PRM / WVM-style baselines
clear coupling ablation
open Sim-SubSuccess generator and labels
```

如果真实闭环强、方法普适性一般，优先 ICRA。  
如果 tri-coupled future modeling 在 held-out settings 上显著强于 value/PRM baselines，
再考虑 ICLR。

## 7. Paper Structure

推荐标题：

```text
Recovering Manipulation Processes with Compact Process-Potential World Action Models
```

结构：

```text
1. Introduction
   nominal policy 的过程扰动问题；final success 不足以描述 process quality。

2. Related Work
   WAM / video-action models, robot value/PRM, simulated manipulation data.

3. Problem Formulation
   in-domain perturbation recovery; F=(a,z,phi); curve-health score.

4. Data Generation
   RoboTwin controlled Sim-SubSuccess; ARX validation protocol.

5. Method
   frozen encoders; compact typed-token DiT; tri-coupled future tokens.

6. Training
   expert/suboptimal/recovery roles; potential curve; pairwise ranking.

7. Experiments
   curve prediction, coupling ablation, test-time reranking, real ARX recovery.

8. Limitations
   controlled perturbation is not full policy error distribution; large video prior is optional.
```

## 8. Immediate Next Milestones

按照当前状态，下一步不是继续造更多 RoboTwin task，而是把 20-task 数据变成论文证据：

```text
M1. 训练 PP-WAM-Base on robotwin_20task_2x_rule_subsuccess_v1
M2. 做 LateFusion / Phi-only / ObsAction / Full PP-WAM ablation
M3. 实现 direct-vs-perturb pairwise training loss
M4. 做 held-out task 和 held-out perturbation-location split
M5. 设计 base-policy candidate reranking benchmark
M6. 开始 ARX 3-task recovery pilot
```

当前 20-task 数据已经满足数据工厂第一阶段：

```text
RoboTwin controlled process signatures: complete
Training objective and recovery evaluation: next bottleneck
```
