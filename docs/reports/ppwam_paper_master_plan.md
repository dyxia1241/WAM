# PP-WAM 论文总指挥计划

## 0. 当前结论

这篇论文的主线不应只是“提出一个 PP-WAM 模型”，也不应只是“采一个 ARX 小数据集”。最强主线是：

```text
用 suboptimal-yet-successful 真实双臂数据，学习 action-conditioned primitive-local process potential，
并用这个 potential 做 candidate action selection，改善机器人执行过程。
```

当前代码和实验已经支持三点：

1. MVP0 证明 primitive-local potential 必须 action-conditioned，并且需要 counterfactual supervision。
2. MVP1.6 `cf_1p0` 证明 DiT-style joint flow 可以承载 future obs latent、action chunk、phi trajectory 的统一建模。
3. phi-only strong baseline 在 synthetic CF 和第一版 hard reranking 上仍然强于 joint-flow `cf_1p0`，所以当前不能声称 joint-flow 已经是更强 critic。

这意味着接下来论文工作的核心不是继续刷简单 ranking，而是回答：

```text
joint-flow 比 phi-only critic 多出来的 future latent/action flow 建模，到底在什么真实任务或推理模式下有价值？
```

## 1. 目标 venue 和论文定位

主投路线：

```text
ICRA 2027
```

原因：真实双臂数据集、真实机器人验证、candidate reranking、执行过程质量改善，更符合 ICRA 的系统和实验偏好。

冲刺路线：

```text
ICLR 2027
```

条件：9 月前必须形成强方法证据，包括 PP-WAM-Base 明显超过 strong critic baseline，semantic hard reranking 成立，ARX pilot 有足够规模，且 paper story 不依赖大量硬件细节。

## 2. 一句话主张

论文主张：

```text
PP-WAM learns a joint flow over future observation latents, future action chunks,
and primitive-local process potential, enabling the same model to predict consequences
and score candidate actions for process advancement.
```

中文解释：

```text
PP-WAM 把未来视觉 latent、未来动作 chunk、primitive-local process potential 放到同一个 flow field 里建模。
当不给定 future action 时，它是 predictor / policy；
当 candidate action 被 clamp 时，它是 critic / reranker。
```

数据集主张：

```text
ARX-SubSuccess provides real-world dual-arm successful trajectories with intentionally
suboptimal execution processes, enabling evaluation of whether models can distinguish
better and worse action segments before final success.
```

中文解释：

```text
ARX-SubSuccess 不追求规模，而追求过程信号：
所有轨迹最终成功，但执行过程有好坏差异。
这让我们能评估模型是否真的理解 action 对当前 primitive progress 的贡献。
```

## 3. 最终贡献

贡献必须收敛成四个，避免碎片化：

1. **PP-WAM 模型。**
   DiT-style typed-token joint flow，统一建模 future observation latent、future action chunk、primitive-local potential trajectory。

2. **Dual-mode inference。**
   同一个模型在 action unknown 时作为 predictor / policy，在 candidate action clamped 时作为 critic / reranker。

3. **ARX-SubSuccess 数据集。**
   真实双臂、全成功但过程次优的数据集，包含 video、action、proprioception、primitive boundary、suboptimality tags。

4. **Action selection utility。**
   用 PP-WAM 做 candidate reranking，在 offline benchmark 和真实机器人执行中减少 suboptimal behavior 或提高执行效率。

## 4. 当前证据和限制

当前 strongest joint-flow candidate：

```text
configs/gm100/joint_flow_cf1p0.yaml
```

当前 strong control：

```text
configs/gm100/phi_only_cf1p0.yaml
```

GM-100 synthetic CF 结果：

| model | DeltaPhi MAE | coarse ranking | all-neg ranking | coarse top-1 |
| --- | ---: | ---: | ---: | ---: |
| MVP1.6 `cf_1p0` | 0.0187+/-0.0027 | 0.8870+/-0.0231 | 0.7801+/-0.0132 | 0.7301+/-0.0543 |
| phi-only `cf1p0` | 0.0256+/-0.0137 | 0.9084+/-0.0186 | 0.7911+/-0.0334 | 0.7790+/-0.0670 |

第一版 hard reranking 结果：

| model | hard pairwise ranking | hard top-1 | margin to best neg |
| --- | ---: | ---: | ---: |
| MVP1.6 `cf_1p0` | 0.8336+/-0.0756 | 0.6391+/-0.1341 | 0.0104+/-0.0078 |
| phi-only `cf1p0` | 0.8795+/-0.0304 | 0.7248+/-0.0590 | 0.0222+/-0.0020 |

战术含义：

```text
不能 claim: joint-flow is already a stronger critic.
可以 claim: joint-flow is a viable world-action-potential formulation, but needs stronger evidence.
```

下一阶段必须证明以下至少一个：

1. joint-flow 在 semantic/base-policy candidate reranking 上超过 phi-only。
2. joint-flow 的 future latent consistency 能提升 action selection。
3. joint-flow 的 action-unknown predictor mode 有实际价值。
4. joint-flow 在真实 ARX suboptimal-yet-successful 数据上比 phi-only 更能区分过程质量。

## 5. 最终模型计划

### 5.1 输入输出

最终模型输入：

```text
language command / task prompt
observation history latent
proprioception history
past action history
optional candidate future action chunk
```

最终模型 flow tokens：

```text
future observation latent tokens
future action chunk tokens
primitive-local potential trajectory tokens
```

最终模型输出：

```text
v_obs
v_action
v_phi
```

其中：

```text
v_obs: future observation latent 的 flow velocity
v_action: future action chunk 的 flow velocity
v_phi: primitive-local potential trajectory 的 flow velocity
```

### 5.2 推理模式

Predictor / policy mode：

```text
不给 candidate action。
future action tokens 从 noise 或 previous-action initialization 开始。
模型生成 future action、future obs latent、potential trajectory。
```

Critic mode：

```text
给定 candidate action chunk。
action tokens clamp。
模型预测这个 action 对当前 primitive-local potential 的推进。
```

Reranking / MPC mode：

```text
给定 N 个 candidate action chunks。
对每个 action 用 critic mode 打分。
选择 potential score 最高且满足 smoothness / consistency 约束的 action。
```

### 5.3 Backbone scale

当前 small model：

```text
hidden_dim=192
layers=3
heads=4
phi_tokens=8
```

Base model，作为最终主模型候选：

```text
hidden_dim=384
layers=6
heads=8
phi_tokens=8
```

Large model，仅在 Base 稳定有效后再做：

```text
hidden_dim=512
layers=8
heads=8
phi_tokens=8 or 16
```

原则：

```text
先训练中等 DiT + 高质量 process supervision。
不要急着 RGB frame generation。
主线输出 observation latent，RGB reconstruction 只作为 appendix 或 qualitative demo。
```

### 5.4 Loss

基础 loss：

```text
L = L_obs_flow
  + L_action_flow
  + lambda_phi * L_phi_flow
  + lambda_critic * L_critic_flow
  + lambda_cf * L_counterfactual
  + optional L_calibration
```

必须保留的 ablation：

```text
remove future obs flow
remove action flow
remove phi trajectory
remove critic-flow loss
remove CF loss
replace joint flow with phi-only Transformer critic
small/base scale comparison
```

## 6. ARX-SubSuccess 数据集计划

### 6.1 数据集定位

ARX-SubSuccess 不与 DROID / Open X-Embodiment / OpenVLA 在规模上竞争。它的贡献是：

```text
真实双臂
所有轨迹最终成功
执行过程有意包含 suboptimal behaviors
可用于 process potential / action-conditioned progress evaluation
```

### 6.2 第一版规模

目标规模：

```text
Tasks: 8-12
Episodes: 每个任务 30-50
Total: 300-500 successful trajectories
Views: 2-3 RGB views, 至少 static + left wrist + right wrist 中的两类
Actions: dual-arm joint or EEF action + gripper
Proprio: dual-arm joint state / EEF pose / gripper state
Labels: primitive chain, primitive boundary, suboptimality tags, success flag
```

Pilot 规模：

```text
Tasks: 2-3
Episodes: 50-80
目标: 验证同步、标签、suboptimal pair construction 是否成立
```

### 6.3 任务选择

优先选择过程质量可区分的双臂任务：

```text
dual-arm pick-and-place
drawer open + retrieve + close
handover between arms
container lid open + object place
peg / tube insertion
tool insertion / keyhole-like task
cable / rope routing
box open / close
stacking / alignment
bimanual folding / flattening
```

### 6.4 Suboptimality taxonomy

每条成功轨迹可以带多个 tag：

```text
hesitation: 中途停顿或等待
overshoot: 越过目标后修正
wrong_approach: 从不合适方向接近
unnecessary_contact: 碰到桌面或无关物体
wrong_arm_attempt: 先用不合适手臂尝试
late_coordination: 双臂配合时序不佳
regrasp: 抓取失败后重新抓
detour: 路径绕远但最终成功
overcorrection: 来回修正
unstable_contact: 接触不稳定但最终完成
```

### 6.5 数据格式

建议每条 episode 存成：

```text
episode_id/
  metadata.json
  rgb_static.mp4
  rgb_left_wrist.mp4
  rgb_right_wrist.mp4
  actions.npy
  proprio.npy
  timestamps.json
  primitive_boundaries.json
  suboptimality_tags.json
  optional_contact_events.json
```

`metadata.json` 至少包含：

```text
episode_id
task_name
language_instruction
primitive_chain
success=true
robot_platform=ARX
operator_id
collection_date
notes
```

## 7. 实验矩阵

### Experiment A: GM-100 offline diagnostic

目的：

```text
证明 PP-WAM 机制成立，并和现有 MVP0 / phi-only control 对齐。
```

模型：

```text
MVP0 prompt critic
MVP0 stage/action critic
MVP1 V1 naive joint flow
MVP1.6 cf_1p0
phi-only strong baseline
PP-WAM-Base
```

指标：

```text
DeltaPhi MAE / RMSE
synthetic coarse ranking
all-neg ranking
hard reranking
calibration
per-negative ranking and margin
```

通过门槛：

```text
PP-WAM-Base 不一定必须在 synthetic coarse negative 上赢 phi-only，
但必须在 semantic hard reranking、future consistency、predictor utility 中至少赢一项。
```

### Experiment B: ARX-SubSuccess benchmark

目的：

```text
证明数据集提供独特的 process-quality supervision。
```

核心 pair：

```text
optimal segment > hesitation segment
direct path > detour segment
stable grasp/contact > regrasp segment
correct arm coordination > late coordination
correct primitive action > wrong-phase action
```

指标：

```text
suboptimal pairwise ranking
human preference agreement
primitive-local progress correlation
calibration within successful trajectories
held-out task generalization
held-out suboptimality type generalization
```

通过门槛：

```text
在同任务、同最终成功标签下，模型能稳定区分更好和更差的 action segment。
```

### Experiment C: Candidate reranking

目的：

```text
证明 potential 可以用于 action selection，而不是离线分数游戏。
```

候选 action 来源：

```text
expert future chunk
same-task wrong-phase chunk
suboptimal successful chunk
base-policy sampled chunk
perturbed chunk
zero / scaled / wrong_arm diagnostic chunk
```

指标：

```text
top-1 expert retrieval
top-k recall
selected action GT progress
regret: best candidate progress - selected candidate progress
margin to best negative
future latent consistency
```

通过门槛：

```text
PP-WAM reranking 必须超过 random、base policy score、phi-only critic 中至少一个强 baseline。
```

### Experiment D: Real robot validation

目的：

```text
形成 ICRA 需要的真实系统证据。
```

最小比较：

```text
base policy only
base policy + phi-only reranking
base policy + PP-WAM reranking
optional PP-WAM predictor mode
```

指标：

```text
success rate
completion time
number of corrective motions
suboptimal behavior count
manual intervention count
primitive progress per step
```

通过门槛：

```text
即使 success rate 不大幅提升，也必须减少 correction count、completion time 或 suboptimal behavior count。
```

## 8. 作战时间线

### Phase 1: 2026-07-03 到 2026-07-15

目标：

```text
冻结问题定义，准备 ARX pilot。
```

产出：

```text
paper outline v0
ARX-SubSuccess protocol v1
task list
suboptimality taxonomy
data schema
sync script plan
semantic/base-policy reranking eval design
```

决策门槛：

```text
如果仍说不清 joint-flow 为什么比 phi-only 必要，不扩模型。
```

### Phase 2: 2026-07-16 到 2026-07-31

目标：

```text
采集 ARX pilot data。
```

产出：

```text
2-3 tasks
50-80 successful trajectories
video/action/proprio sync check
primitive boundary initial labels
suboptimal pair construction
PP-WAM adaptation smoke
```

决策门槛：

```text
如果 ARX pilot 不能构造清晰 optimal-vs-suboptimal pairs，
先修任务设计和标签，不训练大模型。
```

### Phase 3: 2026-08-01 到 2026-08-20

目标：

```text
实现 PP-WAM-Base 和主实验第一版。
```

产出：

```text
Base DiT config
past action history tokens
semantic hard negatives
phi-only Base baseline
GM-100 + ARX pilot joint training
main ablation table draft
```

决策门槛：

```text
如果 PP-WAM-Base 仍全面输 phi-only，
joint-flow 降级为 exploratory，论文主线转向 process-potential dataset + strong critic / reranking。
```

### Phase 4: 2026-08-21 到 2026-09-10

目标：

```text
判断是否冲 ICLR 2027。
```

冲 ICLR 条件：

```text
PP-WAM-Base 在 semantic hard / ARX suboptimal / reranking 中赢 phi-only
ablation 能解释 joint-flow 的收益
ARX pilot 至少 150-250 trajectories
paper story 不依赖大量硬件系统细节
```

不满足：

```text
转 ICRA 主线，保留 ICLR workshop / arXiv 选项。
```

### Phase 5: 2026-09-11 到 2026-10-15

目标：

```text
扩充 ARX-SubSuccess 到 dataset contribution 级别。
```

产出：

```text
300-500 successful trajectories
8-12 tasks
dataset statistics
benchmark splits
annotation protocol
example videos
dataset section draft
```

### Phase 6: 2026-10-16 到 2026-11-20

目标：

```text
完成 downstream 和真实机器人证据。
```

产出：

```text
offline candidate reranking full table
base policy + PP-WAM reranking results
phi-only reranking comparison
real robot qualitative videos
failure case analysis
```

### Phase 7: 2026-11-21 到 2026-12-20

目标：

```text
完成 ICRA 版本论文闭环。
```

产出：

```text
method figure
dataset figure
main results tables
ablation tables
robot validation videos
code/data release plan
paper draft
```

## 9. 近期立即执行项

优先级 1：

```text
写 ARX-SubSuccess protocol v1。
内容包括 task list、数据格式、同步规则、suboptimal taxonomy、标注规则、splits。
```

优先级 2：

```text
扩展 semantic/base-policy reranking evaluator。
当前 hard reranking 只用 action-bank distractors。
下一版要支持 suboptimal successful chunk、base-policy sampled chunk、same-task same-stage lower-quality chunk。
```

优先级 3：

```text
设计 PP-WAM-Base config。
先不要训练，先写清楚 architecture、batch size、memory estimate、expected runtime。
```

优先级 4：

```text
准备 ARX pilot ingest。
先确保 episode schema 能进入现有 WAM data pipeline。
```

## 10. 风险和转向

风险 1：phi-only 持续强于 joint-flow。

应对：

```text
不要硬 claim joint-flow 更强。
把 joint-flow 的价值转向 predictor mode、future latent consistency、test-time reranking。
如果仍无收益，论文主线降级为 ARX process-potential benchmark + strong action-conditioned critic。
```

风险 2：ARX 数据太少，训练不稳定。

应对：

```text
先用 frozen visual/prompt encoder。
主模型用 Base DiT，不上 Large。
优先做 pairwise/reranking evaluation，不急着端到端 policy。
```

风险 3：suboptimal 标签主观。

应对：

```text
设计 taxonomy 和 annotation protocol。
使用 pairwise preference，而不是要求帧级绝对分数。
保留 human agreement 和 inter-annotator consistency。
```

风险 4：真实机器人 closed-loop 时间不够。

应对：

```text
先做 offline replay + candidate reranking。
再做小规模 real robot demo。
ICRA 版本至少要有定量 closed-loop 或半闭环证据。
```

## 11. Repo 执行规则

训练规则：

```text
所有正式训练在 5060 上执行。
本地 WSL 负责代码编辑、测试、文档和 git 同步。
```

数据和产物规则：

```text
data/
outputs/
checkpoints/
*.pt
*.npz
*.npy
```

保持 git ignored，不纳入版本控制。

文档规则：

```text
当前计划和报告放在 docs/reports/。
旧计划放 docs/archive/。
README 和 docs/README.md 只保留当前入口。
```

同步规则：

```text
每轮重要工作结束后：
1. local tests
2. commit
3. sync to 5060
4. 5060 tests if code changed
5. push GitHub
6. ensure local and 5060 git status clean
```

## 12. 最终判据

这篇论文成立的最低条件：

```text
1. 有一个清晰的 primitive-local process potential formulation。
2. 有 strong phi-only baseline，并诚实比较。
3. 有 ARX-SubSuccess 数据，证明 successful trajectories 内部有过程质量差异。
4. 有 candidate reranking 或真实机器人结果，证明 potential 对 action selection 有用。
```

这篇论文变强的条件：

```text
1. PP-WAM-Base 在 semantic/base-policy reranking 上超过 phi-only。
2. Predictor mode 或 future latent consistency 展示 joint-flow 的独特价值。
3. 真实机器人上减少 correction、hesitation、regrasp 或 completion time。
4. ARX-SubSuccess 有足够清楚的 benchmark splits 和 annotation protocol。
```

总指挥原则：

```text
不要为了维护 joint-flow 叙事而忽略 phi-only baseline。
不要为了追大模型而牺牲数据和评估质量。
不要只报离线 ranking，必须把 potential 连接到 action selection。
```
