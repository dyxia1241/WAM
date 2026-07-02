# PP-WAM 面向 ICRA / ICLR 2027 的规划草案

状态：planning draft  
目标：把 MVP0 的 primitive-local potential critic 结果，升级为可投稿主线的 latent joint flow world-action model。  
默认路线：ICRA 2027 first，ICLR 2027 compatible。

## 1. 总体判断

新主线建议收束为：

```text
PP-WAM: Primitive-Progress World-Action Model
```

核心不是训练一个大型 RGB video-action foundation model，而是在有限算力下训练一个 latent-space joint flow model：

```text
language command + observation history + proprioception
    -> future observation latent + future action chunk + primitive-local process potential
```

它要统一四种功能：

```text
predictor / policy / critic / reranker
```

同一个模型通过 mask 或 clamp 不同 target token 切换角色：

- 不给 future action：作为 policy / predictor。
- 给 candidate action：作为 critic。
- 采样多个 action 并打分：作为 reranker / MPC scorer。

最重要的论文差异点不是 scale，而是：

- primitive-local process potential；
- future latent、action、potential 在同一个 denoising trajectory 中联合建模；
- masked modality inference；
- test-time flow refinement。

## 2. 为什么不能做成“大视频世界模型”

不要把 paper 写成：

```text
We train a general video-action world model for robotic manipulation.
```

原因：

- 我们没有上万小时机器人数据和大规模 DiT 训练资源。
- RGB future-frame diffusion / flow 会把计算预算和评价复杂度拉爆。
- 公开方向里已经有更大规模的 VLA / world model 工作，我们不能和它们拼 scale。

应该写成：

```text
We study primitive-local process potential as a dense, action-conditioned interface
for limited-data world-action modeling.
```

也就是说，PP-WAM 是一个有限数据下可训练、可解释、可评估的 process-aware latent world-action model。

## 3. MVP0 在论文中的角色

MVP0 不作为最终方法，而作为动机实验和 baseline。

当前 MVP0 已经支持这些结论：

- 只输入 obs/prompt，不输入 action，ranking 基本是随机。
- 输入 action 但没有 counterfactual ranking loss，action sensitivity 不稳定。
- 加入 CF ranking loss 后，positive action 的 predicted potential 明显高于 negative action。
- 提高 DeltaPhi regression weight 后，MAE/RMSE 改善，但 ranking/margin 下降。

MVP0 的论文作用：

```text
Primitive-local potential is meaningful, but a standalone critic exposes
a calibration-vs-action-sensitivity tradeoff.
```

PP-WAM 的引入逻辑：

```text
Instead of attaching a critic to actions after the fact, we jointly model
future observation latents, future actions, and process potential in one flow field.
```

## 4. 模型定义

### 4.1 条件输入

定义 context：

```text
c = (l, z_{t-H:t}, q_{t-H:t})
```

其中：

- `l`：language command / task prompt embedding；
- `z_{t-H:t}`：history observation latent；
- `q_{t-H:t}`：proprioception history；
- `H`：history length，第一版沿用 MVP0 的 `H=4`。

### 4.2 联合预测目标

定义 joint target：

```text
y_0 = (z_{t+1:t+K}, a_{t:t+K-1}, phi_{t:t+K})
```

其中：

- `z_{t+1:t+K}`：future observation latent；
- `a_{t:t+K-1}`：future action chunk；
- `phi_{t:t+K}`：process potential trajectory，或简化成 primitive-local `DeltaPhi`；
- `K`：horizon，第一版沿用 MVP0 的 `K=8`。

第一版主任务只预测 observation latent，不预测 RGB frame。RGB reconstruction 可以作为 appendix / qualitative demo，不作为主训练目标。

### 4.3 Flow matching 路径

采用线性 flow matching：

```text
epsilon ~ N(0, I)
y_tau = (1 - tau) * epsilon + tau * y_0
v_target = y_0 - epsilon
```

模型学习：

```text
v_theta(y_tau, tau, c, mask)
```

输出：

```text
v_theta = (v_z, v_a, v_phi)
```

核心 claim：

```text
future observation latent, action, and potential share one denoising trajectory.
```

这和普通 critic 的区别：

```text
ordinary critic:
  obs + action + prompt -> scalar score

PP-WAM:
  future latent + action + potential are jointly denoised by one conditional flow field
```

### 4.4 Loss

主 loss：

```text
L =
  lambda_z   * MSE(v_z,   v_z_target)
+ lambda_a   * MSE(v_a,   v_a_target)
+ lambda_phi * MSE(v_phi, v_phi_target)
+ lambda_cf  * L_cf
+ lambda_consistency * L_consistency
```

Counterfactual ranking loss：

```text
L_cf = -logsigmoid(phi_pos - phi_neg - margin)
```

这里需要保留 `L_cf`。不要假设 scalar potential 进入 flow matching 后自然就有 ranking 能力；MVP0 已经显示 action sensitivity 强依赖 counterfactual supervision。

### 4.5 Potential 作为 flow token 还是 scalar head

默认设计：

```text
phi is part of the flow state, but also receives counterfactual ranking supervision.
```

原因：

- 如果 `phi` 只是 CLS head，容易退化成外挂 critic。
- 如果 `phi` 是 flow token，它必须和 future latent / action token 一起被去噪。
- 这样才能自然支持 masked critic / policy / predictor mode。

但实验上必须保留 ablation：

- `phi` as flow token；
- `phi` as scalar head from DiT tokens；
- `phi` flow token + scalar readout。

如果 flow-token 版本不稳定，主 claim 可以改成：

```text
joint flow representation with process-potential decoding
```

不要强行声称 potential 本身必须 flow-matched。

## 5. Backbone 设计

第一版使用 small DiT / Transformer denoiser。

Token 类型：

```text
language token
history observation latent tokens
proprio tokens
future observation noisy latent tokens
action noisy tokens
potential noisy tokens
```

每类 token 需要：

- modality embedding；
- temporal embedding；
- flow timestep embedding；
- mask / clamp embedding。

不要简单 flatten + concat。模型必须知道每个 token 的语义：

```text
history condition
future noisy target
clamped candidate action
masked action
potential target
```

输入输出投影建议：

```text
obs latent -> obs input projection -> shared DiT -> obs velocity head
action     -> action input projection -> shared DiT -> action velocity head
potential  -> phi input projection    -> shared DiT -> phi velocity head
```

如果 shared DiT 出现明显模态冲突，再考虑 dual-stream：

```text
obs stream <-> action / potential stream
```

但 MVP1 先不要上复杂 dual-stream。

## 6. 推理模式

### 6.1 Predictor / Policy Mode

输入：

```text
language + history obs latent + proprio
```

future action 不给。

初始化：

```text
future obs latent: Gaussian
action chunk: Gaussian or previous-action initialized
potential: Gaussian
```

输出：

```text
future obs latent
action chunk
potential trajectory / DeltaPhi
```

用途：

- action generation；
- future latent prediction；
- process-aware rollout。

### 6.2 Critic Mode

输入：

```text
language + history obs latent + proprio + candidate action chunk
```

candidate action 处理：

```text
clamp action tokens
```

或：

```text
low-noise initialization around candidate action
```

模型输出：

```text
predicted DeltaPhi / potential score
predicted future latent consistency
```

用途：

- 给定 action 打分；
- positive vs negative ranking；
- 和 MVP0 critic 直接比较。

### 6.3 Reranking / MPC Mode

流程：

```text
1. base policy 或 PP-WAM 采样 N 个 action chunks
2. 每个 action chunk 进入 critic mode
3. 计算 score:
   score(a) =
      predicted DeltaPhi
    - alpha * uncertainty
    - beta  * action curvature
    - gamma * latent inconsistency
4. 选择最高分 action
```

输出：

```text
a* = argmax score(a)
```

这是最适合 ICRA 的实用接口。

## 7. Test-Time Flow Refinement

这一部分是差异化重点，但必须保守。

### 7.1 不做 full online finetuning

第一版不要承诺：

```text
test-time update full DiT weights
```

风险：

- 不稳定；
- 容易 overfit 当前 episode；
- 计算量高；
- 审稿人会问 safety / forgetting / leakage。

默认只做 training-free 或 lightweight adaptation。

### 7.2 Previous-Action Initialization

标准 flow / diffusion 从 Gaussian noise 采样，推理步数多，时序连续性也差。

PP-WAM 可以比较：

```text
Gaussian init:
  a_tau=0 ~ N(0, I)

Previous-action init:
  a_tau=0 ~ N(a_prev, sigma^2 I)
```

或：

```text
a_tau=0 = repeat(last_action) + noise
```

评估：

- action smoothness；
- denoising steps；
- expert action MSE；
- ranking / reranking performance。

### 7.3 Adaptive Denoising Steps

根据 flow residual 或 velocity stability 动态决定步数：

```text
if velocity direction stable:
    fewer steps
else:
    more steps
```

目标不是追求复杂理论，而是展示有限算力下的 inference-time compute allocation。

### 7.4 Latent Action Optimization

冻结模型参数，只优化当前候选 action latent / noise seed：

```text
maximize:
    predicted DeltaPhi
  - alpha * ||a - a_base||^2
  - beta  * action curvature
  - gamma * uncertainty
  - eta   * future latent inconsistency
```

命名可以是：

```text
test-time process-guided flow refinement
```

### 7.5 Adapter / Memory Update 放到高级版本

如果时间允许，再考虑：

- task memory token；
- process bias token；
- LoRA on small heads；
- primitive calibration head。

这些不作为 MVP1 或投稿必要条件。

## 8. 工程路线

### MVP1：Latent-Action-Potential Joint Flow

目标：

```text
证明 joint flow 比 MVP0 standalone critic 更强或更稳。
```

实现：

- 使用现有 GM-100 window pipeline；
- 读 frozen obs features；
- 读 prompt features；
- 新增 future obs latent target；
- 新增 `JointFlowDiT`；
- 训练输出 `v_z, v_a, v_phi`；
- 加 `L_cf` 做 action ranking。

第一版数据形状：

```text
obs_history_latents:  [B, H, C, D]
proprio_history:      [B, H, q_dim]
prompt_features:      [B, D_text]
future_obs_latents:   [B, K, C, D]
action_chunk:         [B, K, a_dim]
delta_phi_or_phi:     [B, K] or [B, 1]
```

第一版可以 camera mean-pool：

```text
obs_history: [B, H, D]
future_obs:  [B, K, D]
```

成功标准：

- DeltaPhi MAE/RMSE 不差于当前 `prompt_cf_w10/w20`；
- ranking/margin 不低于高权重 prompt CF 太多；
- 相比 `no future latent` ablation 有明确提升。

### MVP2：Masked Critic / Predictor Mode

目标：

```text
证明同一个 checkpoint 可以通过 mask 支持 critic、policy、predictor。
```

训练时随机采样：

```text
mode A: action known
mode B: action masked
mode C: future obs masked
mode D: action partially masked
```

评估：

- critic mode：positive vs negative ranking；
- policy mode：action MSE / expert retrieval；
- predictor mode：future latent MSE / cosine；
- reranking mode：top-1 action selection improvement。

### MVP3：Test-Time Refinement

目标：

```text
把资源限制转化为方法贡献。
```

实验：

```text
Gaussian init
previous-action init
adaptive steps
potential-guided latent optimization
```

指标：

- inference steps；
- latency；
- action smoothness；
- ranking；
- reranking improvement；
- DeltaPhi calibration。

### MVP4：Downstream Action Selection

目标：

```text
证明 PP-WAM 能改善 action selection。
```

离线实验：

```text
每个 state:
  1 expert action
  N generated / corrupted / policy-sampled actions
PP-WAM score all actions
report expert top-k retrieval and progress gain
```

如果能接仿真：

```text
base policy rollout
base policy + PP-WAM reranking rollout
```

主指标：

- success rate；
- primitive completion rate；
- average DeltaPhi improvement；
- failure recovery cases。

## 9. 实验矩阵

### 9.1 Baselines

必须保留：

```text
MVP0 stage_action_cf
MVP0 prompt_cf_w10
MVP0 prompt_cf_w20
```

新增 baseline：

```text
action-only flow policy
future-latent-only predictor
separate policy + external critic
concat critic without joint flow
scalar-head potential without phi flow token
```

### 9.2 PP-WAM Ablations

关键 ablation：

```text
Full PP-WAM
w/o future obs latent
w/o phi flow token
w/o CF ranking
w/o masked training
w/o previous-action init
w/o adaptive solver
w/o latent refinement
```

### 9.3 Metrics

Calibration：

```text
DeltaPhi MAE
DeltaPhi RMSE
potential monotonicity violation
```

Critic：

```text
all-negative tie-aware ranking
all-negative mean margin
per-negative ranking:
  zero
  shuffle
  wrong_arm
  scaled_0.25
  scaled_1.75
  reverse
```

Policy：

```text
action MSE
expert top-k retrieval
action smoothness
```

Predictor：

```text
future latent MSE
future latent cosine similarity
latent consistency under candidate actions
```

Planning：

```text
reranking improvement over base policy
primitive progress improvement
sim rollout success if available
```

Efficiency：

```text
denoising steps
latency
GPU memory
train time
```

## 10. 批判性风险

### 风险 1：和 τ0-WM 太像

应对：

- 强调 primitive-local process potential；
- 强调有限数据；
- 强调 latent-only；
- 强调 masked critic/predictor/policy interface；
- 强调 test-time refinement；
- 不和大模型拼 scale。

### 风险 2：future latent prediction 没有帮助

应对：

- 必须做 `no future latent` ablation；
- 如果 future latent 不提升，把主 claim 缩成 process-potential flow critic；
- predictor mode 作为辅助结果。

### 风险 3：potential flow token 不如 scalar head

应对：

- 同时实现 `phi as flow token` 和 `phi as scalar head`；
- 让实验决定主版本；
- 不预先把 claim 写死。

### 风险 4：hard negatives 仍接近随机

应对：

- 分开报告 easy/hard negatives；
- 增加更语义化 hard negatives；
- 不只报 overall ranking。

### 风险 5：ICRA 时间不够

当前到 ICRA 2027 deadline 的窗口非常紧。不要把真机、RGB video model、大规模预训练作为硬依赖。

## 11. 时间线

### 7 月第 1-2 周：MVP1 最小闭环

目标：

- `JointFlowDiT` forward pass；
- flow interpolation utility；
- joint flow loss；
- toy smoke test；
- GM-100 light 1 seed。

产出：

- loss curve；
- first calibration / ranking table。

### 7 月第 3-4 周：MVP1 三 seed 正式实验

目标：

- 3 seeds；
- compare against MVP0 `prompt_cf_w10/w20`；
- run `no future latent` and `no CF` ablations。

决策 gate：

```text
如果 PP-WAM 没有超过 MVP0:
    缩小 claim，优化 loss 和 representation
如果 PP-WAM 超过 MVP0:
    进入 MVP2
```

### 8 月第 1-2 周：MVP2 Masked Modes

目标：

- critic mode；
- policy mode；
- predictor mode；
- reranking mode。

产出：

- masked inference table；
- action generation metrics；
- critic vs predictor consistency。

### 8 月第 3 周：MVP3 Test-Time Refinement

目标：

- previous-action init；
- adaptive steps；
- latent optimization。

产出：

- latency vs performance；
- reranking improvement；
- test-time trick ablation。

### 8 月第 4 周：写作 + 补强实验

目标：

- finalize figures；
- finalize narrative；
- decide ICRA vs ICLR emphasis；
- write limitation section。

### 9 月第 1-2 周：Submission Polish

目标：

- all tables locked；
- appendix complete；
- code/report reproducible；
- rebuttal-risk audit。

## 12. 论文结构建议

### ICRA 风格题目

```text
Primitive-Progress World-Action Models for Process-Aware Robotic Manipulation
```

### ICLR 风格题目

```text
Masked Joint Flow Fields for Unified Prediction, Policy, and Process Criticism
```

### Abstract 主线

```text
Robotic policies often generate actions without explicitly modeling whether an action
advances the current manipulation process. We propose PP-WAM, a latent world-action
flow model that jointly predicts future observation latents, action chunks, and
primitive-local process potential. By masking or clamping action tokens, the same
model acts as a predictor, policy, critic, or reranker. Experiments show that
primitive-local potential improves action evaluation and that test-time flow
refinement improves action selection under limited robot data.
```

### Contribution

```text
1. We introduce primitive-local process potential as a dense interface for evaluating manipulation progress.
2. We propose PP-WAM, a latent joint flow model over future observation, action, and process potential.
3. We show masked inference turns one model into a predictor, policy, critic, and reranker.
4. We introduce lightweight test-time flow refinement for limited-data action selection.
```

## 13. 工程优先级

### 必须做

- `JointFlowDataset`
- `JointFlowDiT`
- flow interpolation utilities
- modality embeddings
- mask embeddings
- joint loss
- critic-mode evaluation
- policy-mode sampling
- reranking evaluation
- comprehensive report generator

### 可以后做

- multi-camera token 保留；
- RGB decoder；
- LoRA / memory token；
- simulation rollout；
- real-robot demo。

### 第一版不建议做

- 大规模 RGB video diffusion；
- full online DiT finetuning；
- multi-dataset VLA pretraining；
- 复杂 contact physics loss；
- 过多 heuristic scoring。

## 14. 通过 / 不通过标准

### MVP1 通过标准

至少满足一个：

```text
PP-WAM calibration better than MVP0 at similar ranking
```

或：

```text
PP-WAM ranking better than MVP0 at similar calibration
```

更理想：

```text
PP-WAM beats prompt_cf_w10/w20 on both calibration and at least one hard negative.
```

### MVP2 通过标准

必须证明：

```text
same checkpoint supports critic mode and policy/predictor mode
```

不能是三个模型分别训练。

### MVP3 通过标准

test-time trick 至少带来一个明确收益：

```text
fewer denoising steps at same ranking
```

或：

```text
higher ranking at same compute
```

或：

```text
better reranking success at same candidate count
```

## 15. 默认决策

- ICRA-first，ICLR-compatible。
- MVP1 latent-only，不做 RGB 主任务。
- GM-100 light + 可选仿真，不把真机作为硬依赖。
- small DiT，不训练大 foundation model。
- test-time refinement 不更新 full model 权重。
- MVP0 作为动机实验和 baseline。
- PP-WAM 作为最终主方法。

一句话总结：

```text
不要追大规模 video-action foundation model；
做一个有限数据下可训练、可解释、可评估的 primitive-progress latent world-action flow model。
```
