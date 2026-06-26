# MVP-0: Lightweight Action-Grounded Process Critic

## 1. Objective

MVP-0 只做一个轻量 sanity check：训练一个 **action-grounded process critic**。

```text
(o_{\le t}, q_t, a_{t:t+H}, s_t, c) -> Δφ^(s_t)
```

含义：

- `o_{\le t}`: observation history，使用 frozen visual features。
- `q_t`: 当前 proprioception。
- `a_{t:t+H}`: candidate future action chunk。
- `s_t`: 当前 primitive / stage。
- `c`: task id，不做语言泛化。
- `Δφ^(s_t)`: 当前 primitive 内的 potential increment。

MVP-0 只回答一个问题：

> 在当前 primitive 下，这段 candidate action 是否真的推进了过程？

核心命题：

> primitive-local progress 不是 video-only progress，也不是 stage-only progress；它应当由 observation、proprioception、stage 和 candidate action jointly determined。若固定 `(o, q, s, c)` 并替换 action，critic 的 predicted `Δφ` 应该随 action 合理变化。

如果这个命题不成立，后续 PP-WAM、future action generator、future observation latent prediction 都没有可靠支点。

## 2. Scope

### 2.1 必须保留

- episode-level train/val/test split。
- primitive-local potential label。
- simple counterfactual action ranking。
- stage-conditioned action encoding。
- frozen visual encoder features。
- `time_prior` baseline。
- action perturbation sensitivity。
- stage replacement sensitivity。

### 2.2 第一版明确不做

- 不训练视频 DiT 或 RGB future video generator。
- 不预测 future observation latent。
- 不生成 action。
- 不接真实机器人闭环控制。
- 不在线训练视觉 backbone。
- 不依赖 Qwen-VL、LLM 或大 VLM。
- 不做完整 event system。
- 不把 hard negative retrieval 当第一版必要条件。
- 不使用 `phi_future` head。
- 不使用 StageAuxHead。
- 不使用 monotonic loss。
- 不搭完整生产级 repo 结构。

### 2.3 延后项

这些内容保留为后续阶段，不进入第一轮 sanity check：

- **MVP-0d limited events**: 只在标签可靠时加入 `gripper_closed` / `contact`。
- **MVP-0e hard negatives**: simple negatives 成立后，再做 same-stage retrieval / near-state wrong-outcome retrieval。
- **MVP-1**: future action 或 future observation latent prediction。
- **MVP-2**: generator + evaluator dual-mode PP-WAM。

## 3. Data Interface

MVP-0 使用稳定的 episode/window 中间格式，但只保留 critic 必需字段。

### 3.1 Raw Episode

```text
data/episodes/<episode_id>/
  meta.json
  arrays.npz
  images/
    cam0/
      000000.jpg
      000001.jpg
  labels.json
```

`meta.json`:

```json
{
  "episode_id": "taskA_ep0001",
  "task_id": "taskA",
  "language": "pick up the red block and place it in the bowl",
  "fps": 10,
  "num_frames": 180,
  "cameras": ["cam0"],
  "action_dim": 14,
  "proprio_dim": 14,
  "action_space": "absolute"
}
```

`arrays.npz`:

```text
proprio: float32 [T, Q]
action: float32 [T, A]
eef_pose: optional float32 [T, E]
gripper: optional float32 [T, G]
object_pose: optional float32 [T, O]
target_pose: optional float32 [T, P] or float32 [P]
```

`labels.json`:

```json
{
  "primitive_boundaries": [
    {"stage": "approach", "start": 0, "end": 42},
    {"stage": "grasp", "start": 42, "end": 70},
    {"stage": "move", "start": 70, "end": 130},
    {"stage": "place", "start": 130, "end": 160},
    {"stage": "release", "start": 160, "end": 179}
  ],
  "success": true
}
```

### 3.2 Window Output

`prepare_windows` 输出：

```text
data/windows/
  windows.jsonl
  labels.npz
  index.json
```

`windows.jsonl` 每行：

```json
{
  "window_id": "taskA_ep0001_t0050",
  "episode_id": "taskA_ep0001",
  "task_id": "taskA",
  "t": 50,
  "history_indices": [47, 48, 49, 50],
  "future_indices": [50, 51, 52, 53, 54, 55, 56, 57],
  "stage": "grasp",
  "split": "train",
  "cross_boundary": false
}
```

`labels.npz`:

```text
delta_phi: float32 [N]
stage_id: int64 [N]
task_id: int64 [N]
primitive_time: float32 [N]
is_success: bool [N]
cross_boundary: bool [N]
```

默认窗口参数：

```yaml
history: 4
horizon: 8
stride: 2
split_by: episode
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
```

必须按 episode split，不能按 window 随机 split。

## 4. Labels

默认 stage vocabulary:

```text
approach
grasp
move
place
release
```

对每个 primitive boundary `[start, end]`:

```text
φ_t = clamp((t - start) / max(end - start, 1), 0, 1)
Δφ = φ_{min(t+H, end)} - φ_t
primitive_time = φ_t
```

规则：

- `Δφ` 是第一版唯一必需监督信号。
- `primitive_time` 只给 `time_prior` baseline 使用，不作为 full critic 默认输入。
- cross-boundary window 第一版默认排除训练，可保留到 diagnostic eval。
- failure episode 如果 label 不干净，先排除训练，只放入诊断集。
- 第一版不预测 `φ_future`，避免进一步强化线性时间先验。

## 5. Counterfactual Actions

每个 positive window 至少生成 1 个 simple negative。

默认 negative types:

```text
zero       : 全 0 action
reverse    : action chunk 时间维反转
shuffle    : 同 split / 同 primitive 内随机替换 action chunk
wrong_arm  : 双臂 action 左右交换
scaled     : normalized action chunk 乘 0.25 或 1.75，并 clip 到训练集 action range
```

规则：

- `wrong_arm` 只在 action 可明确拆成左右臂时启用。
- 默认 `action_dim=14` 时，前 7 维和后 7 维互换。
- `shuffle` 只能从同 split 内采样，避免 train/test 泄漏。
- simple negative 按 type 单独评估。
- hard negative retrieval 不进入第一版。

训练 pair:

```text
(same obs, same proprio, same stage, same task, positive action)
(same obs, same proprio, same stage, same task, negative action)
```

## 6. Visual Features

第一版使用 frozen image encoder 预提取特征，不在线训练视觉主干。

推荐 encoder:

```text
vit_base_patch14_dinov2.lvd142m
```

轻量备选：

```text
vit_small_patch14_dinov2.lvd142m
siglip_base_patch16_224
clip-vit-base-patch16
```

feature store:

```text
data/features/<episode_id>.npz
  cam0: float16 [T, Dv]
  cam1: optional float16 [T, Dv]
```

training input:

```text
obs_features: [B, L, C, Dv]
```

多相机融合：

```text
per_camera_feature -> shared Linear(Dv, 256) -> mean pool across cameras
```

可以先用 mock features 跑通 pipeline，但正式 sanity result 必须使用 frozen visual features。

## 7. Model

保留两个模型层级，避免一开始把效果归因到复杂 architecture。

### 7.1 Inputs

```text
obs_features: [B, L, C, Dv]
proprio_t: [B, Q]
action_chunk: [B, H, A]
stage_id: [B]
task_id: [B]
primitive_time: optional [B], only for time_prior
```

第一版用 `task_id` embedding，不使用自然语言 encoder。

### 7.2 Output

唯一必需输出：

```text
delta_phi_logit: [B, 1]
pred_delta_phi = sigmoid(delta_phi_logit)
```

### 7.3 MLP Critic Baseline

用于确认数据和 label 是否有基础信号。

```text
obs: mean pool over L/C -> Linear(Dv, 256)
proprio: MLP(Q -> 128 -> 128)
action: flatten or mean pool -> MLP(H*A -> 256 -> 256)
stage: Embedding(5, 64)
task: Embedding(num_tasks, 64)
fusion: concat -> MLP(768 -> 512 -> 256 -> 1)
```

### 7.4 Stage-FiLM Transformer Critic

用于主结果。

```text
proprio_t -> MLP(Q -> 256 -> 256)
action_chunk -> Linear(A -> 256) -> 2-layer temporal Transformer
stage_id -> Embedding(5, 128) -> MLP -> gamma, beta
action_tokens = gamma * action_tokens + beta
```

Fusion tokens:

```text
obs history tokens: L
proprio token: 1
stage token: 1
task token: 1
action tokens: H
```

Default transformer:

```yaml
hidden_dim: 256
layers: 2
heads: 4
dropout: 0.1
```

Full sanity config can use:

```yaml
hidden_dim: 512
layers: 4
heads: 8
dropout: 0.1
```

### 7.5 Action Semantics

不要默认假设 action 和 proprio 同空间。

```yaml
action:
  mode: absolute   # absolute | delta
  convert_absolute_to_delta: false
  normalize: true
```

只有在确认 `action_t` 与 `proprio_t[:A]` 同坐标、同语义、同尺度时，才启用：

```text
action_delta_i = action_{t+i} - proprio_t[:A]
```

否则直接使用归一化后的原始 action chunk。

## 8. Losses

第一版只使用两个 loss。

Regression:

```text
L_delta = SmoothL1(pred_delta_phi, target_delta_phi)
```

Counterfactual ranking:

```text
L_cf = -log sigmoid(pred_delta_phi_pos - pred_delta_phi_neg - margin)
margin = 0.05
```

Total:

```text
L = 1.0 * L_delta + λ_cf * L_cf
```

默认：

```yaml
lambda_cf: 0.5
```

Ablation:

```text
obs_action_stage    : λ_cf = 0.0
obs_action_stage_cf : λ_cf = 0.5
```

不使用：

- `phi_future` loss
- event loss
- stage auxiliary loss
- monotonic loss

## 9. Experiments

第一版只跑 5 个 ablation。

| Name | Inputs | Loss | Purpose |
| --- | --- | --- | --- |
| `time_prior` | primitive_time, stage, task | `L_delta` | 检查线性时间标签有多强 |
| `obs_stage` | obs, proprio, stage, task | `L_delta` | video/stage progress baseline |
| `obs_action` | obs, proprio, action, task | `L_delta` | 检查没有 stage 时 action 是否足够 |
| `obs_action_stage` | obs, proprio, action, stage, task | `L_delta` | 检查 stage 是否帮助解释 action |
| `obs_action_stage_cf` | obs, proprio, action, stage, task | `L_delta + L_cf` | 核心方法 |

关键期望：

```text
ranking(obs_action_stage_cf) > ranking(obs_action_stage) > ranking(obs_stage)
```

`time_prior` 可以 MAE 强，但它在固定 `(o, q, s, c)` 时无法区分 correct action 和 negative action。

### 9.1 Action Perturbation

固定 `(o, q, s, c)`，输入：

```text
correct
zero
reverse
shuffle
wrong_arm
scaled_0.25
scaled_1.75
```

报告：

```text
pairwise_ranking_accuracy
mean_margin
per_negative_type_accuracy
per_stage_accuracy
```

必须输出分布图：

```text
pred_delta_phi(correct)
pred_delta_phi(negative_type)
margin(correct - negative_type)
```

### 9.2 Stage Replacement

固定 `(o, q, a, c)`，替换：

```text
true_stage
previous_stage
next_stage
random_stage
```

报告：

```text
true_vs_wrong_stage_margin
wrong_stage_high_progress_rate
per_stage_replacement_matrix
```

## 10. Metrics And Acceptance

主指标：

```text
ranking_accuracy = mean[pred_delta_phi(a+) > pred_delta_phi(a-)]
mean_margin = mean[pred_delta_phi(a+) - pred_delta_phi(a-)]
delta_phi_MAE = mean(abs(pred_delta_phi - target_delta_phi))
```

必须分别报告：

```text
overall_ranking_acc
zero_ranking_acc
reverse_ranking_acc
shuffle_ranking_acc
wrong_arm_ranking_acc
scaled_ranking_acc
per_stage_ranking_acc
mean_margin
delta_phi_MAE
```

MAE 是辅助指标，不是核心结论。

MVP-0a sanity check 成立的参考标准：

```text
obs_action_stage_cf ranking_acc > obs_stage ranking_acc
overall simple-negative ranking_acc >= 75%
mean_margin > 0.03
per-stage ranking_acc >= 55%
action perturbation plot shows correct distribution shifted right
```

这些阈值是 sanity guide，不是论文级 acceptance。

如果达不到，排查顺序：

1. primitive boundary 是否干净。
2. `time_prior` 是否已经解释大部分 label。
3. action normalization / action semantics 是否正确。
4. negative 是否过弱、过强或采样错误。
5. 最后才扩大模型。

## 11. Minimal Code Structure

第一版代码结构：

```text
mvp0/
  data.py
  labels.py
  counterfactual.py
  model.py
  train.py
  eval.py
  plot.py
  configs/
    debug.yaml
    full.yaml
```

职责：

- `data.py`: episode/window 读取、episode split、dataset/collate。
- `labels.py`: `φ`、`Δφ`、`primitive_time` 生成。
- `counterfactual.py`: simple negative generation 和 paired sampling。
- `model.py`: `TimePrior`, `MLPCritic`, `StageFiLMTransformerCritic`。
- `train.py`: 单卡训练、ablation input mask、checkpoint、metrics。
- `eval.py`: MAE、ranking、margin、stage replacement eval。
- `plot.py`: action perturbation 和 margin distribution plots。

暂时不要拆成 `training/`、`evaluation/`、`vision/`、`labeling/` 等子包。

## 12. Default Config

```yaml
seed: 42
device: cuda
precision: amp_bf16

data:
  history: 4
  horizon: 8
  stride: 2
  batch_size: 64
  split_by: episode

features:
  encoder: vit_base_patch14_dinov2.lvd142m
  image_size: 224
  feature_dim: 768
  mock_features: false

action:
  mode: absolute
  convert_absolute_to_delta: false
  normalize: true

model:
  name: stage_film_transformer
  hidden_dim: 256
  transformer_layers: 2
  transformer_heads: 4
  dropout: 0.1

loss:
  delta_weight: 1.0
  counterfactual_weight: 0.5
  margin: 0.05

negatives:
  simple_per_positive: 1
  types: ["zero", "reverse", "shuffle", "wrong_arm", "scaled"]
  scaled_values: [0.25, 1.75]

optim:
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-4
  grad_clip_norm: 1.0

train:
  max_epochs: 20
  eval_every: 1
  save_best_by: val/ranking_acc
```

## 13. Implementation Order

### MVP-0a: Minimal Critic

1. 实现 episode/window schema。
2. 生成 primitive-local `Δφ` 和 `primitive_time`。
3. 接入 mock features 或 frozen features。
4. 实现 `TimePrior`、`MLPCritic`、`StageFiLMTransformerCritic`。
5. 跑 `time_prior`、`obs_stage`、`obs_action`、`obs_action_stage`。

### MVP-0b: Counterfactual Ranking

1. 实现 simple negative generation。
2. 实现 paired dataset sampling。
3. 加 ranking loss。
4. 跑 `obs_action_stage_cf`。
5. 输出 action perturbation ranking 和 margin plots。

### MVP-0c: Stage Sensitivity

1. 固定 obs/action，替换 stage。
2. 评估 true vs wrong stage margin。
3. 输出 stage replacement matrix。

### MVP-0d: Limited Events

只在标签可靠时加入：

```text
gripper_closed
contact
```

如果没有可靠标签，跳过。

### MVP-0e: Hard Negatives

simple negative 结果成立后再加入：

```text
same_stage_retrieval
near_state_wrong_outcome
```

并单独报告 hard negative ranking。

## 14. Testing

Unit tests:

- label: primitive 内 `φ` 从 0 到 1，`Δφ` 在 primitive end 截断。
- data: history/future index 不越界，split 按 episode，不泄漏。
- counterfactual: 每种 negative shape 不变，`shuffle` 不跨 split，`wrong_arm` 按 action_dim 启用。
- model: MLP 和 transformer forward output shape 都是 `[B, 1]`。
- loss: positive score 高于 negative score 时 ranking loss 更小。

Smoke test:

```text
2 tasks
4 episodes
20 frames each
2-3 stages
random/mock image features
synthetic action-progress relation
```

必须跑通：

```bash
python mvp0/train.py --config mvp0/configs/debug.yaml experiment=time_prior
python mvp0/train.py --config mvp0/configs/debug.yaml experiment=obs_action_stage_cf
python mvp0/eval.py --checkpoint outputs/obs_action_stage_cf/best.pt --split test
python mvp0/plot.py --eval outputs/obs_action_stage_cf/eval
```

## 15. Data Scale

第一轮建议：

```text
3 tasks
10-20 episodes per task
3-5 primitives per episode
history L = 4
horizon H = 8
stride = 2
```

第一轮重点是 clean labels，不是规模。不要一开始扩到 10 个任务，否则 debugging 成本会明显增加。

## 16. Main Risks

### Risk 1: Time Prior Too Strong

症状：

- `time_prior` MAE 接近 full model。
- action perturbation 后 predicted `Δφ` 几乎不变。

处理：

- ranking 作为主指标。
- 加 counterfactual ranking loss。
- 第二版加入 failure / stall / retry windows。

### Risk 2: Action Neglect

症状：

- `obs_stage` 接近 `obs_action_stage_cf`。
- correct vs wrong action ranking 不显著。

处理：

- 检查 action normalization 和 action semantics。
- 提高 `counterfactual_weight`。
- 检查 negative action 是否真的改变行为语义。

### Risk 3: Stage Shortcut Or Noise

症状：

- 模型几乎只靠 stage 给固定 `Δφ`。
- stage transition 附近错误集中。

处理：

- 对比 `obs_action` 和 `obs_action_stage`。
- 单独看 per-stage ranking。
- 排除或降权 boundary 附近 window。

### Risk 4: Counterfactual Too Easy

症状：

- zero/reverse ranking 很高，但 shuffle/scaled 不明显。
- 模型只学会检测 action norm 异常。

处理：

- 按 negative type 单独报告。
- 加强 shuffle negative。
- simple negative 成立后再做 hard negative retrieval。

## 17. Success Statement

MVP-0 成功时，应能给出以下结论：

> 在 primitive-local robot manipulation window 中，video/stage progress prediction 不足以判断 candidate action 是否推进过程。加入 action-conditioned counterfactual learning 后，process critic 能在同一 observation、proprioception、stage 和 task 条件下区分 correct action 与 wrong action，并输出更合理的 primitive-local `Δφ`。

这就是后续 PP-WAM 的最小 method 支点。
