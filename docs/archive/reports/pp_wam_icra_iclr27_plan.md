# PP-WAM ICRA / ICLR 2027 规划草案

状态：planning draft  
默认路线：ICRA 2027 first，ICLR 2027 compatible
核心目标：把 MVP0 的 primitive-local potential critic 升级为 latent joint flow world-action model。

## 1. 核心主张

PP-WAM 不做大规模 RGB video-action foundation model，而做有限数据下可训练、可评估的 latent joint flow：

```text
language command + observation history + proprioception
    -> future observation latent + future action chunk + primitive-local process potential
```

同一个模型通过 mask / clamp 切换功能：

- 不给 future action：policy / predictor。
- 给 candidate action：critic。
- 采样多个 action 并打分：reranker / MPC scorer。

论文差异点：

- primitive-local process potential；
- future latent、action、potential 共享同一个 denoising trajectory；
- masked modality inference；
- test-time flow refinement。

## 2. MVP0 的角色

MVP0 是动机实验和 baseline，不是最终方法。

已有结论：

- action-free 模型 ranking 接近随机；
- action-conditioned 模型如果没有 CF loss，action sensitivity 不稳定；
- 加 CF ranking loss 后，positive action 的 potential 明显高于 negative action；
- 提高 DeltaPhi regression weight 会改善 MAE/RMSE，但 ranking/margin 下降。

论文里的表述：

```text
Primitive-local potential is useful, but a standalone critic exposes a
calibration-vs-action-sensitivity tradeoff.
```

PP-WAM 的自然引入：

```text
Instead of attaching a critic to actions after the fact, PP-WAM jointly models
future observation latents, future actions, and process potential in one flow field.
```

## 3. 模型定义

Condition:

```text
c = (l, z_{t-H:t}, q_{t-H:t})
```

Target:

```text
y_0 = (z_{t+1:t+K}, a_{t:t+K-1}, phi_{t:t+K})
```

含义：

- `l`：language command / prompt embedding；
- `z`：observation latent，第一版不预测 RGB；
- `q`：proprioception；
- `a`：future action chunk；
- `phi`：primitive-local process potential 或 DeltaPhi trajectory；
- 默认沿用 MVP0：`H=4`, `K=8`。

Flow matching：

```text
epsilon ~ N(0, I)
y_tau = (1 - tau) * epsilon + tau * y_0
v_target = y_0 - epsilon
v_theta = (v_z, v_a, v_phi)
```

Loss:

```text
L =
  lambda_z   * MSE(v_z,   v_z_target)
+ lambda_a   * MSE(v_a,   v_a_target)
+ lambda_phi * MSE(v_phi, v_phi_target)
+ lambda_cf  * L_cf
+ lambda_consistency * L_consistency
```

其中：

```text
L_cf = -logsigmoid(phi_pos - phi_neg - margin)
```

关键原则：`phi` 可以作为 flow token 进入 joint denoising，但仍必须保留 CF ranking supervision；不要只依赖 scalar flow loss 产生 critic 能力。

## 4. Backbone

第一版用 small DiT / Transformer denoiser。

Token 类型：

- language token；
- history observation latent tokens；
- proprio tokens；
- future observation noisy latent tokens；
- action noisy tokens；
- potential noisy tokens。

每类 token 必须有：

- modality embedding；
- temporal embedding；
- flow timestep embedding；
- mask / clamp embedding。

第一版使用 shared DiT + modality-specific projections/heads：

```text
obs latent -> obs proj    -> shared DiT -> obs velocity head
action     -> action proj -> shared DiT -> action velocity head
potential  -> phi proj    -> shared DiT -> phi velocity head
```

如果模态冲突明显，再考虑 dual-stream；MVP1 不先做复杂 dual-stream。

## 5. 推理模式

### Predictor / Policy

输入：

```text
language + history obs latent + proprio
```

future obs/action/potential 从 Gaussian 或 previous-action init 开始 denoise，输出：

```text
future obs latent, action chunk, potential
```

### Critic

输入 candidate action：

```text
language + history obs latent + proprio + candidate action chunk
```

action tokens clamp 或 low-noise init，输出：

```text
predicted DeltaPhi / potential score
future latent consistency
```

### Reranking / MPC

```text
1. sample N action chunks
2. critic mode score each action
3. choose argmax predicted DeltaPhi, optionally penalized by uncertainty/smoothness/inconsistency
```

## 6. Test-Time Flow Refinement

不做 full online DiT finetuning。第一版只做低风险方法：

- previous-action initialization；
- adaptive denoising steps；
- latent action / noise seed optimization；
- 可选 small memory token / calibration head，不作为 MVP1 依赖。

Latent optimization objective:

```text
maximize:
    predicted DeltaPhi
  - alpha * ||a - a_base||^2
  - beta  * action curvature
  - gamma * uncertainty
  - eta   * future latent inconsistency
```

## 7. MVP 路线

### MVP1: Latent-Action-Potential Joint Flow

目标：证明 joint flow 比 MVP0 standalone critic 更强或更稳。

必须实现：

- `JointFlowDataset`；
- `JointFlowDiT`；
- flow interpolation utilities；
- modality / mask embeddings；
- joint velocity loss；
- CF ranking loss；
- critic-mode evaluation。

第一版数据形状：

```text
obs_history_latents: [B, H, C, D]  -> 可先 camera mean-pool 到 [B, H, D]
proprio_history:     [B, H, q_dim]
prompt_features:     [B, D_text]
future_obs_latents:  [B, K, C, D]  -> 可先 camera mean-pool 到 [B, K, D]
action_chunk:        [B, K, a_dim]
phi_or_delta_phi:    [B, K] or [B, 1]
```

通过标准：

- 在 calibration 上不差于 `prompt_cf_w10/w20`；
- 在 ranking/margin 上不比高权重 prompt CF 明显更差；
- `full model` 相比 `no future latent` 有明确收益。

### MVP2: Masked Modes

目标：同一个 checkpoint 支持 critic / policy / predictor。

训练 mask：

- action known；
- action masked；
- future obs masked；
- action partially masked。

评估：

- critic ranking；
- action MSE / expert retrieval；
- future latent MSE / cosine；
- reranking improvement。

### MVP3: Test-Time Refinement

比较：

- Gaussian init；
- previous-action init；
- adaptive steps；
- potential-guided latent optimization。

指标：

- denoising steps；
- latency；
- action smoothness；
- ranking/margin；
- reranking improvement。

### MVP4: Downstream Action Selection

离线动作选择：

```text
state + {expert action, generated negatives, policy samples}
PP-WAM scores all actions
report expert top-k retrieval and progress gain
```

如果有仿真资源，再做：

```text
base policy rollout vs base policy + PP-WAM reranking
```

## 8. 实验矩阵

Baselines:

- MVP0 `stage_action_cf`；
- MVP0 `prompt_cf_w10`；
- MVP0 `prompt_cf_w20`；
- action-only flow policy；
- separate policy + external critic；
- concat critic without joint flow。

Ablations:

- full PP-WAM；
- w/o future obs latent；
- w/o phi flow token；
- scalar phi head only；
- w/o CF ranking；
- w/o masked training；
- w/o previous-action init；
- w/o adaptive solver。

Metrics:

- DeltaPhi MAE/RMSE；
- all-negative tie-aware ranking；
- all-negative mean margin；
- per-negative ranking：zero, shuffle, wrong_arm, scaled_0.25, scaled_1.75, reverse；
- action MSE / expert top-k；
- future latent MSE / cosine；
- reranking improvement；
- latency / denoising steps / GPU memory。

## 9. 主要风险

- **和 τ0-WM 太像**：强调 primitive-local potential、limited-data、latent-only、masked critic/predictor/policy interface、test-time refinement。
- **future latent 没帮助**：必须做 `no future latent` ablation；若失败，主张缩成 process-potential flow critic。
- **phi flow token 不稳定**：同时测 scalar head 版本，实验决定最终写法。
- **hard negatives 仍接近随机**：分 easy/hard negatives 报告，并增加更语义化 hard negatives。
- **ICRA 时间紧**：不要把 RGB video model、真机、大规模预训练作为硬依赖。

## 10. 时间线

- **7 月第 1-2 周**：MVP1 最小闭环，toy smoke + GM-100 light 1 seed。
- **7 月第 3-4 周**：MVP1 三 seed，和 MVP0 baselines 比较，跑关键 ablations。
- **8 月第 1-2 周**：MVP2 masked modes，完成 critic/policy/predictor/reranking 评估。
- **8 月第 3 周**：MVP3 test-time refinement。
- **8 月第 4 周**：写作、补实验、确定 ICRA/ICLR 叙事。
- **9 月第 1-2 周**：锁表、补 appendix、做 reproducibility audit。

## 11. 默认决策

- ICRA-first，ICLR-compatible。
- MVP1 latent-only，不做 RGB 主任务。
- GM-100 light + 可选仿真，不把真机作为硬依赖。
- small DiT，不训练大 foundation model。
- test-time refinement 不更新 full model 权重。
- MVP0 作为动机实验和 baseline。
- PP-WAM 作为最终主方法。

一句话：

```text
不要追大规模 video-action foundation model；
做一个有限数据下可训练、可解释、可评估的 primitive-progress latent world-action flow model。
```
