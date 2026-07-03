# WAM Device, Repo, Data, and Compute Plan

## 1. Core Decisions

当前 WAM 的基础设施决策：

- **主 git repo 放在 4090 台式机**。
- **数据主副本放在 4090 台式机**。
- **MVP-0 使用 transformer-based backbone / critic**。
- **MVP-0 不训练视觉 backbone，先使用 frozen transformer visual features**。
- **H200 集群作为离线批处理训练资源，不作为第一开发环境**。

这里的 transformer-based backbone 第一阶段具体指：

- 视觉特征 backbone 使用 ViT 系列 frozen encoder，例如 DINOv2 ViT / SigLIP ViT。
- critic 主模型使用 Stage-FiLM Transformer Critic。
- 不从零训练 RGB/video backbone。
- 不在 MVP-0 阶段微调 visual backbone。

## 2. Machine Roles

### 2.1 4090 Desktop

4090 是 WAM 的中心机器。

职责：

- 维护主 git repo。
- 保存 raw data、converted episodes、windows、features、counterfactual indices。
- 下载并缓存 transformer visual backbone 权重。
- 运行 frozen feature extraction。
- 运行 MVP-0 正式 sanity ablation。
- 保存正式 checkpoints、metrics、plots 和 run manifests。
- 作为同步到其他机器和 H200 的 artifact source。

建议目录：

```text
/data/WAM/
  raw/
  episodes/
  windows/
  features/
  counterfactuals/
  outputs/
  checkpoints/
  manifests/
  pretrained/
```

建议 repo 放置：

```text
~/projects/WAM/
```

### 2.2 Windows Desktop / WSL

WSL 是轻量开发和文档编辑环境。

职责：

- 编辑代码和文档。
- 跑 unit tests。
- 跑 toy dataset smoke test。
- 使用 mock features 或极小真实 feature subset。
- 不保存主数据副本。
- 不跑正式 feature extraction。

适合任务：

```text
pytest
toy dataset pipeline
labels / counterfactual / model shape debugging
markdown planning
small config editing
```

### 2.3 Ubuntu 5060 Laptop

5060 笔记本是移动 GPU debug 机器。

职责：

- 检查 CUDA / AMP / DataLoader / checkpoint save-load。
- 跑 2-4 episode 的真实 GPU smoke test。
- 跑一个短 epoch ablation。
- 验证环境在非 4090 机器上可复现。

不建议：

- 不作为数据主盘。
- 不跑完整 ablation。
- 不做大规模 feature extraction。

### 2.4 Shanghai AI Lab H200 Cluster

H200 集群用于离线批处理和后续扩展实验。

约束：

- GPU 节点不能联网。
- 不适合作为第一开发环境。
- 不适合动态下载 Python packages、model weights 或 datasets。

职责：

- 消费 4090 打包好的 repo snapshot、环境、features、configs。
- 跑 multi-seed ablation。
- 跑更大 batch / 更长 epoch。
- 后续跑 hard negative、大模型、MVP-1/MVP-2 实验。

H200 不负责：

- 在线下载 backbone 权重。
- 在线安装依赖。
- 原始数据清洗。
- 第一轮交互式 debug。

## 3. Repo Policy

主 repo 在 4090 上维护。

推荐 repo 结构：

```text
WAM/
  README.md
  mvp0_design.md
  device_compute_plan.md
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
  scripts/
    sync_artifacts.sh
    pack_offline_env.sh
    submit_h200.sh
  docs/
```

git 应追踪：

```text
source code
configs
docs
small toy metadata
scripts
README
```

git 不应追踪：

```text
data/
features/
outputs/
checkpoints/
wandb/
*.pt
*.pth
*.ckpt
*.npz
*.hdf5
*.parquet
pretrained/
```

如果之后需要远程备份，4090 repo 可以 mirror 到 GitHub private repo 或 lab 内网 Git，但 source of truth 仍然以 4090 本地 repo 为准。

## 4. Data And Artifact Policy

4090 保存数据主副本。

核心 artifact：

```text
data/episodes/
data/windows/windows.jsonl
data/windows/labels.npz
data/windows/index.json
data/features/<episode_id>.npz
data/counterfactuals/simple_pairs.npz
outputs/<run_id>/
checkpoints/<run_id>/
manifests/<run_id>.yaml
```

每个正式 run 必须保存：

```text
git commit hash
config snapshot
dataset version
feature encoder name
feature encoder checkpoint/hash
feature dtype
action normalization stats
train/val/test episode split
negative generation config
machine name
CUDA / PyTorch version
```

同步策略：

- 大数据不进 git。
- 4090 到 WSL/5060 只同步 toy subset 或小 feature subset。
- 4090 到 H200 同步 prepared features、windows、configs、repo snapshot、offline env。
- H200 输出的 checkpoints / metrics / plots 回传到 4090 汇总。

## 5. Backbone And Model Policy

MVP-0 的 backbone 策略：

- 使用 transformer-based frozen visual backbone。
- 首选 DINOv2 ViT：

```text
vit_base_patch14_dinov2.lvd142m
```

轻量备选：

```text
vit_small_patch14_dinov2.lvd142m
siglip_base_patch16_224
clip-vit-base-patch16
```

训练策略：

- MVP-0 不训练 visual backbone。
- 4090 负责下载权重和预提 features。
- H200 默认直接读取预提 features，不重新跑 image encoder。
- critic 使用 transformer-based Stage-FiLM Transformer Critic。
- MLP Critic 只作为 sanity baseline，不作为最终主模型。

只有在 MVP-0 证明 action-grounded critic 成立后，才考虑：

- partial fine-tune visual encoder。
- larger transformer critic。
- future observation latent prediction。
- video/world backbone training。

## 6. Experiment Routing

### WSL

运行：

```text
unit tests
toy smoke test
mock feature pipeline
config validation
document updates
```

### 5060 Laptop

运行：

```text
small CUDA smoke test
2-4 episode real feature training
short epoch obs_action_stage_cf run
checkpoint load/save test
```

### 4090 Desktop

运行：

```text
feature extraction
time_prior
obs_stage
obs_action
obs_action_stage
obs_action_stage_cf
action perturbation plots
stage replacement evaluation
```

### H200 Cluster

运行：

```text
offline dry run
multi-seed ablation
larger batch runs
longer training
future hard-negative experiments
future MVP-1/MVP-2 runs
```

H200 执行前必须由 4090 准备：

```text
repo snapshot
offline Python environment or wheelhouse
pretrained weights if needed
features
windows
configs
submit scripts
```

## 7. Immediate Priority

当前最先做的事情：

1. 在 4090 上创建 WAM 主 git repo。
2. 添加 `.gitignore`，排除 data/features/outputs/checkpoints/pretrained。
3. 建立 `/data/WAM/` 数据目录结构。
4. 把现有 `mvp0_design.md` 和本规划文档迁入 4090 repo。
5. 搭建 `mvp0/` 最小代码骨架。
6. 先实现 toy dataset + mock features 的 smoke test。
7. 再接入真实 episode schema 和 frozen transformer feature extraction。
8. 在 4090 跑第一轮 MVP-0a/0b sanity ablation。

## 8. Success Criteria

基础设施成功的标准：

- WSL、5060、4090 都能 clone 或同步同一份 WAM repo。
- WSL 能跑 unit tests 和 toy smoke test。
- 5060 能跑小规模 CUDA smoke test。
- 4090 能完成 feature extraction 和正式 MVP-0 ablation。
- H200 能在无联网 GPU 节点上加载离线环境、features 和 config 跑通 single epoch。
- 所有正式实验都能通过 manifest 回溯到代码、数据、feature 和配置版本。
