# 4090 Handoff Checklist

This document is the handoff from WSL toy development to the 4090 data/compute machine.

## 1. Clone And Environment

```bash
git clone git@github.com:dyxia1241/WAM.git
cd WAM
conda env create -f environment.yml
conda activate wam
python -m pytest
```

## 2. Data Layout

Use the 4090 as the source of truth for data and artifacts:

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

For the first GM-100 subset download and conversion plan, see
`docs/gm100_data_plan.md`.

Episode directories must contain:

```text
data/episodes/<episode_id>/
  meta.json
  arrays.npz
  labels.json
  images/
    cam0/
      000000.jpg
```

## 3. Prepare Windows

```bash
python -m ppwam.prepare_windows \
  --episodes /data/WAM/episodes \
  --output /data/WAM/windows \
  --history 4 \
  --horizon 8 \
  --stride 2
```

## 4. Extract Frozen Transformer Features

```bash
python -m ppwam.extract_vision_features \
  --episodes /data/WAM/episodes \
  --output /data/WAM/features \
  --model vit_base_patch14_dinov2.lvd142m \
  --image-size 224 \
  --batch-size 128 \
  --device cuda
```

## 5. Generate Counterfactual Pair Index

```bash
python -m ppwam.make_counterfactuals \
  --windows /data/WAM/windows \
  --output /data/WAM/counterfactuals \
  --types zero,reverse,shuffle,wrong_arm,scaled_0.25,scaled_1.75
```

## 6. Run MVP-0 Ablation

```bash
python -m ppwam.run_ablation \
  --config configs/full.yaml \
  --output-dir /data/WAM/outputs \
  data.windows_dir=/data/WAM/windows \
  data.episodes_dir=/data/WAM/episodes \
  data.features_dir=/data/WAM/features
```

## 7. Evaluate, Plot, And Report

```bash
python -m ppwam.eval \
  --checkpoint /data/WAM/outputs/obs_action_stage_cf/best.pt \
  --split test \
  --output /data/WAM/outputs/obs_action_stage_cf/eval

python -m ppwam.plot \
  --eval /data/WAM/outputs/obs_action_stage_cf/eval

python -m ppwam.reports \
  --outputs /data/WAM/outputs \
  --output /data/WAM/outputs/report
```

Every run writes `metrics.json` and `manifest.json`; eval additionally writes predictions and sensitivity CSV files.
