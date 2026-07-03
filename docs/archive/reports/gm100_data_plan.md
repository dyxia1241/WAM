# GM-100 Subset Data Plan

This note defines the first real-data path for MVP-0.

## Source

Use the Hugging Face dataset:

```text
rhos-ai/gm100-cobotmagic-lerobot
```

The dataset is a GM-100 Cobot Magic preview in LeRobot 2.1 format. Each task has its own folder with `meta/`, per-episode parquet files under `data/chunk-000/`, and per-episode videos under `videos/chunk-000/`.

## First Subset

Initial target:

```text
20 tasks x 5 episodes per task = 100 episodes
```

This is enough to test:

- whether WAM's importer can parse real robot state/action arrays;
- whether video frame extraction aligns with parquet `frame_index`;
- whether frozen DINOv2 features fit on the 5060 and 4090;
- whether MVP-0 ranking metrics move on real data.

Do not download the full dataset for the first pass.

## Machine Assignment

### 4090 Desktop

Use the 4090 as the authoritative data machine.

Recommended layout:

```text
/data/WAM/
  raw/gm100-cobotmagic-lerobot_subset/
  episodes/gm100_subset/
  windows/gm100_subset/
  features/gm100_dinov2_base/
  counterfactuals/gm100_subset/
  outputs/gm100_mvp0/
  pretrained/
```

4090 jobs:

- download the GM-100 raw subset;
- cache DINOv2 / timm checkpoints;
- convert LeRobot parquet + mp4 into WAM episode layout;
- extract frozen visual features for all selected episodes;
- run formal MVP-0 ablations.

### 5060 Laptop

Use the 5060 for a smaller real-data smoke test.

Recommended subset:

```text
2 tasks x 2 episodes per task
```

5060 jobs:

- verify CUDA environment;
- download or receive a tiny raw subset;
- run importer on a few episodes;
- run real DINOv2 feature extraction with a smaller batch size;
- run one short `obs_action_stage_cf` training job.

### WSL Desktop

WSL stays lightweight.

WSL jobs:

- edit code and docs;
- run unit tests;
- run toy pipeline and mock-feature smoke tests;
- run GM-100 dry-run planning commands.

Avoid storing the raw GM-100 subset in WSL unless debugging a single episode.

### H200 Cluster

H200 GPU nodes are offline, so they should receive prepared artifacts from the 4090:

```text
repo snapshot
wheelhouse / conda pack
features/
windows/
counterfactuals/
configs/
```

Do not rely on H200 GPU nodes to download Hugging Face data, Python wheels, or timm checkpoints.

## Download

Install the Hub client on the machine doing the download:

```bash
python -m pip install huggingface_hub
```

Dry-run the default 20 x 5 plan:

```bash
python scripts/download_gm100_subset.py \
  --output-dir /data/WAM/raw/gm100-cobotmagic-lerobot_subset \
  --dry-run
```

Download the subset:

```bash
python scripts/download_gm100_subset.py \
  --output-dir /data/WAM/raw/gm100-cobotmagic-lerobot_subset
```

Download an even smaller laptop subset:

```bash
python scripts/download_gm100_subset.py \
  --tasks 2 \
  --episodes-per-task 2 \
  --output-dir /data/WAM/raw/gm100-cobotmagic-lerobot_tiny
```

Download explicit tasks with deterministic random episodes:

```bash
python scripts/download_gm100_subset.py \
  --task-ids task001,task002 \
  --episodes-per-task 2 \
  --random-episodes \
  --seed 42 \
  --output-dir /mnt/d/WAM/raw/gm100_task001_task002_random2
```

If `huggingface.co` is unreachable, use the mirror endpoint:

```bash
python scripts/download_gm100_subset.py \
  --endpoint https://hf-mirror.com \
  --task-ids task001,task002 \
  --episodes-per-task 2 \
  --random-episodes \
  --seed 42 \
  --output-dir /mnt/d/WAM/raw/gm100_task001_task002_random2
```

Select explicit tasks if the first sorted tasks are not useful:

```bash
python scripts/download_gm100_subset.py \
  --task-ids task_00001,task_00007,task_00013 \
  --episodes-per-task 5 \
  --output-dir /data/WAM/raw/gm100-cobotmagic-lerobot_subset
```

The script writes:

```text
gm100_subset_manifest.json
```

under the output directory after a successful download.

## Import Raw GM-100 To WAM Episodes

The first importer is label-free. It writes images and arrays for visual feature
extraction, but it intentionally does not write `labels.json`.

```bash
python -m ppwam.import_gm100 \
  --raw-root /mnt/d/WAM/raw/gm100_task001_task002_random2 \
  --output /mnt/d/WAM/episodes/gm100_task001_task002_random2 \
  --jpeg-quality 95 \
  --overwrite
```

For a fast smoke import:

```bash
python -m ppwam.import_gm100 \
  --raw-root /mnt/d/WAM/raw/gm100_task001_task002_random2 \
  --output /tmp/wam_gm100_import_smoke \
  --max-frames 16 \
  --overwrite
```

The importer writes one WAM episode directory per selected GM-100 episode:

```text
<episode_id>/
  meta.json
  arrays.npz
  import_manifest.json
  images/
    camera_top/
    camera_wrist_left/
    camera_wrist_right/
```

`arrays.npz` contains:

```text
proprio = observation.state.arm.position + observation.state.effector.position
action  = action.arm.position + action.effector.position
```

Both are 14-D for the current GM-100 Cobot Magic tasks.

## DINOv2 Checkpoint

MVP-0 currently extracts frozen features with `timm`:

```text
vit_base_patch14_dinov2.lvd142m
```

The first real feature extraction will download and cache the checkpoint automatically unless the cache already exists. On the 4090, keep the cache on persistent storage and include it when packaging offline H200 runs.

Recommended 4090 command after conversion:

```bash
python -m ppwam.extract_vision_features \
  --episodes /data/WAM/episodes/gm100_subset \
  --output /data/WAM/features/gm100_dinov2_base \
  --model vit_base_patch14_dinov2.lvd142m \
  --image-size 224 \
  --batch-size 128 \
  --device cuda
```

Recommended 5060 command:

```bash
python -m ppwam.extract_vision_features \
  --episodes /mnt/d/WAM/episodes/gm100_task001_task002_random2 \
  --output /mnt/d/WAM/features/gm100_task001_task002_dinov2_base \
  --model vit_base_patch14_dinov2.lvd142m \
  --image-size 224 \
  --batch-size 32 \
  --device cuda
```

If the 5060 runs out of memory, switch to:

```text
vit_small_patch14_dinov2.lvd142m
```

## Next Implementation Step

Add an importer:

```text
mvp0/import_gm100.py
```

Expected raw input:

```text
task_00001/
  meta/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.camera_top/episode_000000.mp4
  videos/chunk-000/observation.images.camera_wrist_left/episode_000000.mp4
  videos/chunk-000/observation.images.camera_wrist_right/episode_000000.mp4
```

Expected WAM output:

```text
episodes/<task_id>_<episode_id>/
  meta.json
  arrays.npz
  labels.json
  images/
    camera_top/
    camera_wrist_left/
    camera_wrist_right/
```

The importer must inspect the real parquet schema before finalizing:

- `proprio = concat(observation.state.arm.position, observation.state.effector.position)` initially;
- `action = concat(action.arm.position, action.effector.position)` initially;
- `num_frames` must match extracted video frames for every camera;
- primitive boundaries are not in GM-100 raw LeRobot metadata, so first pass should use a coarse placeholder boundary or a small manual label file.
