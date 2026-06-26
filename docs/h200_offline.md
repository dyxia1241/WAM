# H200 Offline Workflow

H200 GPU nodes cannot rely on network access. Prepare everything on the 4090 or a login node before submitting GPU jobs.

## 1. Prepare Offline Inputs

Required inputs:

```text
repo snapshot
wheelhouse or conda-pack environment
prepared windows
pre-extracted features
configs
submit script
```

Do not download model weights or Python packages on GPU nodes.

## 2. Build A Wheelhouse

On a networked machine:

```bash
scripts/pack_offline_env.sh /data/WAM/offline/wheelhouse
```

The script downloads wheels for `requirements.txt` into a local directory.

## 3. Sync Artifacts

From 4090 to the cluster staging area:

```bash
scripts/sync_artifacts.sh /data/WAM user@cluster:/path/to/WAM_artifacts
```

Adjust the destination path for the lab environment.

## 4. Submit Template

Generate a Slurm template:

```bash
scripts/submit_h200.sh /path/to/WAM /path/to/WAM_artifacts
```

Edit partition/account/module lines according to the lab cluster.

