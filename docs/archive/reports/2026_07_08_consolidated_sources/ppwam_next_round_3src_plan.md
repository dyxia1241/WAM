# PP-WAM Next-Round Three-Source Plan

Date: 2026-07-07

Remote execution machine:

```text
5060: dayu@192.168.137.49
repo: /data/projects/WAM
ssh: /mnt/c/WINDOWS/System32/OpenSSH/ssh.exe
```

## 0. Current State

The local WSL repo and the 5060 repo are currently aligned:

```text
branch: main...origin/main
HEAD: 1e3ee94 Improve multisource prompt generation
tracked worktree: clean before this document was added
```

Formal three-source merge is complete on the 5060:

```text
prepared windows: data/prepared/ppwam_3src_equal_signal_v1
prompt features:  data/prompts/ppwam_3src_equal_siglip/prompt_features.npz
prompt table:     data/prompts/ppwam_3src_equal_siglip/prompt_table.jsonl
```

Merged dataset audit:

```text
num_windows = 75000
train: gm100 20000 / rh20t 20000 / reassemble 20000
val:   gm100 2500  / rh20t 2500  / reassemble 2500
test:  gm100 2500  / rh20t 2500  / reassemble 2500
canonical_action_dim = 14
canonical_proprio_dim = 14
canonical_num_cameras = 3
source_to_id = {"gm100": 0, "rh20t": 1, "reassemble": 2}
```

Merged prompt features:

```text
shape = (222, 768)
source counts = gm100 110 / rh20t 76 / reassemble 36
```

Loader and model sanity already passed with `configs/multisource/joint_flow_3src_equal_cf1p0.yaml`:

```text
obs     (3, 4, 3, 768)
proprio (3, 14)
action  (3, 8, 14)
prompt  (3, 768)
source_ids [0, 2, 1]
model_params = 2458959
```

## 1. Scientific Goal

The next round should answer a narrow question:

```text
Does joint future-observation/action/potential flow provide useful evidence beyond a strong phi-only critic when trained and evaluated across GM100, RH20T, and REASSEMBLE?
```

The current stance must stay conservative:

```text
Do not claim joint-flow beats phi-only unless source-stratified metrics support it.
```

Existing GM100 evidence shows that `cf1p0` joint-flow is viable but does not beat the strong phi-only baseline on the current hard/synthetic reranking setup. The next round therefore needs phi-only controls for every new joint-flow result.

## 2. Primary Deliverables

1. Train seed-42 three-source joint-flow and phi-only models.
2. Train seed-42 single-source RH20T and REASSEMBLE joint-flow and phi-only controls.
3. Reuse existing GM100 seed-42/43/44 checkpoints unless there is a code/config reason to rerun them.
4. Evaluate aggregate and source-stratified metrics for the three-source models.
5. Compare three-source training against single-source controls per source.
6. Decide whether multi-source transfer is promising enough to expand to seeds 43 and 44.
7. Produce a compact report that can update `docs/reports/ppwam_roadmap.md`, `docs/reports/checkpoint_registry.md`, and eventually the paper master plan.

## 3. Non-Negotiable Controls

Every joint-flow claim needs a phi-only counterpart:

| scope | joint-flow config | phi-only config |
| --- | --- | --- |
| 3-source equal | `configs/multisource/joint_flow_3src_equal_cf1p0.yaml` | `configs/multisource/phi_only_3src_equal_cf1p0.yaml` |
| GM100 | `configs/gm100/joint_flow_cf1p0.yaml` | `configs/gm100/phi_only_cf1p0.yaml` |
| RH20T | `configs/rh20t/joint_flow_cf1p0.yaml` | `configs/rh20t/phi_only_cf1p0.yaml` |
| REASSEMBLE | `configs/reassemble/joint_flow_cf1p0.yaml` | `configs/reassemble/phi_only_cf1p0.yaml` |

Primary comparison should be:

```text
3src joint-flow vs 3src phi-only
single-source joint-flow vs single-source phi-only
3src joint-flow on each source vs same-source joint-flow
3src phi-only on each source vs same-source phi-only
```

Do not compare a three-source model only against old GM100 aggregate metrics and call it a win.

## 4. Success Criteria

The minimum useful result is not simply a high aggregate score. The result is useful if it cleanly distinguishes one of these outcomes:

### Outcome A: Joint-flow wins or ties phi-only on three-source

Required evidence:

```text
3src joint-flow >= 3src phi-only on overall coarse_action_cf_ranking_acc
3src joint-flow >= 3src phi-only on at least 2 of 3 source-specific coarse ranking metrics
no source collapses below an obvious single-source baseline
```

Interpretation:

```text
Multi-source joint-flow is a real candidate for the next paper path.
Expand to seeds 43 and 44.
```

### Outcome B: Phi-only still wins, but joint-flow has source-specific value

Required evidence:

```text
phi-only wins overall, but joint-flow is better on one source or on calibration/MAE
the advantage is not caused by a tiny source-specific sample count
```

Interpretation:

```text
Joint-flow may need semantic/base-policy candidate sets or predictor-mode evaluation.
Do not scale blindly; inspect failure mode first.
```

### Outcome C: Phi-only wins everywhere

Required evidence:

```text
3src phi-only beats joint-flow overall and on every source
single-source phi-only beats joint-flow on RH20T and REASSEMBLE too
```

Interpretation:

```text
The critic/reranking story should foreground phi-only as the strong control.
Joint-flow must justify itself through future latent consistency or action-unknown predictor utility, not simple critic ranking.
```

### Outcome D: Multi-source hurts one or more sources

Required evidence:

```text
3src model underperforms the same-source model on that source by a large margin
source-stratified metrics reveal which source is being sacrificed
```

Interpretation:

```text
Investigate source normalization, action padding, prompt table mapping, camera padding, and source id conditioning before adding more seeds.
```

## 5. Execution Policy

Run on the 5060, not inside local WSL:

```bash
/mnt/c/WINDOWS/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=10 dayu@192.168.137.49 \
'cd /data/projects/WAM && <command>'
```

Use the project environment:

```bash
cd /data/projects/WAM
.conda/wam/bin/python -m ppwam.joint_flow --config <config>
```

Before any training run:

```bash
cd /data/projects/WAM
git status -sb
git rev-parse --short HEAD
test ! -e <output_dir> || find <output_dir> -maxdepth 3 -type f | sort | head -50
nvidia-smi
```

Do not overwrite existing result directories unless explicitly deciding that a run is disposable.

## 6. Phase 1: Preflight

Confirm repo and data:

```bash
cd /data/projects/WAM
git status -sb
git rev-parse --short HEAD

.conda/wam/bin/python - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path("data/prepared/ppwam_3src_equal_signal_v1")
idx = json.loads((root / "index.json").read_text())
print("num_windows", idx["num_windows"])
print("counts_by_split_source", idx["counts_by_split_source"])
print("params", idx["params"])

with np.load("data/prompts/ppwam_3src_equal_siglip/prompt_features.npz") as data:
    print("prompt features", data["features"].shape)
    print("task ids", len(data["task_ids"]))

table_ids = []
for line in Path("data/prompts/ppwam_3src_equal_siglip/prompt_table.jsonl").read_text().splitlines():
    table_ids.append(json.loads(line)["task_id"])
with np.load("data/prompts/ppwam_3src_equal_siglip/prompt_features.npz") as data:
    feature_ids = [str(x) for x in data["task_ids"]]
print("prompt_table_matches_features", table_ids == feature_ids)
PY
```

Confirm target output dirs are absent or intentionally reusable:

```bash
cd /data/projects/WAM
for d in \
  outputs/ppwam_3src_equal_cf1p0/seed_42 \
  outputs/ppwam_3src_equal_phi_only_cf1p0/seed_42 \
  outputs/rh20t_cf1p0/seed_42 \
  outputs/rh20t_phi_only_cf1p0/seed_42 \
  outputs/reassemble_cf1p0/seed_42 \
  outputs/reassemble_phi_only_cf1p0/seed_42
do
  if [ -e "$d" ]; then
    echo "EXISTS $d"
    find "$d" -maxdepth 3 -type f | sort | head -30
  else
    echo "MISSING $d"
  fi
done
```

Expected before a fresh seed-42 run:

```text
all six target dirs are MISSING
```

## 7. Phase 2: Seed-42 Training Matrix

Run the two three-source jobs first because they are the main comparison:

```bash
cd /data/projects/WAM
.conda/wam/bin/python -m ppwam.joint_flow --config configs/multisource/joint_flow_3src_equal_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/multisource/phi_only_3src_equal_cf1p0.yaml
```

Then run the RH20T single-source controls:

```bash
cd /data/projects/WAM
.conda/wam/bin/python -m ppwam.joint_flow --config configs/rh20t/joint_flow_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/rh20t/phi_only_cf1p0.yaml
```

Then run the REASSEMBLE single-source controls:

```bash
cd /data/projects/WAM
.conda/wam/bin/python -m ppwam.joint_flow --config configs/reassemble/joint_flow_cf1p0.yaml
.conda/wam/bin/python -m ppwam.joint_flow --config configs/reassemble/phi_only_cf1p0.yaml
```

Default config details:

```text
seed = 42
batch_size = 96
max_epochs = 12
save_best_by = val/coarse_action_cf_ranking_acc
optimizer = adamw
lr = 2.0e-4
precision = fp32
```

Expected run outputs:

```text
outputs/ppwam_3src_equal_cf1p0/seed_42/mvp1_joint_flow/best.pt
outputs/ppwam_3src_equal_cf1p0/seed_42/mvp1_joint_flow/metrics.json
outputs/ppwam_3src_equal_cf1p0/seed_42/mvp1_joint_flow/history.jsonl
outputs/ppwam_3src_equal_cf1p0/seed_42/mvp1_joint_flow/eval_test/metrics.json
```

The phi-only and single-source runs follow the same nested `mvp1_joint_flow/` structure.

## 8. Phase 3: Immediate Result Audit

After each run, inspect the best validation metrics and test metrics:

```bash
cd /data/projects/WAM
.conda/wam/bin/python - <<'PY'
import json
from pathlib import Path

runs = {
    "3src_joint": "outputs/ppwam_3src_equal_cf1p0/seed_42/mvp1_joint_flow",
    "3src_phi": "outputs/ppwam_3src_equal_phi_only_cf1p0/seed_42/mvp1_joint_flow",
    "rh20t_joint": "outputs/rh20t_cf1p0/seed_42/mvp1_joint_flow",
    "rh20t_phi": "outputs/rh20t_phi_only_cf1p0/seed_42/mvp1_joint_flow",
    "reassemble_joint": "outputs/reassemble_cf1p0/seed_42/mvp1_joint_flow",
    "reassemble_phi": "outputs/reassemble_phi_only_cf1p0/seed_42/mvp1_joint_flow",
}

keys = [
    "delta_phi_mae",
    "delta_phi_rmse",
    "coarse_action_cf_ranking_acc",
    "coarse_action_cf_top1_acc",
    "all_negatives_tie_aware_ranking_acc",
    "all_negatives_mean_margin",
]

for name, run in runs.items():
    print(f"\n## {name}")
    for label, path in [("val", Path(run) / "metrics.json"), ("test", Path(run) / "eval_test" / "metrics.json")]:
        if not path.exists():
            print(label, "MISSING", path)
            continue
        metrics = json.loads(path.read_text())
        print(label)
        for key in keys:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.6f}")
        source_keys = [k for k in sorted(metrics) if k.startswith("source_") and (
            k.endswith("coarse_action_cf_ranking_acc")
            or k.endswith("coarse_action_cf_top1_acc")
            or k.endswith("delta_phi_mae")
            or k.endswith("all_negatives_tie_aware_ranking_acc")
        )]
        for key in source_keys:
            print(f"  {key}: {metrics[key]:.6f}")
PY
```

Required checks:

```text
metrics.json exists for train-selected best validation metrics
eval_test/metrics.json exists for final test metrics
source_gm100_* metrics exist for 3src runs
source_rh20t_* metrics exist for 3src runs
source_reassemble_* metrics exist for 3src runs
source-specific window counts match the expected test split scale
```

If source-specific keys are missing, stop and inspect `ppwam/joint_flow.py` and the dataloader output. Do not continue to multi-seed runs without source-stratified metrics.

## 9. Phase 4: Comparison Table

Create a manual or scripted table with these columns:

```text
model
scope
seed
split
overall_mae
overall_rmse
overall_coarse_ranking
overall_coarse_top1
overall_all_neg_ranking
gm100_mae
gm100_coarse_ranking
gm100_coarse_top1
rh20t_mae
rh20t_coarse_ranking
rh20t_coarse_top1
reassemble_mae
reassemble_coarse_ranking
reassemble_coarse_top1
checkpoint
```

The first table can be seed-42 only. It should answer:

```text
Is 3src joint-flow better than 3src phi-only overall?
Is 3src joint-flow better than 3src phi-only on each source?
Does 3src training help or hurt each source compared with single-source training?
Are failures ranking failures, calibration failures, or both?
```

## 10. Phase 5: Decision Gate for Seeds 43 and 44

Run seeds 43 and 44 only after the seed-42 table is readable.

Good reasons to expand:

```text
3src joint-flow wins or ties phi-only on the main source-stratified metrics
3src has an interesting transfer effect on at least one source
seed-42 results contradict existing GM100 priors and need stability check
```

Bad reasons to expand:

```text
seed-42 source metrics are missing
there is an obvious data/prompt/action-dim bug
joint-flow loses everywhere and the failure mode is already clear
```

If expanding, avoid editing tracked config files just to change seeds. Use overrides:

```bash
cd /data/projects/WAM

.conda/wam/bin/python -m ppwam.joint_flow \
  --config configs/multisource/joint_flow_3src_equal_cf1p0.yaml \
  seed=43 output_dir=outputs/ppwam_3src_equal_cf1p0/seed_43

.conda/wam/bin/python -m ppwam.joint_flow \
  --config configs/multisource/phi_only_3src_equal_cf1p0.yaml \
  seed=43 output_dir=outputs/ppwam_3src_equal_phi_only_cf1p0/seed_43
```

Repeat for seed 44 and for single-source runs only if seed-42 indicates they are worth stabilizing.

## 11. Known Risks and Diagnostics

### Source normalization risk

Three-source data uses:

```text
action.representation = source_zscore_padded_absolute_proxy
canonical_action_dim = 14
canonical_proprio_dim = 14
canonical_num_cameras = 3
norm_stats = null
```

If one source collapses, inspect whether source-specific z-score and padding are being applied correctly. A collapse can look like good aggregate metrics if another source dominates easier negatives.

### Prompt id risk

Merged prompt ids are prefixed as:

```text
gm100::<raw_task_id>
rh20t::<raw_task_id>
reassemble::<raw_task_id>
```

If prompt features look random or repeated, recheck:

```text
data/prompts/ppwam_3src_equal_siglip/prompt_table.jsonl
data/prompts/ppwam_3src_equal_siglip/prompt_features.npz
```

The prompt table ids must exactly match the `task_ids` array in the npz.

### Camera padding risk

The canonical camera count is 3. RH20T and REASSEMBLE may not naturally match GM100's camera structure. If source-specific observation metrics or ranking are odd, inspect whether padded camera slots are masked or being treated as real latent signals.

### Aggregate metric risk

Overall metrics can hide source imbalance or source-specific collapse. Treat source-stratified metrics as primary for the three-source model.

### Phi-only overstrength risk

Phi-only is not a weak baseline. It has the same prompt, observation history, proprioception, action chunk, phi target, and counterfactual supervision, but removes future-observation/action flow losses:

```text
obs_weight = 0.0
action_weight = 0.0
critic_flow_weight = 0.0
action_condition_prob = 1.0
```

If joint-flow loses, the conclusion is not automatically that PP-WAM is wrong. It may mean that critic-only reranking does not use the extra modeled variables.

## 12. Reporting Template

Write the next result report around these statements:

```text
We trained matched joint-flow and phi-only PP-WAM variants on an equal three-source dataset spanning GM100, RH20T, and REASSEMBLE.
```

```text
We report both aggregate and source-stratified counterfactual ranking metrics because aggregate performance can hide cross-source transfer or collapse.
```

```text
The phi-only model is treated as a strong control, not as an ablation-only strawman.
```

Then include:

1. Data recap.
2. Training matrix.
3. Aggregate seed-42 table.
4. Source-stratified seed-42 table.
5. Single-source vs three-source comparison.
6. Decision on whether to run seeds 43 and 44.
7. Next technical action.

## 13. If Seed-42 Is Bad

Do not immediately tune hyperparameters. First check:

```text
data index source ids
prompt id alignment
source-specific counts
action/proprio canonical dims
norm stats semantics
source-stratified metric names and counts
history.jsonl for divergence or unstable validation
```

Only after those pass, consider:

```text
lower lr
larger hidden_dim/layers
source embedding or source-conditioned normalization
different source sampling weights
separate loss weights per source
semantic/base-policy candidate reranking
```

## 14. Immediate Next Command

The next action should be the preflight audit on 5060:

```bash
/mnt/c/WINDOWS/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=10 dayu@192.168.137.49 \
'cd /data/projects/WAM && git status -sb && git rev-parse --short HEAD && nvidia-smi'
```

If clean and no target output dirs exist, start:

```bash
/mnt/c/WINDOWS/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=10 dayu@192.168.137.49 \
'cd /data/projects/WAM && .conda/wam/bin/python -m ppwam.joint_flow --config configs/multisource/joint_flow_3src_equal_cf1p0.yaml'
```
