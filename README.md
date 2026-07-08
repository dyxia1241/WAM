# PP-WAM

PP-WAM is a compact research codebase for primitive-local process potential in robot manipulation.

Current main line:

```text
language/prompt + observation history + proprioception + candidate action
  -> primitive-local DeltaPhi / process potential
```

The newest MVP1 experiments use a lightweight DiT-style joint flow over future observation latents, action chunks, and process-potential tokens. The current main GM-100 joint-flow config is `configs/gm100/joint_flow_cf1p0.yaml`.

Current conclusion:

```text
phi-only is the stronger current critic/reranker;
joint-flow is still useful as a world-action-potential model candidate because
it learns future action and future observation structure.
```

The next evidence focus is source controls, semantic/base-policy candidate reranking, future-latent/action consistency, and predictor-mode utility.

See [docs/reports/ppwam_current_report.md](docs/reports/ppwam_current_report.md) for the current consolidated report.

## Layout

```text
ppwam/              main package
mvp0/               temporary compatibility wrappers for older commands
configs/            active and archived YAML configs
docs/reports/       current consolidated report
docs/archive/       older reports, source reports, and figures
scripts/            thin CLI wrappers and utility shell scripts
tests/              unit and smoke tests
data/, outputs/     local artifacts, ignored by git
```

## Development

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest
```

Run the toy smoke pipeline:

```bash
python -m ppwam.smoke --root /tmp/wam_smoke
```

Run a small toy training/eval path:

```bash
python -m ppwam.train --config configs/debug.yaml experiment=obs_action_stage_cf
python -m ppwam.eval --checkpoint outputs/obs_action_stage_cf/best.pt --split test
python -m ppwam.plot --eval outputs/obs_action_stage_cf/eval
```

Run the current GM-100 joint-flow entrypoint on the 5060:

```bash
python -m ppwam.joint_flow --config configs/gm100/joint_flow_cf1p0.yaml
```

Run hard candidate reranking from an existing checkpoint:

```bash
python -m ppwam.joint_flow_rerank --checkpoint outputs/gm100_mvp1_6_cf1p0/seed_42/mvp1_joint_flow/best.pt --split test
```

`python -m mvp0.<module>` remains available as a temporary compatibility path, but new code and docs should use `ppwam`.
