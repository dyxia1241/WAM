# PP-WAM

PP-WAM is a compact research codebase for process-potential-guided robot
manipulation world-action modeling.

Current main line:

```text
language/prompt + observation history + proprioception
  -> future action chunk + future observation latent + future process potential
```

The active direction is in-domain perturbation recovery: use signed process
gain to rerank or select action futures when a nominal policy enters hesitation,
detour, overshoot, or recovery behavior.

Current status:

```text
RoboTwin Sim-SubSuccess: 20 tasks x 3 perturbations
signed labels: phi_t, phi_future, delta_phi_raw
active model path: potential-guided joint flow
```

See [docs/reports/ppwam_status.md](docs/reports/ppwam_status.md) for current
status and [docs/reports/ppwam_paper_plan.md](docs/reports/ppwam_paper_plan.md)
for the compact paper plan.

## Layout

```text
ppwam/              main package
mvp0/               temporary compatibility wrappers for older commands
configs/            active and archived YAML configs
docs/reports/       active compact status and paper plan
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
