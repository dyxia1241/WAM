# PP-WAM

PP-WAM is a compact research codebase for primitive-local process potential in robot manipulation.

Current main line:

```text
language/prompt + observation history + proprioception + candidate action
  -> primitive-local DeltaPhi / process potential
```

The newest MVP1 experiments use a lightweight DiT-style joint flow over future observation latents, action chunks, and process-potential tokens. The current strongest GM-100 validation config is `configs/gm100/joint_flow_cf1p0.yaml`.

Current GM-100 headline result:

| model | DeltaPhi MAE | coarse ranking | all-neg ranking | coarse top-1 |
| --- | ---: | ---: | ---: | ---: |
| MVP1 V2 | 0.0158+/-0.0040 | 0.7816+/-0.0369 | 0.6881+/-0.0241 | 0.4102+/-0.0512 |
| MVP1.6 `cf_1p0` | 0.0187+/-0.0027 | 0.8870+/-0.0231 | 0.7801+/-0.0132 | 0.7301+/-0.0543 |

See [docs/README.md](docs/README.md) for the compact report index.

## Layout

```text
ppwam/              main package
mvp0/               temporary compatibility wrappers for older commands
configs/            active and archived YAML configs
docs/reports/       current concise experiment reports
docs/archive/       older reports and figures
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

`python -m mvp0.<module>` remains available as a temporary compatibility path, but new code and docs should use `ppwam`.
