# WAM

WAM is currently focused on MVP-0: a lightweight action-grounded process critic for primitive-local robot manipulation progress.

The MVP-0 critic predicts:

```text
(observation history, proprioception, candidate future action chunk, stage, task) -> delta_phi
```

The first target is a sanity check, not a full world model:

- use frozen transformer visual features;
- train a small Stage-FiLM Transformer Critic;
- compare against `time_prior`, `obs_stage`, `obs_action`, and `obs_action_stage` baselines;
- use simple counterfactual action ranking as the main signal.

See:

- [mvp0_design.md](mvp0_design.md)
- [device_compute_plan.md](device_compute_plan.md)

## Local Development

Install minimal dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest
```

Run the CPU toy pipeline:

```bash
python -m mvp0.train --config mvp0/configs/debug.yaml experiment=time_prior
python -m mvp0.train --config mvp0/configs/debug.yaml experiment=obs_action_stage_cf
python -m mvp0.eval --checkpoint outputs/obs_action_stage_cf/best.pt --split test
python -m mvp0.plot --eval outputs/obs_action_stage_cf/eval
```

The first implementation phase is CPU/toy-data only. Real data, features, checkpoints, and outputs are intentionally ignored by git.
