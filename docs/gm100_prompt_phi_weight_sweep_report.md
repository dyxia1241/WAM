# GM-100 Prompt Phi Loss Weight Sweep

- Machine: 5060 (`dayu@192.168.137.49`)
- Code commit used for the sweep: `1e42536cddd47cb3fc24a941304d0c936e424eb8`
- Experiment: `obs_action_prompt_cf_multi`
- Changed parameter: `loss.delta_weight`
- Fixed CF parameter: `loss.counterfactual_weight=0.1`
- Seeds: `42, 43, 44`
- Evaluation: prediction-only test `DeltaPhi` MAE/RMSE
- Output root on 5060: `outputs/gm100_prompt_phi_weight_sweep`

Run command:

```bash
.conda/wam/bin/python -m mvp0.prompt_phi_weight_sweep \
  --config mvp0/configs/gm100_prompt_formal.yaml \
  --output-dir outputs/gm100_prompt_phi_weight_sweep \
  --weights 1,2,5,10,20 \
  --seeds 42,43,44
```

## Results

| loss.delta_weight | DeltaPhi MAE | DeltaPhi RMSE |
| ---: | ---: | ---: |
| 1 | 0.0236+/-0.0074 | 0.0410+/-0.0144 |
| 2 | 0.0166+/-0.0024 | 0.0396+/-0.0087 |
| 5 | 0.0139+/-0.0020 | 0.0345+/-0.0043 |
| 10 | 0.0125+/-0.0006 | 0.0330+/-0.0003 |
| 20 | 0.0121+/-0.0008 | 0.0321+/-0.0019 |

Increasing the DeltaPhi regression weight consistently improved prediction calibration in this sweep. The best mean MAE/RMSE among the tested values was at `loss.delta_weight=20`, with `10` close behind and slightly lower RMSE variance.

The `+/-` values use sample standard deviation across seeds. Ranking and margin metrics for these same checkpoints are reported in `docs/gm100_prompt_comprehensive_report.md`.

## Figure Artifacts

Generated on 5060:

- `outputs/gm100_prompt_phi_weight_sweep/figures/test_delta_phi_mae.png`
- `outputs/gm100_prompt_phi_weight_sweep/figures/test_delta_phi_mae.svg`
- `outputs/gm100_prompt_phi_weight_sweep/figures/test_delta_phi_rmse.png`
- `outputs/gm100_prompt_phi_weight_sweep/figures/test_delta_phi_rmse.svg`

Summary files:

- `outputs/gm100_prompt_phi_weight_sweep/summary/aggregate_eval_test_predict_phi.csv`
- `outputs/gm100_prompt_phi_weight_sweep/summary/eval_test_predict_phi_by_seed.csv`
- `outputs/gm100_prompt_phi_weight_sweep/summary/experiment_report.md`

## Verification

- 5060 full pytest: `86 passed, 12 warnings`
- Sweep completed all 15 runs: 5 weights x 3 seeds
- Full action-sensitivity eval completed for all 15 checkpoints
- PNG figures were checked locally after copying from 5060
