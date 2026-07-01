# GM-100 Prompt Comprehensive Experiment Report

- Machine: 5060 (`dayu@192.168.137.49`)
- Current code commit: `f198f88b138b74f73129b18d5018864e0c3331d8`
- Dataset/features: same GM-100 50x5 light split, frozen DINOv2 visual features, history=4, horizon=8, stride=2
- Prompt encoder: frozen `google/siglip-base-patch16-224`
- Seeds: `42, 43, 44`
- Output root on 5060: `outputs/gm100_prompt_comprehensive_report`

This report merges:

- the first stage-conditioned action experiment: `outputs/gm100_action_fair_s42_44`;
- the prompt-conditioned formal experiment: `outputs/gm100_prompt_formal`;
- the prompt CF `loss.delta_weight` sweep: `outputs/gm100_prompt_phi_weight_sweep`.

## Metric Definitions

- `DeltaPhi MAE/RMSE`: prediction error for primitive-local `DeltaPhi`.
- `all-neg ranking`: tie-aware rate that the positive action has higher predicted potential than generated negative actions.
- `all-neg margin`: mean `pred_delta_phi(positive) - pred_delta_phi(negative)` over all evaluated negative types.
- `+/-`: sample standard deviation across seeds.

## Main Comparison

| family | label | delta_weight | seeds | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin | zero | shuffle | wrong_arm | scaled_1.75 | reverse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage_first_experiment | `stage_obs` | -- | 3 | 0.0139+/-0.0006 | 0.0349+/-0.0040 | 0.5000+/-0.0000 | 0.0000+/-0.0000 | 0.5000+/-0.0000 | 0.5000+/-0.0000 | 0.5000+/-0.0000 | 0.5000+/-0.0000 | 0.5000+/-0.0000 |
| stage_first_experiment | `stage_action` | -- | 3 | 0.0125+/-0.0017 | 0.0326+/-0.0034 | 0.5048+/-0.0640 | -0.0002+/-0.0003 | 0.5565+/-0.0982 | 0.4996+/-0.0032 | 0.4227+/-0.1060 | 0.4126+/-0.1093 | 0.4998+/-0.0020 |
| stage_first_experiment | `stage_action_cf` | -- | 3 | 0.0223+/-0.0043 | 0.0360+/-0.0006 | 0.8196+/-0.0272 | 0.0162+/-0.0025 | 1.0000+/-0.0000 | 0.5790+/-0.0359 | 0.9565+/-0.0222 | 0.8667+/-0.1035 | 0.5157+/-0.0120 |
| prompt_formal | `prompt_obs` | -- | 3 | 0.0120+/-0.0006 | 0.0321+/-0.0003 | 0.5000+/-0.0000 | 0.0000+/-0.0000 | 0.5000+/-0.0000 | 0.5000+/-0.0000 | 0.5000+/-0.0000 | 0.5000+/-0.0000 | 0.5000+/-0.0000 |
| prompt_formal | `prompt_action` | -- | 3 | 0.0132+/-0.0011 | 0.0346+/-0.0024 | 0.4792+/-0.0335 | 0.0005+/-0.0002 | 0.5456+/-0.0855 | 0.5005+/-0.0060 | 0.3135+/-0.0215 | 0.5317+/-0.0957 | 0.5047+/-0.0042 |
| prompt_formal | `prompt_cf_w1` | 1 | 3 | 0.0236+/-0.0074 | 0.0410+/-0.0144 | 0.7701+/-0.0440 | 0.0121+/-0.0063 | 0.9917+/-0.0145 | 0.5523+/-0.0507 | 0.9010+/-0.0629 | 0.8659+/-0.0891 | 0.5398+/-0.0512 |
| prompt_phi_weight_sweep | `prompt_cf_w2` | 2 | 3 | 0.0166+/-0.0024 | 0.0396+/-0.0087 | 0.8013+/-0.0218 | 0.0111+/-0.0008 | 1.0000+/-0.0000 | 0.5378+/-0.0237 | 0.9038+/-0.0557 | 0.8572+/-0.0522 | 0.5093+/-0.0059 |
| prompt_phi_weight_sweep | `prompt_cf_w5` | 5 | 3 | 0.0139+/-0.0020 | 0.0345+/-0.0043 | 0.8057+/-0.0074 | 0.0076+/-0.0022 | 0.9982+/-0.0031 | 0.5340+/-0.0081 | 0.9217+/-0.0310 | 0.8676+/-0.0103 | 0.5174+/-0.0170 |
| prompt_phi_weight_sweep | `prompt_cf_w10` | 10 | 3 | 0.0125+/-0.0006 | 0.0330+/-0.0003 | 0.7512+/-0.0215 | 0.0066+/-0.0001 | 1.0000+/-0.0000 | 0.5137+/-0.0276 | 0.7897+/-0.1000 | 0.7029+/-0.0349 | 0.5014+/-0.0047 |
| prompt_phi_weight_sweep | `prompt_cf_w20` | 20 | 3 | 0.0121+/-0.0008 | 0.0321+/-0.0019 | 0.7243+/-0.0226 | 0.0056+/-0.0006 | 1.0000+/-0.0000 | 0.5214+/-0.0134 | 0.6641+/-0.0985 | 0.6567+/-0.0338 | 0.5038+/-0.0021 |

## Phi Weight Sweep

| loss.delta_weight | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin | zero | shuffle | wrong_arm | scaled_1.75 | reverse |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0236+/-0.0074 | 0.0410+/-0.0144 | 0.8065+/-0.0339 | 0.0144+/-0.0073 | 0.9917+/-0.0145 | 0.5483+/-0.0427 | 0.9010+/-0.0629 | 0.8659+/-0.0891 | 0.5398+/-0.0512 |
| 2 | 0.0166+/-0.0024 | 0.0396+/-0.0087 | 0.8013+/-0.0218 | 0.0111+/-0.0008 | 1.0000+/-0.0000 | 0.5378+/-0.0237 | 0.9038+/-0.0557 | 0.8572+/-0.0522 | 0.5093+/-0.0059 |
| 5 | 0.0139+/-0.0020 | 0.0345+/-0.0043 | 0.8057+/-0.0074 | 0.0076+/-0.0022 | 0.9982+/-0.0031 | 0.5340+/-0.0081 | 0.9217+/-0.0310 | 0.8676+/-0.0103 | 0.5174+/-0.0170 |
| 10 | 0.0125+/-0.0006 | 0.0330+/-0.0003 | 0.7512+/-0.0215 | 0.0066+/-0.0001 | 1.0000+/-0.0000 | 0.5137+/-0.0276 | 0.7897+/-0.1000 | 0.7029+/-0.0349 | 0.5014+/-0.0047 |
| 20 | 0.0121+/-0.0008 | 0.0321+/-0.0019 | 0.7243+/-0.0226 | 0.0056+/-0.0006 | 1.0000+/-0.0000 | 0.5214+/-0.0134 | 0.6641+/-0.0985 | 0.6567+/-0.0338 | 0.5038+/-0.0021 |

## Main Findings

- The first stage-conditioned CF model (`stage_action_cf`) has the strongest all-negative ranking among the main comparisons: `0.8196+/-0.0272`, but its DeltaPhi calibration is weaker than high-weight prompt CF settings.
- The prompt CF baseline at `loss.delta_weight=1` has high action sensitivity but poor DeltaPhi MAE/RMSE.
- Raising `loss.delta_weight` improves DeltaPhi calibration monotonically in the tested range, from MAE `0.0236` at weight 1 to `0.0121` at weight 20.
- The tradeoff is reduced action sensitivity: all-neg ranking drops from `0.8065` at weight 1 to `0.7243` at weight 20, and all-neg margin drops from `0.0144` to `0.0056`.
- `loss.delta_weight=10` and `20` are the best calibrated settings in this sweep. Weight 10 keeps more ranking sensitivity; weight 20 gives the best MAE/RMSE.

## Figure Artifacts

Generated on 5060:

- `outputs/gm100_prompt_comprehensive_report/figures/comparison_delta_phi_mae.png`
- `outputs/gm100_prompt_comprehensive_report/figures/comparison_delta_phi_rmse.png`
- `outputs/gm100_prompt_comprehensive_report/figures/comparison_all_negative_ranking.png`
- `outputs/gm100_prompt_comprehensive_report/figures/comparison_all_negative_margin.png`
- `outputs/gm100_prompt_comprehensive_report/figures/sweep_all_negative_ranking.png`
- `outputs/gm100_prompt_comprehensive_report/figures/sweep_all_negative_margin.png`
- `outputs/gm100_prompt_comprehensive_report/figures/sweep_per_negative_ranking.png`

Summary files:

- `outputs/gm100_prompt_comprehensive_report/summary/aggregate_metrics.csv`
- `outputs/gm100_prompt_comprehensive_report/summary/metrics_by_seed.csv`
- `outputs/gm100_prompt_comprehensive_report/summary/experiment_report.md`

## Verification

- 5060 full action eval completed for all 15 phi-weight sweep checkpoints.
- 5060 report tests: `5 passed` for `test_comprehensive_report.py` and `test_prompt_phi_weight_sweep.py`.
- Figure PNGs were copied locally and inspected for nonblank rendering and readable labels.
