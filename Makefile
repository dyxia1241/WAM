PYTHON ?= .venv/bin/python
CONFIG ?= mvp0/configs/debug.yaml
OUTPUTS ?= outputs

.PHONY: test train-toy eval-toy plot-toy ablation report prepare-counterfactuals

test:
	$(PYTHON) -m pytest

train-toy:
	$(PYTHON) -m mvp0.train --config $(CONFIG) experiment=obs_action_stage_cf

eval-toy:
	$(PYTHON) -m mvp0.eval --checkpoint $(OUTPUTS)/obs_action_stage_cf/best.pt --split test

plot-toy:
	$(PYTHON) -m mvp0.plot --eval $(OUTPUTS)/obs_action_stage_cf/eval

ablation:
	$(PYTHON) -m mvp0.run_ablation --config $(CONFIG) --output-dir $(OUTPUTS)

report:
	$(PYTHON) -m mvp0.reports --outputs $(OUTPUTS) --output $(OUTPUTS)/report

prepare-counterfactuals:
	$(PYTHON) -m mvp0.make_counterfactuals --windows data/windows --output data/counterfactuals

