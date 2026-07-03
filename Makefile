PYTHON ?= .venv/bin/python
CONFIG ?= configs/debug.yaml
OUTPUTS ?= outputs

.PHONY: test smoke train-toy eval-toy plot-toy ablation report prepare-counterfactuals

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m ppwam.smoke --root /tmp/wam_smoke

train-toy:
	$(PYTHON) -m ppwam.train --config $(CONFIG) experiment=obs_action_stage_cf

eval-toy:
	$(PYTHON) -m ppwam.eval --checkpoint $(OUTPUTS)/obs_action_stage_cf/best.pt --split test

plot-toy:
	$(PYTHON) -m ppwam.plot --eval $(OUTPUTS)/obs_action_stage_cf/eval

ablation:
	$(PYTHON) -m ppwam.run_ablation --config $(CONFIG) --output-dir $(OUTPUTS)

report:
	$(PYTHON) -m ppwam.reports --outputs $(OUTPUTS) --output $(OUTPUTS)/report

prepare-counterfactuals:
	$(PYTHON) -m ppwam.make_counterfactuals --windows data/windows --output data/counterfactuals
