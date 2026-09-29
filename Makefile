.PHONY: install test smoke smoke-mnist smoke-cifar smoke-conv lint-imports

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -q

# Tiny synthetic pipeline test (no dataset download)
smoke:
	pytest tests/test_pipeline_smoke.py -q

# 1-epoch end-to-end runs (downloads MNIST / CIFAR)
smoke-mnist:
	bash scripts/mnist/run_spline_baseline.sh smoke
	bash scripts/mnist/run_all_kan.sh smoke

smoke-cifar:
	bash scripts/cifar/run_cifar_experiments.sh cifar10_smoke

# 2-epoch wiring check of the convolutional KAGN track
smoke-conv:
	bash scripts/paper/09_conv_cifar.sh smoke
