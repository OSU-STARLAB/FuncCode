# Branch-Aware Base/Spline Codebook Patch

Copy/overwrite these files in your current FuncCode-KAN MNIST codebase:

```text
src/models.py
src/clustering.py
src/quantization.py
src/storage.py
src/experiment_mnist.py
run_branch_aware.sh
```

This patch preserves the existing coefficient-space, fixed-grid function-space, and activation-domain function-space modes, and adds:

```bash
--cluster-method branch
--branch-spline-clusters 16
--branch-base-clusters 8
--branch-spline-method function
--branch-spline-method coefficient
--branch-function-samples 128
--branch-function-domain grid
--branch-function-domain activation
```

Smoke test:

```bash
bash run_branch_aware.sh smoke
```

Main comparisons:

```bash
bash run_branch_aware.sh compare_k16
bash run_branch_aware.sh compare_k32
bash run_branch_aware.sh compare_k64
```

Manual example:

```bash
python -m src.experiment_mnist \
  --run-name branch_k16_s16_b8 \
  --cluster-method branch \
  --branch-spline-method function \
  --branch-spline-clusters 16 \
  --branch-base-clusters 8 \
  --branch-function-samples 128 \
  --branch-function-domain grid \
  --epochs 10 \
  --finetune-epochs 20 \
  --width 64 \
  --clusters 16 \
  --bits-list 8 6 4 3 2 \
  --batch-size 1024 \
  --test-batch-size 2048
```
