# Function-Space Clustering Patch

Copy these files into your existing standalone MNIST codebase:

```text
src/function_space.py
src/clustering.py
src/experiment_mnist.py
run_function_space.sh
```

This patch adds:

```bash
--cluster-method coefficient
--cluster-method function
--function-samples 128
--function-domain grid
--function-domain activation
--include-base-in-function
```

Smoke test:

```bash
bash run_function_space.sh smoke
```

Main comparisons:

```bash
bash run_function_space.sh compare_k16
bash run_function_space.sh compare_k32
bash run_function_space.sh compare_k64
bash run_function_space.sh compare_k128
```

Manual function-space run:

```bash
python -m src.experiment_mnist \
  --run-name func_k64 \
  --cluster-method function \
  --function-samples 128 \
  --function-domain grid \
  --include-base-in-function \
  --epochs 10 \
  --finetune-epochs 20 \
  --width 64 \
  --clusters 64 \
  --bits-list 8 6 4 3 2 \
  --batch-size 1024 \
  --test-batch-size 2048
```
