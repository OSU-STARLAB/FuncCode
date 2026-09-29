# CIFAR-10 / CIFAR-100 Experiment Patch for FuncCode-KAN

## Progress Tracker

```text
[done] MNIST experiments completed
[done] CIFAR experiment patch created
[now] Run CIFAR-10 smoke test
[next] Run CIFAR-10 main single-seed experiment
[next] Run CIFAR-10 multi-seed experiment
[next] Run CIFAR-100 smoke test
[next] Run CIFAR-100 main experiment
```

## Install

```bash
unzip CIFAR_Experiments_patch.zip
chmod +x scripts/run_cifar_experiments.sh
```

## CIFAR-10 smoke

```bash
bash scripts/run_cifar_experiments.sh cifar10_smoke
```

## CIFAR-10 main

```bash
bash scripts/run_cifar_experiments.sh cifar10_main
```

## CIFAR-10 multi-seed

```bash
bash scripts/run_cifar_experiments.sh cifar10_multiseed
```

## Summarize

```bash
python tools/summarize_cifar_runs.py --root runs_cifar --out runs_cifar/cifar_summary.csv
```

## Make LaTeX table

```bash
python tools/make_cifar_latex_table.py   --csv runs_cifar/cifar_summary.csv   --out-tex runs_cifar/cifar10_table.tex   --dataset cifar10   --stage clustered_hwq_w4   --best-only   --caption "CIFAR-10 compressed KAN results under W4 codebook quantization."   --label "tab:cifar10_results"
```

## CIFAR-100

```bash
bash scripts/run_cifar_experiments.sh cifar100_smoke
bash scripts/run_cifar_experiments.sh cifar100_main
```

This first CIFAR version uses flattened CIFAR images, so it is directly comparable to the MNIST pipeline. If dense CIFAR accuracy is too low, improve dense architecture/training before interpreting compression.
