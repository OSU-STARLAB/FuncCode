"""Driver preparing the three HW models (see the design contract M1/M2).

  --model branch_source   run the verified mnist pipeline (branch Ks=32 Kb=16)
                          -> runs_hw/branch_k32_s32_b16/{dense.pt,
                             clustered_finetuned.pt, summary.csv}
                          (supplies D1 weights + D2 init + D3 init)
  --model lsq_w4a4        D2: LSQ W4A4 QAT from the dense checkpoint
  --model funccode_w4a4   D3: LSQ A4 + W4 codebook fake-quant finetune of the
                          verified branch model (indices frozen)

All outputs land under runs_hw/. Storage numbers for D3 come from
funcodekan.analysis.storage (never re-derived by hand).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from funcodekan.analysis.storage import (add_kib_columns, bits_to_kib,
                                         compressed_storage_breakdown,
                                         compression_ratio,
                                         dense_fp32_bits_from_model,
                                         hwq_bits_from_model)
from funcodekan.data.mnist import get_mnist_loaders
from funcodekan.hw.act_quant import QuantBranchKAN, rebuild_branch_model
from funcodekan.hw.qat import QuantSplineKAN, run_qat
from funcodekan.hw.veriflog import append_verification_entry
from funcodekan.models.spline import DenseSplineKAN
from funcodekan.utils.training import get_device, set_seed

BRANCH_SOURCE_RUN = "branch_k32_s32_b16"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["branch_source", "lsq_w4a4", "funccode_w4a4"])
    p.add_argument("--out-dir", type=str, default="runs_hw")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None,
                   help="QAT/finetune epochs (default: 20 for lsq_w4a4, "
                        "10 for funccode_w4a4)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--test-batch-size", type=int, default=2048)
    # 0 by default: get_mnist_loaders' lambda transform cannot be pickled to
    # spawn-based DataLoader workers on Windows (see HW_DESIGN_CONTRACT.md M0).
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dense-ckpt", type=str, default=None)
    p.add_argument("--clustered-ckpt", type=str, default=None)
    return p.parse_args(argv)


def run_branch_source(args):
    """Delegate to the verified pipeline (funcodekan.experiments.mnist)."""
    from funcodekan.experiments import mnist as mnist_exp
    argv = [
        "hw_prepare-branch_source",
        "--run-name", BRANCH_SOURCE_RUN, "--out-dir", args.out_dir,
        "--seed", str(args.seed),
        "--cluster-method", "branch", "--clusters", "32",
        "--branch-spline-method", "function",
        "--branch-spline-clusters", "32", "--branch-base-clusters", "16",
        "--branch-function-samples", "128", "--branch-function-domain", "grid",
        "--epochs", "10", "--finetune-epochs", "20",
        "--width", str(args.width),
        "--bits-list", "8", "6", "4", "3", "2",
        "--batch-size", str(args.batch_size),
        "--test-batch-size", str(args.test_batch_size),
        "--num-workers", str(args.num_workers),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        mnist_exp.main()
    finally:
        sys.argv = old_argv


def load_dense(ckpt_path: Path, args) -> DenseSplineKAN:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dense = DenseSplineKAN(input_dim=784, hidden_width=args.width,
                           output_dim=10, grid_size=args.grid_size,
                           spline_order=args.spline_order)
    dense.load_state_dict(ckpt["model"])
    return dense


def save_run(run_dir: Path, args, wrapper, results, rows, extra=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    payload = {"state_dict": wrapper.state_dict(), "args": vars(args),
               "results": {k: v for k, v in results.items() if k != "history"}}
    if extra:
        payload.update(extra)
    torch.save(payload, run_dir / "model.pt")
    pd.DataFrame(results["history"]).to_csv(run_dir / "qat_history.csv",
                                            index=False)
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)


def dump_scales(run_dir: Path, wrapper, per_layer_getter, n_layers):
    scales = {"act_steps": wrapper.act_steps()}
    for l in range(n_layers):
        d = per_layer_getter(l)
        scales[f"layer{l}"] = {k: v for k, v in d.items()
                               if k.endswith("_step")}
    with open(run_dir / "scales.json", "w") as f:
        json.dump(scales, f, indent=2)


def run_lsq_w4a4(args, data, device):
    epochs = args.epochs if args.epochs is not None else 20
    dense_ckpt = Path(args.dense_ckpt or
                      Path(args.out_dir) / BRANCH_SOURCE_RUN / "dense.pt")
    dense = load_dense(dense_ckpt, args)
    wrapper = QuantSplineKAN(dense)
    results = run_qat(wrapper, data, device, epochs=epochs, lr=args.lr,
                      weight_decay=args.weight_decay)
    run_dir = Path(args.out_dir) / (args.run_name or "d2_lsq_w4a4")
    dense_bits = dense_fp32_bits_from_model(dense)
    # D2 stores the SAME parameter count as dense, at 4 bits each, plus the
    # FP32 steps: 2 weight steps per layer (spline/base) + 2 act steps.
    n_scales = 2 * len(dense.weights) + 2
    w4_bits = sum(w.numel() * 4 for w in dense.weights) + n_scales * 32
    rows = [{"stage": "lsq_w4a4_qat", "design": "d2",
             "test_acc": results["test_acc"],
             "best_val_acc": results["best_val_acc"],
             "storage_kib": bits_to_kib(w4_bits),
             "compression_vs_dense": compression_ratio(dense_bits, w4_bits),
             "epochs": epochs}]
    save_run(run_dir, args, wrapper, results, rows)
    dump_scales(run_dir, wrapper, wrapper.int_weights, len(dense.layers))
    append_verification_entry({
        "milestone": "M1", "design": "d2_lsq_w4a4",
        "check": "qat_test_accuracy", "test_acc": results["test_acc"],
        "target_min": 95.2, "passed": results["test_acc"] >= 95.2,
        "run_dir": str(run_dir)})
    print(f"[hw_prepare] D2 test acc {results['test_acc']:.2f}% "
          f"(target >= 95.2) -> {run_dir}")


def run_funccode_w4a4(args, data, device):
    epochs = args.epochs if args.epochs is not None else 10
    clustered_ckpt = Path(args.clustered_ckpt or
                          Path(args.out_dir) / BRANCH_SOURCE_RUN /
                          "clustered_finetuned.pt")
    ckpt = torch.load(clustered_ckpt, map_location="cpu", weights_only=True)
    branch = rebuild_branch_model(ckpt["clustered_export"], 784, args.width,
                                  10, args.grid_size, args.spline_order,
                                  train_codebooks=True)
    wrapper = QuantBranchKAN(branch)
    results = run_qat(wrapper, data, device, epochs=epochs, lr=args.lr,
                      weight_decay=0.0)
    run_dir = Path(args.out_dir) / (args.run_name or "d3_funccode_w4a4")
    # Storage: EXACTLY the verified analytical model (analysis/storage.py)
    branch_cpu = wrapper.branch.cpu()
    q_bits = hwq_bits_from_model(branch_cpu, codebook_bits=4, scale_bits=32)
    breakdown = add_kib_columns(compressed_storage_breakdown(branch_cpu, 4, 32))
    dense_bits = 50816 * 9 * 32  # verified dense reference size (§3)
    rows = [{"stage": "funccode_w4a4_ft", "design": "d3",
             "test_acc": results["test_acc"],
             "best_val_acc": results["best_val_acc"],
             "storage_kib": bits_to_kib(q_bits),
             "compression_vs_dense": compression_ratio(dense_bits, q_bits),
             "epochs": epochs, **breakdown}]
    save_run(run_dir, args, wrapper, results, rows,
             extra={"clustered_export": branch_cpu.export_clustered_state()})
    dump_scales(run_dir, wrapper, wrapper.int_codebooks, len(branch.layers))
    append_verification_entry({
        "milestone": "M2", "design": "d3_funccode_w4a4",
        "check": "act_quant_finetune_test_accuracy",
        "test_acc": results["test_acc"], "target_min": 95.2,
        "passed": results["test_acc"] >= 95.2,
        "storage_kib": bits_to_kib(q_bits), "run_dir": str(run_dir)})
    print(f"[hw_prepare] D3 test acc {results['test_acc']:.2f}% "
          f"(target >= 95.2), storage {bits_to_kib(q_bits):.1f} KiB "
          f"-> {run_dir}")


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    if args.model == "branch_source":
        run_branch_source(args)
        return
    device = get_device(args.device)
    data = get_mnist_loaders(batch_size=args.batch_size,
                             test_batch_size=args.test_batch_size,
                             seed=args.seed, num_workers=args.num_workers)
    if args.model == "lsq_w4a4":
        run_lsq_w4a4(args, data, device)
    else:
        run_funccode_w4a4(args, data, device)


if __name__ == "__main__":
    main()
