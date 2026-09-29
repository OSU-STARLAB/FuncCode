"""Driver preparing the GRAM HW models (G-series; see HW_DESIGN_CONTRACT.md).

  --model gram_source     verified all_kan_mnist run (dense gram + branch
                          Ks=32/Kb=16 + finetune + PTQ sweep)
                          -> runs_hw/gram_hw_source/gram/{dense.pt,
                             branch_k32/clustered_finetuned.pt, summary.csv}
  --model gram_lsq_w4a4   G2: LSQ W4A4 QAT from the dense gram checkpoint
  --model gram_funccode_w4a4  G3: LSQ A4 + per-tensor W4 codebook finetune
                          of the verified BranchCodebookKAN (indices frozen)

Accuracy gates (documented, per family — no paper-pinned MNIST number
exists for gram): milestone gate = within 1.0 pp of the family FP32 dense;
L4 (near-lossless, 0.3 pp) is additionally recorded pass/fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

# NOTE: funcodekan.models must be imported before funcodekan.compression —
# the two packages have a pre-existing circular dependency that resolves
# only in that order (models/__init__ -> soft_codebook -> compression).
import funcodekan.models  # noqa: F401  (import-order guard)
from funcodekan.compression.cross_variant import compressed_storage_breakdown
from funcodekan.analysis.storage import bits_to_kib, compression_ratio
from funcodekan.data.mnist import get_mnist_loaders
from funcodekan.hw.gram_qat import (QuantGramBranchKAN, QuantGramKAN,
                                    rebuild_gram_branch)
from funcodekan.hw.qat import run_qat
from funcodekan.hw.veriflog import append_verification_entry
from funcodekan.models.variants import DirectKANVariant
from funcodekan.utils.training import get_device, set_seed

SOURCE_RUN = "gram_hw_source"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["gram_source", "gram_lsq_w4a4",
                            "gram_funccode_w4a4"])
    p.add_argument("--out-dir", type=str, default="runs_hw")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--test-batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dense-ckpt", type=str, default=None)
    p.add_argument("--clustered-ckpt", type=str, default=None)
    return p.parse_args(argv)


def run_gram_source(args):
    from funcodekan.experiments import all_kan_mnist
    argv = ["hw_prepare_gram-source",
            "--run-name", SOURCE_RUN, "--out-dir", args.out_dir,
            "--seed", str(args.seed), "--variants", "gram",
            "--methods", "branch", "--clusters-list", "32",
            "--bits-list", "8", "6", "4", "3", "2",
            "--epochs", "10", "--finetune-epochs", "20",
            "--width", str(args.width), "--degree", str(args.degree),
            "--batch-size", str(args.batch_size),
            "--test-batch-size", str(args.test_batch_size),
            "--num-workers", str(args.num_workers)]
    old = sys.argv
    try:
        sys.argv = argv
        all_kan_mnist.main()
    finally:
        sys.argv = old


def _dense_gram_acc(out_dir: Path) -> float | None:
    csv = out_dir / SOURCE_RUN / "gram" / "summary.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    row = df[df["stage"] == "dense_fp32"]
    return float(row["test_acc"].iloc[0]) if len(row) else None


def _save(run_dir: Path, args, wrapper, results, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    torch.save({"state_dict": wrapper.state_dict(), "args": vars(args),
                "results": {k: v for k, v in results.items()
                            if k != "history"}}, run_dir / "model.pt")
    pd.DataFrame(results["history"]).to_csv(run_dir / "qat_history.csv",
                                            index=False)
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    scales = {"act_steps": wrapper.act_steps()}
    getter = (wrapper.int_weights if hasattr(wrapper, "int_weights")
              else wrapper.int_codebooks)
    for l in range(len(wrapper.gram_layers())):
        d = getter(l)
        scales[f"layer{l}"] = {k: v for k, v in d.items()
                               if k.endswith("_step")}
    with open(run_dir / "scales.json", "w") as f:
        json.dump(scales, f, indent=2)


def _gates(design, results, dense_acc, run_dir, extra=None):
    delta = None if dense_acc is None else dense_acc - results["test_acc"]
    entry = {"milestone": "G1/G2-models", "design": design,
             "check": "gram_qat_test_accuracy",
             "test_acc": results["test_acc"], "dense_fp32_acc": dense_acc,
             "delta_pp": delta,
             "gate_within_1pp": None if delta is None else delta <= 1.0,
             "l4_within_0p3pp": None if delta is None else delta <= 0.3,
             "run_dir": str(run_dir)}
    if extra:
        entry.update(extra)
    append_verification_entry(entry)
    print(f"[hw_prepare_gram] {design} test acc {results['test_acc']:.2f}% "
          f"(dense {dense_acc}, delta {delta})")


def run_g2(args, data, device):
    epochs = args.epochs if args.epochs is not None else 20
    ckpt_path = Path(args.dense_ckpt or Path(args.out_dir) / SOURCE_RUN /
                     "gram" / "dense.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dense = DirectKANVariant("gram", 784, args.width, 10, degree=args.degree)
    dense.load_state_dict(ckpt["model"])
    wrapper = QuantGramKAN(dense)
    results = run_qat(wrapper, data, device, epochs=epochs, lr=args.lr,
                      weight_decay=args.weight_decay)
    run_dir = Path(args.out_dir) / (args.run_name or "g2_gram_lsq_w4a4")
    dense_bits = 254080 * 32
    w4_bits = 254080 * 4 + (2 * 2 + 2) * 32     # + weight/act FP32 steps
    rows = [{"stage": "gram_lsq_w4a4_qat", "design": "g2",
             "test_acc": results["test_acc"],
             "best_val_acc": results["best_val_acc"],
             "storage_kib": bits_to_kib(w4_bits),
             "compression_vs_dense": compression_ratio(dense_bits, w4_bits),
             "epochs": epochs}]
    _save(run_dir, args, wrapper, results, rows)
    _gates("g2_gram_lsq_w4a4", results, _dense_gram_acc(Path(args.out_dir)),
           run_dir)


def run_g3(args, data, device):
    epochs = args.epochs if args.epochs is not None else 25
    ckpt_path = Path(args.clustered_ckpt or Path(args.out_dir) / SOURCE_RUN /
                     "gram" / "branch_k32" / "clustered_finetuned.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    branch = rebuild_gram_branch(ckpt["model"], 784, args.width, 10,
                                 args.degree)
    wrapper = QuantGramBranchKAN(branch)
    results = run_qat(wrapper, data, device, epochs=epochs, lr=args.lr,
                      weight_decay=0.0)
    run_dir = Path(args.out_dir) / (args.run_name or "g3_gram_funccode_w4a4")
    branch_cpu = wrapper.branch.cpu()
    breakdown = compressed_storage_breakdown(branch_cpu, codebook_bits=4,
                                             scale_bits_per_value=32)
    q_bits = breakdown["storage_total_bits"]
    dense_bits = 254080 * 32
    rows = [{"stage": "gram_funccode_w4a4_ft", "design": "g3",
             "test_acc": results["test_acc"],
             "best_val_acc": results["best_val_acc"],
             "storage_kib": bits_to_kib(q_bits),
             "compression_vs_dense": compression_ratio(dense_bits, q_bits),
             "epochs": epochs, **breakdown}]
    _save(run_dir, args, wrapper, results, rows)
    _gates("g3_gram_funccode_w4a4", results,
           _dense_gram_acc(Path(args.out_dir)), run_dir,
           extra={"storage_kib": bits_to_kib(q_bits)})


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    if args.model == "gram_source":
        run_gram_source(args)
        return
    device = get_device(args.device)
    data = get_mnist_loaders(batch_size=args.batch_size,
                             test_batch_size=args.test_batch_size,
                             seed=args.seed, num_workers=args.num_workers)
    if args.model == "gram_lsq_w4a4":
        run_g2(args, data, device)
    else:
        run_g3(args, data, device)


if __name__ == "__main__":
    main()
