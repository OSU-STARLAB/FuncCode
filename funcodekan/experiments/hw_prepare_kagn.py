"""Driver preparing the KAGN-Conv HW models (K-series; ADDITIVE research
code — see funcodekan/hw/kagn_conv.py and docs/HW_DESIGN_CONTRACT.md).

  --model kagnconv_source        train dense KagnConvNet on MNIST, then
                                 branch-cluster (Ks=32, Kb=16) + finetune
                                 -> runs_hw/kagnconv_hw_source/
  --model kagnconv_lsq_w4a4      K2: LSQ W4A4 QAT from the dense ckpt
  --model kagnconv_funccode_w4a4 K3: codebook finetune (indices frozen)

Gates mirror the gram family: within 1.0 pp of the family FP32 dense
(milestone gate), 0.3 pp additionally recorded (L4).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch

import funcodekan.models  # noqa: F401  (import-order guard)
from funcodekan.analysis.storage import (bits_to_kib, compression_ratio,
                                         compressed_storage_breakdown,
                                         add_kib_columns)
from funcodekan.data.bundles import get_dataset_bundle
from funcodekan.hw.kagn_conv import CompressedKagnConvNet, KagnConvNet
from funcodekan.hw.kagn_qat import QuantKagnBranchNet, QuantKagnConvNet
from funcodekan.hw.qat import run_qat
from funcodekan.hw.veriflog import append_verification_entry
from funcodekan.utils.training import (evaluate, get_device, set_seed,
                                       train_one_epoch)

SOURCE_RUN = "kagnconv_hw_source"


def parse_args(argv=None):
    p = argparse.ArgumentParser
    p.add_argument("--model", required=True,
                   choices=["kagnconv_source", "kagnconv_lsq_w4a4",
                            "kagnconv_funccode_w4a4"])
    p.add_argument("--out-dir", type=str, default="runs_hw")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--channels", type=int, nargs=2, default=[16, 32])
    p.add_argument("--ks", type=int, default=32)
    p.add_argument("--kb", type=int, default=16)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--dense-epochs", type=int, default=20)
    p.add_argument("--finetune-epochs", type=int, default=15)
    # W4 PTQ on the FP32-trained dense collapses (23%: ~25% of the tiny
    # conv weights round to zero and the small GAP-head net has no
    # redundancy to absorb it — measured, see the development notes). The conv family
    # therefore trains K2 with QAT FROM SCRATCH and clusters K3 from the
    # K2-trained (W4-native) weights.
    p.add_argument("--from-scratch", action="store_true",
                   help="K2: QAT from random init instead of the dense ckpt")
    p.add_argument("--cluster-from", choices=["source", "k2"], default="k2",
                   help="K3: cluster the K2 QAT weights (default) or the "
                        "FP32 dense source")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qat-lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args(argv)


def _data(args):
    return get_dataset_bundle("mnist", batch_size=args.batch_size,
                              test_batch_size=args.test_batch_size,
                              seed=args.seed, num_workers=args.num_workers,
                              flatten=False)


def _train_best(model, data, device, epochs, lr, wd, desc):
    opt = torch.optim.AdamW(model.parameters, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        _, tr_acc = train_one_epoch(model, data.train_loader, opt, device,
                                    epoch, desc=desc)
        sched.step
        _, val_acc = evaluate(model, data.val_loader, device, desc="val")
        print(f"[{desc}] epoch {epoch}/{epochs} train {tr_acc:.2f} "
              f"val {val_acc:.2f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(
                {k: v.detach.cpu for k, v in model.state_dict.items})
    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, data.test_loader, device, desc="test")
    return test_acc, best_val


def run_source(args, data, device):
    run_dir = Path(args.out_dir) / SOURCE_RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    rows = []
    dense = KagnConvNet(tuple(args.channels)).to(device)
    test_acc, _ = _train_best(dense, data, device, args.dense_epochs,
                              args.lr, 1e-4, "kagn dense")
    torch.save({"model": dense.state_dict, "args": vars(args)},
               run_dir / "dense.pt")
    dense_bits = dense.dense_storage_bits
    rows.append({"stage": "dense_fp32", "test_acc": test_acc,
                 "storage_kib": bits_to_kib(dense_bits),
                 "dense_storage_kib": bits_to_kib(dense_bits),
                 "compression_vs_dense": 1.0})
    print(f"[kagn] dense test acc {test_acc:.2f}", flush=True)

    comp = CompressedKagnConvNet(dense.cpu, ks=args.ks, kb=args.kb,
                                 seed=args.seed).to(device)
    _, before_acc = evaluate(comp, data.test_loader, device, desc="pre-FT")
    bd0 = add_kib_columns(compressed_storage_breakdown(comp, 32, 0))
    rows.append({"stage": "clustered_before_finetune_fp32_codebook",
                 "test_acc": before_acc,
                 "storage_kib": bd0["storage_total_kib"],
                 "compression_vs_dense": compression_ratio(
                     dense_bits, bd0["storage_total_bits"]), **bd0})
    ft_acc, _ = _train_best(comp, data, device, args.finetune_epochs,
                            5e-4, 0.0, "kagn branch FT")
    torch.save({"model": comp.cpu.state_dict, "args": vars(args)},
               run_dir / "clustered_finetuned.pt")
    bd = add_kib_columns(compressed_storage_breakdown(comp.cpu, 32, 0))
    rows.append({"stage": "clustered_finetuned_fp32_codebook",
                 "test_acc": ft_acc, "storage_kib": bd["storage_total_kib"],
                 "compression_vs_dense": compression_ratio(
                     dense_bits, bd["storage_total_bits"]), **bd})
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    append_verification_entry({
        "milestone": "K1", "design": "kagnconv_source",
        "dense_test_acc": test_acc, "branch_ft_test_acc": ft_acc,
        "run_dir": str(run_dir)})
    print(f"[kagn] branch FT test acc {ft_acc:.2f}", flush=True)


def _dense_acc(out_dir: Path):
    csv = out_dir / SOURCE_RUN / "summary.csv"
    if not csv.exists:
        return None
    df = pd.read_csv(csv)
    row = df[df["stage"] == "dense_fp32"]
    return float(row["test_acc"].iloc[0]) if len(row) else None


def _save(run_dir, args, wrapper, results, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    torch.save({"state_dict": wrapper.state_dict, "args": vars(args),
                "results": {k: v for k, v in results.items
                            if k != "history"}}, run_dir / "model.pt")
    pd.DataFrame(results["history"]).to_csv(run_dir / "qat_history.csv",
                                            index=False)
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)


def run_k2(args, data, device):
    epochs = args.epochs if args.epochs is not None else (
        30 if args.from_scratch else 15)
    dense = KagnConvNet(tuple(args.channels))
    if not args.from_scratch:
        ckpt = torch.load(Path(args.out_dir) / SOURCE_RUN / "dense.pt",
                          map_location="cpu", weights_only=True)
        dense.load_state_dict(ckpt["model"])
    wrapper = QuantKagnConvNet(dense)
    results = run_qat(wrapper, data, device, epochs=epochs, lr=args.qat_lr,
                      weight_decay=1e-4)
    run_dir = Path(args.out_dir) / (args.run_name or "k2_kagnconv_lsq_w4a4")
    dense_bits = dense.dense_storage_bits
    w4_bits = dense_bits // 8 + (3 * 2 + 4) * 32
    rows = [{"stage": "kagnconv_lsq_w4a4_qat", "design": "k2",
             "test_acc": results["test_acc"],
             "best_val_acc": results["best_val_acc"],
             "storage_kib": bits_to_kib(w4_bits),
             "compression_vs_dense": compression_ratio(dense_bits, w4_bits),
             "epochs": epochs}]
    _save(run_dir, args, wrapper, results, rows)
    dense_acc = _dense_acc(Path(args.out_dir))
    delta = None if dense_acc is None else dense_acc - results["test_acc"]
    append_verification_entry({
        "milestone": "K1", "design": "k2_kagnconv_lsq_w4a4",
        "check": "kagn_qat_test_accuracy", "test_acc": results["test_acc"],
        "dense_fp32_acc": dense_acc, "delta_pp": delta,
        "gate_within_1pp": None if delta is None else delta <= 1.0,
        "l4_within_0p3pp": None if delta is None else delta <= 0.3,
        "run_dir": str(run_dir)})
    print(f"[hw_prepare_kagn] K2 test acc {results['test_acc']:.2f}% "
          f"(dense {dense_acc}, delta {delta})")


def run_k3(args, data, device):
    epochs = args.epochs if args.epochs is not None else 25
    dense = KagnConvNet(tuple(args.channels))
    if args.cluster_from == "k2":
        # cluster the K2 QAT-trained weights (W4-native); indices frozen
        k2 = torch.load(Path(args.out_dir) / "k2_kagnconv_lsq_w4a4" /
                        "model.pt", map_location="cpu", weights_only=True)
        net_sd = {k[len("net."):]: v for k, v in k2["state_dict"].items
                  if k.startswith("net.")}
        dense.load_state_dict(net_sd)
        comp = CompressedKagnConvNet(dense, ks=args.ks, kb=args.kb,
                                     seed=args.seed)
    else:
        comp = CompressedKagnConvNet(dense, ks=args.ks, kb=args.kb,
                                     seed=args.seed)
        ckpt = torch.load(Path(args.out_dir) / SOURCE_RUN /
                          "clustered_finetuned.pt", map_location="cpu",
                          weights_only=True)
        comp.load_state_dict(ckpt["model"])
    wrapper = QuantKagnBranchNet(comp)
    results = run_qat(wrapper, data, device, epochs=epochs, lr=args.qat_lr,
                      weight_decay=0.0)
    run_dir = Path(args.out_dir) / (args.run_name or
                                    "k3_kagnconv_funccode_w4a4")
    comp_cpu = wrapper.net.cpu
    bd = add_kib_columns(compressed_storage_breakdown(comp_cpu, 4, 32))
    dense_bits = 811520  # KagnConvNet(16,32).dense_storage_bits
    rows = [{"stage": "kagnconv_funccode_w4a4_ft", "design": "k3",
             "test_acc": results["test_acc"],
             "best_val_acc": results["best_val_acc"],
             "storage_kib": bd["storage_total_kib"],
             "compression_vs_dense": compression_ratio(
                 dense_bits, bd["storage_total_bits"]),
             "epochs": epochs, **bd}]
    _save(run_dir, args, wrapper, results, rows)
    dense_acc = _dense_acc(Path(args.out_dir))
    delta = None if dense_acc is None else dense_acc - results["test_acc"]
    append_verification_entry({
        "milestone": "K1", "design": "k3_kagnconv_funccode_w4a4",
        "check": "kagn_codebook_ft_test_accuracy",
        "test_acc": results["test_acc"], "dense_fp32_acc": dense_acc,
        "delta_pp": delta,
        "gate_within_1pp": None if delta is None else delta <= 1.0,
        "l4_within_0p3pp": None if delta is None else delta <= 0.3,
        "storage_kib": bd["storage_total_kib"], "run_dir": str(run_dir)})
    print(f"[hw_prepare_kagn] K3 test acc {results['test_acc']:.2f}% "
          f"(dense {dense_acc}, delta {delta})")


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    data = _data(args)
    if args.model == "kagnconv_source":
        run_source(args, data, device)
    elif args.model == "kagnconv_lsq_w4a4":
        run_k2(args, data, device)
    else:
        run_k3(args, data, device)


if __name__ == "__main__":
    main
