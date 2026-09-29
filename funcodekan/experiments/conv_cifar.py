"""FuncCode on convolutional KAGN networks, CIFAR-10 / CIFAR-100.

This replaces the flattened-image protocol in ``all_kan_cifar.py`` for the CIFAR
rows of the main table. The dense backbones are two convolutional KAGN
architectures (``SimpleConvKAGN`` 4-layer, ``EightSimpleConvKAGN`` 8-layer); the
compression machinery is unchanged FuncCode, applied at the edge level.

Stages, in order:
  1. train the dense backbone (or reuse a cached checkpoint)
  2. for each (method, K) : cluster -> fine-tune codebooks -> quantise -> test
  3. for each baseline    : apply -> equal-budget fine-tune -> test
  4. write one tidy CSV row per (method, operating point)

Every arm shares the dense checkpoint and the fine-tune budget, so the resulting
table is an accuracy-versus-bits-per-edge comparison at equal cost.

Example
-------
python -m funcodekan.experiments.conv_cifar \\
    --dataset cifar10 --preset kagn_simple_cifar10 --seed 42 \\
    --epochs 200 --finetune-epochs 30 \\
    --methods function branch --clusters-list 8 16 32 64 256 \\
    --baselines uniform lsq pq prune
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from ..compression import conv_baselines as base
from ..compression.conv_compression import (bits_to_kib, compress_model,
                                            dense_storage_bits, index_bits_for,
                                            quantize_codebooks_, storage_breakdown)
from ..data.cifar_conv import get_conv_cifar_loaders
from ..models.conv_kagn import (CONV_KAGN_PRESETS, build_conv_kagn,
                                compressible_layers, model_edge_bits, solve_iso_width)
from ..utils.conv_training import evaluate, fit, set_seed, setup_backend


# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--preset", default="kagn_simple_cifar10", choices=sorted(CONV_KAGN_PRESETS))
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-dir", default="runs_conv_cifar")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")

    # dense training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--subset-train", type=int, default=0)
    p.add_argument("--cutout", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-amp", action="store_true")

    # compression
    p.add_argument("--methods", nargs="*", default=["function", "branch"],
                   choices=["function", "branch", "coefficient"])
    p.add_argument("--clusters-list", nargs="+", type=int, default=[16, 32, 64, 256])
    p.add_argument("--base-clusters-list", nargs="*", type=int, default=[],
                   help="K_b for branch; defaults to K/2 for each K")
    p.add_argument("--codebook-bits", nargs="+", type=int, default=[4])
    p.add_argument("--metric", default="whiten", choices=["whiten", "signature", "coefficient"])
    p.add_argument("--signature-normalize", action="store_true",
                   help="legacy z-scored signatures (ablation arm)")
    p.add_argument("--function-samples", type=int, default=128)
    p.add_argument("--fit-samples", type=int, default=300000)
    p.add_argument("--skip-first", action="store_true")
    p.add_argument("--skip-head", action="store_true")

    # fine-tuning (shared budget across every compressed arm)
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--finetune-lr", type=float, default=2e-4)

    # baselines
    p.add_argument("--baselines", nargs="*", default=["uniform", "lsq", "pq", "prune", "iso"],
                   choices=["uniform", "lsq", "pq", "prune", "iso"])
    p.add_argument("--uniform-bits", nargs="+", type=int, default=[8, 4, 3, 2])
    p.add_argument("--lsq-bits", nargs="+", type=int, default=[4, 2])
    p.add_argument("--prune-sparsity", nargs="+", type=float, default=[0.9, 0.95])
    p.add_argument("--pq-subvectors", nargs="+", type=int, default=[2])

    # plumbing
    p.add_argument("--dense-ckpt", default=None, help="reuse a trained dense checkpoint")
    p.add_argument("--require-ckpt", action="store_true",
                   help="fail instead of training if --dense-ckpt is missing; use this "
                        "in queued compression jobs so a missing dependency is loud")
    p.add_argument("--save-dense", action="store_true", default=True)
    p.add_argument("--stages", nargs="*", default=["dense", "funccode", "baselines"])
    p.add_argument("--smoke", action="store_true", help="tiny budgets, for wiring checks")
    return p.parse_args(argv)


# --------------------------------------------------------------------------

def _emit(rows: List[Dict], run_dir: Path, **kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    keys = ["method", "config", "bits_per_edge", "test_acc", "storage_kib", "compression"]
    print("  ROW  " + "  ".join(f"{k}={kw.get(k)}" for k in keys if k in kw), flush=True)


def _bits_per_edge(storage_bits: int, n_edges: int) -> float:
    return storage_bits / max(n_edges, 1)


def _finetune(model, loaders, device, args, tag):
    return fit(model, loaders, device,
               epochs=args.finetune_epochs, lr=args.finetune_lr,
               weight_decay=0.0, optimizer=args.optimizer,
               warmup_epochs=min(2, args.finetune_epochs),
               label_smoothing=args.label_smoothing, mixup_alpha=0.0,
               amp=not args.no_amp, ema_decay=args.ema_decay, tag=tag,
               log_every=5)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args.epochs, args.finetune_epochs = 2, 1
        args.clusters_list = [16]
        args.uniform_bits, args.lsq_bits = [4], [4]
        args.prune_sparsity, args.pq_subvectors = [0.9], [2]
        args.num_workers = 2

    run_name = args.run_name or f"{args.dataset}_{args.preset}_s{args.seed}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    set_seed(args.seed)
    setup_backend()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    loaders = get_conv_cifar_loaders(
        dataset=args.dataset, data_root=args.data_root, batch_size=args.batch_size,
        test_batch_size=args.test_batch_size, seed=args.seed, val_size=args.val_size,
        subset_train=args.subset_train, num_workers=args.num_workers,
        augment=not args.no_augment, cutout=args.cutout)

    print("=" * 78)
    print(f"FuncCode conv-KAGN | {args.dataset} | {args.preset} | seed {args.seed}")
    print(f"run dir: {run_dir}  device: {device}")
    print("=" * 78, flush=True)

    rows: List[Dict] = []
    common = dict(dataset=args.dataset, preset=args.preset, seed=args.seed)

    # ---------------- stage 1: dense backbone -----------------------------
    dense = build_conv_kagn(args.preset, num_classes=loaders.num_classes)
    n_edges = sum(l.wprov.n_edges for _, l in compressible_layers(dense))
    dense_bits = dense_storage_bits(dense)
    ckpt_path = Path(args.dense_ckpt) if args.dense_ckpt else run_dir / "dense.pt"

    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        dense.load_state_dict(blob["state_dict"])
        dense.to(device)
        dense_acc = blob.get("test_acc") or evaluate(dense, loaders.test_loader, device)
        print(f"[dense] loaded {ckpt_path}  test_acc={dense_acc:.2f}", flush=True)
    else:
        if args.require_ckpt:
            raise SystemExit(f"[fatal] --require-ckpt set but no dense checkpoint at {ckpt_path}. "
                             f"The dense stage for this config has not completed.")
        print(f"[dense] training {n_edges:,} edges, {dense_bits/8/1024/1024:.2f} MiB FP32", flush=True)
        res = fit(dense, loaders, device, epochs=args.epochs, lr=args.lr,
                  weight_decay=args.weight_decay, optimizer=args.optimizer,
                  warmup_epochs=args.warmup_epochs, label_smoothing=args.label_smoothing,
                  mixup_alpha=args.mixup, amp=not args.no_amp,
                  ema_decay=args.ema_decay, tag=f"dense/{args.dataset}")
        dense_acc = res.test_acc
        if args.save_dense:
            torch.save({"state_dict": dense.state_dict(), "test_acc": dense_acc,
                        "args": vars(args), "history": res.history}, ckpt_path)
        pd.DataFrame(res.history).to_csv(run_dir / "dense_history.csv", index=False)

    _emit(rows, run_dir, **common, method="dense", config="fp32", family="dense",
          bits_per_edge=_bits_per_edge(dense_bits, n_edges), test_acc=dense_acc,
          storage_kib=bits_to_kib(dense_bits), compression=1.0, n_edges=n_edges)

    dense_state = {k: v.detach().cpu().clone() for k, v in dense.state_dict().items()}

    def fresh():
        m = build_conv_kagn(args.preset, num_classes=loaders.num_classes)
        m.load_state_dict(dense_state)
        return m

    # ---------------- stage 2: FuncCode -----------------------------------
    if "funccode" in args.stages:
        for method in args.methods:
            for i, K in enumerate(args.clusters_list):
                Kb = (args.base_clusters_list[i] if i < len(args.base_clusters_list)
                      else max(2, K // 2))
                tag = f"{method}-K{K}" + (f"-B{Kb}" if method == "branch" else "")
                print(f"\n### FuncCode {tag}", flush=True)
                t0 = time.time()

                m = fresh()
                m, plans = compress_model(
                    m, method=method, clusters=K, base_clusters=Kb, seed=args.seed,
                    samples=args.function_samples, metric=args.metric,
                    normalize=args.signature_normalize, train_codebooks=True,
                    device=device, skip_first=args.skip_first, skip_head=args.skip_head,
                    fit_samples=args.fit_samples)
                cluster_s = time.time() - t0

                res = _finetune(m, loaders, device, args, tag)

                for cb in args.codebook_bits:
                    mq = copy.deepcopy(m)
                    quantize_codebooks_(mq, cb)
                    acc = evaluate(mq.to(device), loaders.test_loader, device, amp=not args.no_amp)
                    sb = storage_breakdown(mq, codebook_bits=cb)
                    _emit(rows, run_dir, **common, method=f"funccode_{method}",
                          config=f"K{K}" + (f"_B{Kb}" if method == "branch" else "") + f"_w{cb}",
                          family="funccode", clusters=K, base_clusters=Kb if method == "branch" else 0,
                          codebook_bits=cb,
                          bits_per_edge=_bits_per_edge(sb["storage_total_bits"], n_edges),
                          test_acc=acc, storage_kib=sb["storage_total_kib"],
                          compression=dense_bits / max(sb["storage_total_bits"], 1),
                          ft_val=res.best_val, cluster_seconds=round(cluster_s, 1),
                          n_edges=n_edges,
                          fn_rel_err=float(sum(p.stats["fn_rel"] for p in plans) / max(len(plans), 1)),
                          **{k: v for k, v in sb.items() if k.endswith("_bits_total")})
                del m
                torch.cuda.empty_cache()

    # ---------------- stage 3: baselines ----------------------------------
    if "baselines" in args.stages:
        if "uniform" in args.baselines:
            for b in args.uniform_bits:
                print(f"\n### baseline uniform W{b} (PTQ)", flush=True)
                m = fresh().to(device)
                base.apply_uniform_ptq_(m, b)
                acc = evaluate(m, loaders.test_loader, device, amp=not args.no_amp)
                sb = base.baseline_storage_bits(m, "uniform", bits=b)
                _emit(rows, run_dir, **common, method="uniform_ptq", config=f"w{b}",
                      family="baseline", codebook_bits=b,
                      bits_per_edge=_bits_per_edge(sb["storage_total_bits"], n_edges),
                      test_acc=acc, storage_kib=sb["storage_total_kib"],
                      compression=dense_bits / sb["storage_total_bits"], n_edges=n_edges)

        if "lsq" in args.baselines:
            for b in args.lsq_bits:
                print(f"\n### baseline LSQ W{b} (QAT, equal budget)", flush=True)
                m = fresh()
                base.attach_lsq_(m, b, args.skip_first, args.skip_head)
                res = _finetune(m, loaders, device, args, f"lsq-w{b}")
                sb = base.baseline_storage_bits(m, "lsq", bits=b)
                _emit(rows, run_dir, **common, method="lsq_qat", config=f"w{b}",
                      family="baseline", codebook_bits=b,
                      bits_per_edge=_bits_per_edge(sb["storage_total_bits"], n_edges),
                      test_acc=res.test_acc, storage_kib=sb["storage_total_kib"],
                      compression=dense_bits / sb["storage_total_bits"],
                      ft_val=res.best_val, n_edges=n_edges)

        if "pq" in args.baselines:
            for mvec in args.pq_subvectors:
                for K in args.clusters_list:
                    print(f"\n### baseline PQ m={mvec} K={K}", flush=True)
                    m = fresh()
                    base.apply_product_quantization_(m, mvec, K, seed=args.seed, device=device)
                    res = _finetune(m, loaders, device, args, f"pq-m{mvec}-K{K}")
                    sb = base.baseline_storage_bits(m, "pq", subvectors=mvec, clusters=K)
                    _emit(rows, run_dir, **common, method="product_quant",
                          config=f"m{mvec}_K{K}", family="baseline", clusters=K,
                          bits_per_edge=_bits_per_edge(sb["storage_total_bits"], n_edges),
                          test_acc=res.test_acc, storage_kib=sb["storage_total_kib"],
                          compression=dense_bits / sb["storage_total_bits"], n_edges=n_edges)

        if "prune" in args.baselines:
            for s in args.prune_sparsity:
                print(f"\n### baseline prune {s:.0%} + W4", flush=True)
                m = fresh()
                m, st = base.apply_magnitude_prune_(m, s, bits=4)
                res = _finetune(m, loaders, device, args, f"prune-{s}")
                sb = base.baseline_storage_bits(m, "prune", bits=4, sparsity=s)
                _emit(rows, run_dir, **common, method="prune_w4", config=f"s{s}",
                      family="baseline",
                      bits_per_edge=_bits_per_edge(sb["storage_total_bits"], n_edges),
                      test_acc=res.test_acc, storage_kib=sb["storage_total_kib"],
                      compression=dense_bits / sb["storage_total_bits"], n_edges=n_edges)

        if "iso" in args.baselines:
            # a narrower dense net at the storage of each FuncCode point
            targets = sorted({index_bits_for(K) for K in args.clusters_list})
            for ib in targets:
                target_bits = int(ib * n_edges)
                scale = solve_iso_width(args.preset, target_bits, loaders.num_classes)
                print(f"\n### baseline iso-storage dense (width x{scale:.3f}, {ib} bits/edge)", flush=True)
                m = build_conv_kagn(args.preset, num_classes=loaders.num_classes, width_scale=scale)
                res = fit(m, loaders, device, epochs=args.epochs, lr=args.lr,
                          weight_decay=args.weight_decay, optimizer=args.optimizer,
                          warmup_epochs=args.warmup_epochs,
                          label_smoothing=args.label_smoothing, mixup_alpha=args.mixup,
                          amp=not args.no_amp, ema_decay=args.ema_decay,
                          tag=f"iso-{ib}bpe", log_every=20)
                sb = base.baseline_storage_bits(m, "dense")
                iso_edges = sum(l.wprov.n_edges for _, l in compressible_layers(m))
                _emit(rows, run_dir, **common, method="iso_dense", config=f"{ib}bpe",
                      family="baseline", width_scale=round(scale, 4),
                      bits_per_edge=_bits_per_edge(sb["storage_total_bits"], n_edges),
                      test_acc=res.test_acc, storage_kib=sb["storage_total_kib"],
                      compression=dense_bits / sb["storage_total_bits"],
                      n_edges=iso_edges)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "summary.csv", index=False)
    print("\n" + "=" * 78)
    print(df[["method", "config", "bits_per_edge", "test_acc", "storage_kib", "compression"]]
          .to_string(index=False))
    print(f"\nwrote {run_dir/'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
