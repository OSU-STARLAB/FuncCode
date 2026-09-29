import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim

from ..data.mnist import get_mnist_loaders
from ..utils.training import set_seed, get_device, train_one_epoch, evaluate
from ..models.variants import DirectKANVariant
from ..compression.cross_variant import dense_bits, bits_to_kib
from ..models.ablations import (
    build_index_efficient_from_dense,
    build_srb_from_dense,
    build_soft_to_hard_from_dense,
    quantize_variant_ablation_model,
    variant_ablation_storage_breakdown,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run-name", type=str, default="variant_ablations_mnist")
    p.add_argument("--out-dir", type=str, default="runs_variant_ablations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--variants", nargs="+", default=["spline", "fast", "gram"], choices=["spline", "fast", "gram"])
    p.add_argument("--methods", nargs="+", default=["srb", "index", "soft"], choices=["srb", "index", "soft"])

    p.add_argument("--clusters-list", nargs="+", type=int, default=[16, 32])
    p.add_argument("--residual-fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    p.add_argument("--residual-clusters", type=int, default=8)
    p.add_argument("--bits-list", nargs="+", type=int, default=[4])

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--num-grids", type=int, default=8)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--activation", type=str, default="silu")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--soft-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=5e-4)
    p.add_argument("--soft-lr", type=float, default=5e-4)
    p.add_argument("--soft-tau-start", type=float, default=1.0)
    p.add_argument("--soft-tau-end", type=float, default=0.10)

    p.add_argument("--function-samples", type=int, default=128)

    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--test-batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)

    return p.parse_args()


def build_dense(args, variant, data):
    return DirectKANVariant(
        variant=variant,
        input_dim=data.input_dim,
        hidden_width=args.width,
        output_dim=data.num_classes,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        num_grids=args.num_grids,
        degree=args.degree,
        activation=args.activation,
    )


def train_dense(model, data, device, args, run_dir, variant):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, data.train_loader, opt, device, epoch, desc=f"{variant} dense")
        _, val_acc = evaluate(model, data.val_loader, device, desc=f"{variant} dense val")
        print(f"[{variant}] dense epoch {epoch}: val_acc={val_acc:.2f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, data.test_loader, device, desc=f"{variant} dense test")

    (run_dir / variant).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "variant": variant,
            "args": vars(args),
            "test_acc": test_acc,
        },
        run_dir / variant / "dense.pt",
    )
    return model.cpu(), test_acc


def append_row(rows, **kwargs):
    rows.append(kwargs)
    print(kwargs, flush=True)


def fine_tune_hard_model(model, data, device, epochs, lr, desc):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, data.train_loader, opt, device, epoch, desc=f"{desc} ft")
        _, val_acc = evaluate(model, data.val_loader, device, desc=f"{desc} val")
        print(f"[{desc}] epoch {epoch}: val_acc={val_acc:.2f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model.cpu()


def train_soft_model(model, data, device, args, desc):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=args.soft_lr, weight_decay=0.0)

    best_acc = -1.0
    best_state = None

    epochs = max(1, args.soft_epochs)

    for epoch in range(1, epochs + 1):
        if epochs == 1:
            tau = args.soft_tau_end
        else:
            t = (epoch - 1) / (epochs - 1)
            tau = args.soft_tau_start * ((args.soft_tau_end / args.soft_tau_start) ** t)
        model.set_tau(tau)

        train_one_epoch(model, data.train_loader, opt, device, epoch, desc=f"{desc} soft tau={tau:.3f}")
        _, val_acc = evaluate(model, data.val_loader, device, desc=f"{desc} soft val")
        print(f"[{desc}] soft epoch {epoch}: tau={tau:.4f}, val_acc={val_acc:.2f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model.cpu()


def evaluate_and_record(rows, model, data, device, dense_storage_bits, variant, method, clusters, residual_fraction, stage, codebook_bits):
    model = model.cpu()
    qmodel = model.to(device)
    _, acc = evaluate(qmodel, data.test_loader, device, desc=f"{variant} {method} {stage}")
    model = qmodel.cpu()

    scale_bits = 32 if codebook_bits < 32 else 0
    breakdown = variant_ablation_storage_breakdown(
        model,
        codebook_bits=codebook_bits,
        scale_bits_per_value=scale_bits,
    )
    total_bits = breakdown["storage_total_bits"]

    append_row(
        rows,
        variant=variant,
        method=method,
        clusters=clusters,
        residual_fraction=residual_fraction,
        stage=stage,
        codebook_bits=codebook_bits,
        test_acc=acc,
        storage_kib=bits_to_kib(total_bits),
        compression_vs_dense=dense_storage_bits / max(total_bits, 1),
        **breakdown,
    )
    return acc


def run_variant(args, data, device, run_dir, variant):
    rows = []

    dense_model = build_dense(args, variant, data)
    dense_model, dense_acc = train_dense(dense_model, data, device, args, run_dir, variant)
    dense_storage = dense_bits(dense_model)

    append_row(
        rows,
        variant=variant,
        method="dense",
        clusters=0,
        residual_fraction=0.0,
        stage="dense_fp32",
        codebook_bits=32,
        test_acc=dense_acc,
        storage_kib=bits_to_kib(dense_storage),
        compression_vs_dense=1.0,
        storage_total_bits=dense_storage,
        storage_total_kib=bits_to_kib(dense_storage),
        spline_codebook_bits=0,
        spline_index_bits=0,
        conditional_base_bits=0,
        residual_codebook_bits=0,
        residual_index_bits=0,
        residual_mask_bits=0,
        scale_bits=0,
    )

    for clusters in args.clusters_list:
        if "index" in args.methods:
            print(f"\n[{variant}] Building index-efficient K={clusters}")
            model = build_index_efficient_from_dense(
                dense_model=dense_model,
                clusters=clusters,
                seed=args.seed,
                function_samples=args.function_samples,
            )
            evaluate_and_record(rows, model, data, device, dense_storage, variant, "index", clusters, 0.0, "before_finetune_fp32", 32)

            model = fine_tune_hard_model(model, data, device, args.finetune_epochs, args.finetune_lr, f"{variant} index K{clusters}")
            evaluate_and_record(rows, model, data, device, dense_storage, variant, "index", clusters, 0.0, "finetuned_fp32", 32)

            for bits in args.bits_list:
                qmodel = quantize_variant_ablation_model(model, bits=bits)
                evaluate_and_record(rows, qmodel, data, device, dense_storage, variant, "index", clusters, 0.0, f"hwq_w{bits}", bits)

        if "soft" in args.methods:
            print(f"\n[{variant}] Building soft-to-hard K={clusters}")
            soft = build_soft_to_hard_from_dense(
                dense_model=dense_model,
                clusters=clusters,
                seed=args.seed,
                function_samples=args.function_samples,
                tau=args.soft_tau_start,
            )
            soft = train_soft_model(soft, data, device, args, f"{variant} soft K{clusters}")

            # Evaluate deployable hard model before hard fine-tuning.
            hard = soft.harden()
            evaluate_and_record(rows, hard, data, device, dense_storage, variant, "soft_hard", clusters, 0.0, "hardened_before_ft_fp32", 32)

            # Optional hard fine-tuning after soft assignment.
            hard = fine_tune_hard_model(hard, data, device, args.finetune_epochs, args.finetune_lr, f"{variant} soft_hard K{clusters}")
            evaluate_and_record(rows, hard, data, device, dense_storage, variant, "soft_hard", clusters, 0.0, "hardened_finetuned_fp32", 32)

            for bits in args.bits_list:
                qmodel = quantize_variant_ablation_model(hard, bits=bits)
                evaluate_and_record(rows, qmodel, data, device, dense_storage, variant, "soft_hard", clusters, 0.0, f"hwq_w{bits}", bits)

        if "srb" in args.methods:
            for rf in args.residual_fractions:
                print(f"\n[{variant}] Building SRB K={clusters}, residual_fraction={rf}")
                model = build_srb_from_dense(
                    dense_model=dense_model,
                    clusters=clusters,
                    residual_fraction=rf,
                    residual_clusters=args.residual_clusters,
                    seed=args.seed,
                    function_samples=args.function_samples,
                )
                evaluate_and_record(rows, model, data, device, dense_storage, variant, "srb", clusters, rf, "before_finetune_fp32", 32)

                model = fine_tune_hard_model(model, data, device, args.finetune_epochs, args.finetune_lr, f"{variant} srb K{clusters} r{rf}")
                evaluate_and_record(rows, model, data, device, dense_storage, variant, "srb", clusters, rf, "finetuned_fp32", 32)

                for bits in args.bits_list:
                    qmodel = quantize_variant_ablation_model(model, bits=bits)
                    evaluate_and_record(rows, qmodel, data, device, dense_storage, variant, "srb", clusters, rf, f"hwq_w{bits}", bits)

    vdf = pd.DataFrame(rows)
    out_path = run_dir / variant / "variant_ablation_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vdf.to_csv(out_path, index=False)

    return rows


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    data = get_mnist_loaders(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    all_rows = []

    print("=" * 80)
    print("Variant ablation MNIST experiment")
    print("=" * 80)
    print("Run dir:", run_dir)
    print("Device:", device)
    print("Variants:", args.variants)
    print("Methods:", args.methods)
    print("Clusters:", args.clusters_list)
    print("Residual fractions:", args.residual_fractions)

    for variant in args.variants:
        print("\n" + "#" * 80)
        print(f"Variant: {variant}")
        print("#" * 80)
        rows = run_variant(args, data, device, run_dir, variant)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(run_dir / "combined_variant_ablation_summary.csv", index=False)

    print("\nSaved:", run_dir / "combined_variant_ablation_summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
