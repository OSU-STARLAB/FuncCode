import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from ..data.mnist import get_mnist_loaders
from ..utils.training import set_seed, get_device, train_one_epoch, evaluate
from ..models.variants import DirectKANVariant, MLPBaseline
from ..compression.cross_variant import (
    build_compressed_kan,
    quantize_compressed_kan,
    quantize_mlp_copy,
    dense_bits,
    compressed_storage_breakdown,
    bits_to_kib,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run-name", type=str, default="all_kan_mnist")
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--variants", nargs="+", default=["spline", "fast", "gram", "mlp"], choices=["spline", "fast", "gram", "mlp"])
    p.add_argument("--methods", nargs="+", default=["function", "branch"], choices=["none", "coefficient", "function", "branch"])
    p.add_argument("--clusters-list", nargs="+", type=int, default=[16, 32])
    p.add_argument("--bits-list", nargs="+", type=int, default=[8, 6, 4, 3, 2])

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--num-grids", type=int, default=8)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--activation", type=str, default="silu")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=5e-4)
    p.add_argument("--function-samples", type=int, default=128)

    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--test-batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)

    return p.parse_args()


def build_dense_variant(args, variant, data):
    if variant == "mlp":
        return MLPBaseline(
            input_dim=data.input_dim,
            hidden_width=args.width,
            output_dim=data.num_classes,
            activation=args.activation,
        )

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

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, data.train_loader, opt, device, epoch, desc=f"{variant} dense train")
        val_loss, val_acc = evaluate(model, data.val_loader, device, desc=f"{variant} dense val")
        print(f"[{variant}] dense epoch {epoch}: train_acc={tr_acc:.2f}, val_acc={val_acc:.2f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, data.test_loader, device, desc=f"{variant} dense test")

    torch.save(
        {
            "model": model.state_dict(),
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


def run_kan_compression(args, data, device, run_dir, variant, dense_model, dense_acc, dense_storage_bits, rows):
    for method in args.methods:
        if method == "none":
            continue

        for clusters in args.clusters_list:
            base_clusters = max(2, clusters // 2)
            print("\n" + "=" * 80)
            print(f"[{variant}] method={method}, clusters={clusters}, base_clusters={base_clusters}")
            print("=" * 80)

            compressed = build_compressed_kan(
                dense_model=dense_model,
                method=method,
                clusters=clusters,
                seed=args.seed,
                function_samples=args.function_samples,
                base_clusters=base_clusters,
                train_codebooks=True,
            ).to(device)

            _, before_acc = evaluate(compressed, data.test_loader, device, desc=f"{variant} {method} K{clusters} before FT")

            before_breakdown = compressed_storage_breakdown(compressed.cpu(), codebook_bits=32, scale_bits_per_value=0)
            before_bits = before_breakdown["storage_total_bits"]

            append_row(
                rows,
                variant=variant,
                method=method,
                clusters=clusters,
                base_clusters=base_clusters if method == "branch" else 0,
                stage="clustered_before_finetune_fp32_codebook",
                codebook_bits=32,
                test_acc=before_acc,
                storage_kib=bits_to_kib(before_bits),
                compression_vs_dense=dense_storage_bits / max(before_bits, 1),
                **before_breakdown,
            )

            compressed = compressed.to(device)
            opt = optim.AdamW(compressed.parameters(), lr=args.finetune_lr, weight_decay=0.0)
            best_val_acc = -1.0
            best_state = None

            for epoch in range(1, args.finetune_epochs + 1):
                tr_loss, tr_acc = train_one_epoch(compressed, data.train_loader, opt, device, epoch, desc=f"{variant} {method} FT")
                val_loss, val_acc = evaluate(compressed, data.val_loader, device, desc=f"{variant} {method} val")
                print(f"[{variant}] {method} K{clusters} FT epoch {epoch}: train_acc={tr_acc:.2f}, val_acc={val_acc:.2f}", flush=True)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in compressed.state_dict().items()}

            compressed.load_state_dict(best_state)
            _, ft_acc = evaluate(compressed, data.test_loader, device, desc=f"{variant} {method} K{clusters} FT test")

            method_dir = run_dir / variant / f"{method}_k{clusters}"
            method_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": compressed.cpu().state_dict(),
                    "variant": variant,
                    "method": method,
                    "clusters": clusters,
                    "base_clusters": base_clusters,
                    "args": vars(args),
                },
                method_dir / "clustered_finetuned.pt",
            )

            ft_breakdown = compressed_storage_breakdown(compressed.cpu(), codebook_bits=32, scale_bits_per_value=0)
            ft_bits = ft_breakdown["storage_total_bits"]

            append_row(
                rows,
                variant=variant,
                method=method,
                clusters=clusters,
                base_clusters=base_clusters if method == "branch" else 0,
                stage="clustered_finetuned_fp32_codebook",
                codebook_bits=32,
                test_acc=ft_acc,
                storage_kib=bits_to_kib(ft_bits),
                compression_vs_dense=dense_storage_bits / max(ft_bits, 1),
                **ft_breakdown,
            )

            for bits in args.bits_list:
                qmodel = quantize_compressed_kan(compressed.cpu(), bits=bits).to(device)
                _, q_acc = evaluate(qmodel, data.test_loader, device, desc=f"{variant} {method} K{clusters} W{bits}")

                q_breakdown = compressed_storage_breakdown(qmodel.cpu(), codebook_bits=bits, scale_bits_per_value=32)
                q_bits = q_breakdown["storage_total_bits"]

                append_row(
                    rows,
                    variant=variant,
                    method=method,
                    clusters=clusters,
                    base_clusters=base_clusters if method == "branch" else 0,
                    stage=f"clustered_hwq_w{bits}",
                    codebook_bits=bits,
                    test_acc=q_acc,
                    storage_kib=bits_to_kib(q_bits),
                    compression_vs_dense=dense_storage_bits / max(q_bits, 1),
                    **q_breakdown,
                )


def run_mlp_quantization(args, data, device, variant, dense_model, dense_acc, dense_storage_bits, rows):
    for bits in args.bits_list:
        qmodel = quantize_mlp_copy(dense_model.cpu(), bits=bits).to(device)
        _, q_acc = evaluate(qmodel, data.test_loader, device, desc=f"mlp W{bits}")

        q_bits = 0
        for p in qmodel.parameters():
            q_bits += p.numel() * bits

        append_row(
            rows,
            variant=variant,
            method="uniform_weight_ptq",
            clusters=0,
            base_clusters=0,
            stage=f"dense_uniform_ptq_w{bits}",
            codebook_bits=bits,
            test_acc=q_acc,
            storage_kib=bits_to_kib(q_bits),
            compression_vs_dense=dense_storage_bits / max(q_bits, 1),
            storage_total_bits=q_bits,
            storage_total_kib=bits_to_kib(q_bits),
            shared_codebook_bits=0,
            shared_index_bits=0,
            spline_codebook_bits=0,
            spline_index_bits=0,
            base_codebook_bits=0,
            base_index_bits=0,
            scale_bits=0,
        )


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

    print("=" * 80)
    print("All-KAN MNIST experiment")
    print("=" * 80)
    print("Run dir:", run_dir)
    print("Device:", device)
    print("Variants:", args.variants)
    print("Methods:", args.methods)
    print("Clusters:", args.clusters_list)

    all_rows = []

    for variant in args.variants:
        print("\n" + "#" * 80)
        print(f"Variant: {variant}")
        print("#" * 80)

        (run_dir / variant).mkdir(parents=True, exist_ok=True)

        dense_model = build_dense_variant(args, variant, data)
        dense_model, dense_acc = train_dense(dense_model, data, device, args, run_dir, variant)
        dense_storage_bits = dense_bits(dense_model)

        variant_rows = []
        append_row(
            variant_rows,
            variant=variant,
            method="dense",
            clusters=0,
            base_clusters=0,
            stage="dense_fp32",
            codebook_bits=32,
            test_acc=dense_acc,
            storage_kib=bits_to_kib(dense_storage_bits),
            compression_vs_dense=1.0,
            storage_total_bits=dense_storage_bits,
            storage_total_kib=bits_to_kib(dense_storage_bits),
            shared_codebook_bits=0,
            shared_index_bits=0,
            spline_codebook_bits=0,
            spline_index_bits=0,
            base_codebook_bits=0,
            base_index_bits=0,
            scale_bits=0,
        )

        if variant == "mlp":
            run_mlp_quantization(args, data, device, variant, dense_model, dense_acc, dense_storage_bits, variant_rows)
        else:
            run_kan_compression(args, data, device, run_dir, variant, dense_model, dense_acc, dense_storage_bits, variant_rows)

        vdf = pd.DataFrame(variant_rows)
        vdf.to_csv(run_dir / variant / "summary.csv", index=False)
        all_rows.extend(variant_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(run_dir / "combined_summary.csv", index=False)

    print("\nCombined summary saved to:", run_dir / "combined_summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
