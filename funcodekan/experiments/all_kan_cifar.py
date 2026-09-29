import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.training import set_seed, get_device, train_one_epoch, evaluate
from ..data.cifar import get_cifar_loaders
from ..models.variants import DirectKANVariant
from ..compression import cross_variant as comp


class DirectMLP(nn.Module):
    def __init__(self, input_dim, hidden_width, output_dim, activation="silu"):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.SiLU()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            act,
            nn.Linear(hidden_width, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def scalar_metric(value):
    if isinstance(value, (tuple, list)):
        value = value[0] if len(value) > 0 else 0.0
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def dense_bits(model):
    return comp.dense_bits(model)


def bits_to_kib(bits):
    return comp.bits_to_kib(bits)


def uniform_quantize_dense_model(model, bits):
    if hasattr(comp, "uniform_quantize_dense_model"):
        return comp.uniform_quantize_dense_model(model, bits=bits)
    if hasattr(comp, "quantize_mlp_copy"):
        return comp.quantize_mlp_copy(model, bits=bits)
    raise AttributeError("No dense MLP quantizer found: expected uniform_quantize_dense_model or quantize_mlp_copy.")


def cluster_function_shared_compat(dense_model, clusters, seed, samples):
    if hasattr(comp, "cluster_function_shared"):
        fn = comp.cluster_function_shared
        attempts = [
            lambda: fn(dense_model=dense_model, clusters=clusters, seed=seed, samples=samples),
            lambda: fn(dense_model, clusters, seed, samples),
        ]
    elif hasattr(comp, "cluster_function_space"):
        fn = comp.cluster_function_space
        attempts = [
            lambda: fn(dense_model=dense_model, clusters=clusters, seed=seed, samples=samples),
            lambda: fn(dense_model=dense_model, n_clusters=clusters, seed=seed, samples=samples),
            lambda: fn(dense_model, clusters, seed, samples),
        ]
    else:
        raise AttributeError("No function-space clustering found: expected cluster_function_shared or cluster_function_space.")

    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as err:
            last_error = err
    raise last_error


def cluster_branch_aware_compat(dense_model, spline_clusters, base_clusters, seed, samples):
    if not hasattr(comp, "cluster_branch_aware"):
        raise AttributeError("No branch-aware clustering found: expected cluster_branch_aware.")

    fn = comp.cluster_branch_aware
    attempts = [
        lambda: fn(
            dense_model=dense_model,
            spline_clusters=spline_clusters,
            base_clusters=base_clusters,
            seed=seed,
            samples=samples,
            spline_method="function",
        ),
        lambda: fn(
            dense_model=dense_model,
            spline_clusters=spline_clusters,
            base_clusters=base_clusters,
            seed=seed,
            samples=samples,
        ),
        lambda: fn(dense_model, spline_clusters, base_clusters, seed, samples),
    ]

    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as err:
            last_error = err
    raise last_error


def make_shared_codebook_model(dense_model, codebooks, ids, train_codebooks=True):
    if hasattr(comp, "SharedCodebookKAN"):
        return comp.SharedCodebookKAN(dense_model, codebooks, ids, train_codebooks=train_codebooks)
    raise AttributeError("No shared codebook model found: expected SharedCodebookKAN.")


def make_branch_codebook_model(dense_model, scb, sid, bcb, bid, train_codebooks=True):
    if hasattr(comp, "BranchAwareCodebookKAN"):
        return comp.BranchAwareCodebookKAN(dense_model, scb, sid, bcb, bid, train_codebooks=train_codebooks)
    if hasattr(comp, "BranchCodebookKAN"):
        return comp.BranchCodebookKAN(dense_model, scb, sid, bcb, bid, train_codebooks=train_codebooks)
    raise AttributeError("No branch codebook model found: expected BranchAwareCodebookKAN or BranchCodebookKAN.")


def quantize_shared_model(model, bits):
    if hasattr(comp, "quantize_shared_model"):
        return comp.quantize_shared_model(model, bits=bits)
    if hasattr(comp, "quantize_compressed_kan"):
        return comp.quantize_compressed_kan(model, bits=bits)
    raise AttributeError("No shared quantizer found: expected quantize_shared_model or quantize_compressed_kan.")


def quantize_branch_model(model, bits):
    if hasattr(comp, "quantize_branch_model"):
        return comp.quantize_branch_model(model, bits=bits)
    if hasattr(comp, "quantize_compressed_kan"):
        return comp.quantize_compressed_kan(model, bits=bits)
    raise AttributeError("No branch quantizer found: expected quantize_branch_model or quantize_compressed_kan.")


def storage_breakdown_shared(model, codebook_bits, scale_bits_per_codebook=32):
    if hasattr(comp, "storage_breakdown_shared"):
        return comp.storage_breakdown_shared(
            model,
            codebook_bits=codebook_bits,
            scale_bits_per_codebook=scale_bits_per_codebook,
        )
    if hasattr(comp, "compressed_storage_breakdown"):
        try:
            return comp.compressed_storage_breakdown(
                model,
                codebook_bits=codebook_bits,
                scale_bits_per_value=scale_bits_per_codebook,
            )
        except TypeError:
            return comp.compressed_storage_breakdown(model, codebook_bits=codebook_bits)
    raise AttributeError("No shared storage estimator found: expected storage_breakdown_shared or compressed_storage_breakdown.")


def storage_breakdown_branch(model, codebook_bits, scale_bits_per_codebook=32):
    if hasattr(comp, "storage_breakdown_branch"):
        return comp.storage_breakdown_branch(
            model,
            codebook_bits=codebook_bits,
            scale_bits_per_codebook=scale_bits_per_codebook,
        )
    if hasattr(comp, "compressed_storage_breakdown"):
        try:
            return comp.compressed_storage_breakdown(
                model,
                codebook_bits=codebook_bits,
                scale_bits_per_value=scale_bits_per_codebook,
            )
        except TypeError:
            return comp.compressed_storage_breakdown(model, codebook_bits=codebook_bits)
    raise AttributeError("No branch storage estimator found: expected storage_breakdown_branch or compressed_storage_breakdown.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--run-name", default="cifar_funcode")
    p.add_argument("--out-dir", default="runs_cifar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    p.add_argument("--variants", nargs="+", default=["spline", "fast", "gram", "mlp"],
                   choices=["spline", "fast", "gram", "mlp"])
    p.add_argument("--methods", nargs="+", default=["function", "branch"],
                   choices=["function", "branch"])

    p.add_argument("--clusters-list", nargs="+", type=int, default=[16, 32])
    p.add_argument("--base-cluster-ratio", type=float, default=0.5)
    p.add_argument("--bits-list", nargs="+", type=int, default=[4])

    p.add_argument("--width", type=int, default=128)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--num-grids", type=int, default=8)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--activation", default="silu")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--test-batch-size", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--subset-train", type=int, default=0)
    p.add_argument("--subset-test", type=int, default=0)
    p.add_argument("--augment", action="store_true")

    p.add_argument("--function-samples", type=int, default=128)
    return p.parse_args()


def build_model(args, data, variant):
    if variant == "mlp":
        return DirectMLP(data.input_dim, args.width, data.num_classes, activation=args.activation)

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


def train_dense_model(model, data, device, args, desc):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, data.train_loader, opt, device, epoch, desc=f"{desc} dense")
        _, val_acc = evaluate(model, data.val_loader, device, desc=f"{desc} val")
        print(f"[{desc}] dense epoch {epoch}: train_loss={scalar_metric(tr_loss):.4f} val_acc={val_acc:.2f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc = evaluate(model, data.test_loader, device, desc=f"{desc} dense test")
    return model.cpu(), test_acc


def finetune(model, data, device, args, desc):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=0.0)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.finetune_epochs + 1):
        tr_loss = train_one_epoch(model, data.train_loader, opt, device, epoch, desc=f"{desc} ft")
        _, val_acc = evaluate(model, data.val_loader, device, desc=f"{desc} ft val")
        print(f"[{desc}] ft epoch {epoch}: train_loss={scalar_metric(tr_loss):.4f} val_acc={val_acc:.2f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu()


def add_row(rows, **kwargs):
    rows.append(kwargs)
    print(kwargs, flush=True)


def run_variant(args, data, device, run_dir, variant):
    rows = []
    desc = f"{args.dataset}-{variant}"

    model = build_model(args, data, variant)
    model, dense_acc = train_dense_model(model, data, device, args, desc)
    dense_storage = dense_bits(model)

    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "dense_acc": dense_acc,
            "variant": variant,
            "dataset": args.dataset,
        },
        variant_dir / "dense.pt",
    )

    add_row(
        rows,
        dataset=args.dataset,
        variant=variant,
        method="dense",
        clusters=0,
        base_clusters=0,
        stage="dense_fp32",
        codebook_bits=32,
        test_acc=dense_acc,
        storage_kib=bits_to_kib(dense_storage),
        compression_vs_dense=1.0,
    )

    if variant == "mlp":
        for bits in args.bits_list:
            qmodel = uniform_quantize_dense_model(model, bits=bits)
            _, acc = evaluate(qmodel.to(device), data.test_loader, device, desc=f"{desc} W{bits}")
            storage = dense_storage * bits / 32.0
            add_row(
                rows,
                dataset=args.dataset,
                variant=variant,
                method="uniform_weight_ptq",
                clusters=0,
                base_clusters=0,
                stage=f"dense_uniform_ptq_w{bits}",
                codebook_bits=bits,
                test_acc=acc,
                storage_kib=bits_to_kib(storage),
                compression_vs_dense=dense_storage / max(storage, 1),
            )

        pd.DataFrame(rows).to_csv(variant_dir / "summary.csv", index=False)
        return rows

    for clusters in args.clusters_list:
        base_clusters = max(2, int(round(clusters * args.base_cluster_ratio)))

        if "function" in args.methods:
            print(f"\n[{desc}] function-space K={clusters}", flush=True)
            codebooks, ids = cluster_function_shared_compat(model, clusters, args.seed, args.function_samples)
            cmodel = make_shared_codebook_model(model, codebooks, ids, train_codebooks=True)
            cmodel = finetune(cmodel, data, device, args, f"{desc} func K{clusters}")

            for bits in args.bits_list:
                qmodel = quantize_shared_model(cmodel, bits=bits)
                _, acc = evaluate(qmodel.to(device), data.test_loader, device, desc=f"{desc} func K{clusters} W{bits}")
                breakdown = storage_breakdown_shared(qmodel, codebook_bits=bits, scale_bits_per_codebook=32)
                storage = breakdown["storage_total_bits"]
                add_row(
                    rows,
                    dataset=args.dataset,
                    variant=variant,
                    method="function",
                    clusters=clusters,
                    base_clusters=0,
                    stage=f"clustered_hwq_w{bits}",
                    codebook_bits=bits,
                    test_acc=acc,
                    storage_kib=bits_to_kib(storage),
                    compression_vs_dense=dense_storage / max(storage, 1),
                    **breakdown,
                )

        if "branch" in args.methods:
            print(f"\n[{desc}] branch-aware K={clusters}, B={base_clusters}", flush=True)
            scb, sid, bcb, bid = cluster_branch_aware_compat(model, clusters, base_clusters, args.seed, args.function_samples)
            bmodel = make_branch_codebook_model(model, scb, sid, bcb, bid, train_codebooks=True)
            bmodel = finetune(bmodel, data, device, args, f"{desc} branch K{clusters} B{base_clusters}")

            for bits in args.bits_list:
                qmodel = quantize_branch_model(bmodel, bits=bits)
                _, acc = evaluate(qmodel.to(device), data.test_loader, device, desc=f"{desc} branch K{clusters} B{base_clusters} W{bits}")
                breakdown = storage_breakdown_branch(qmodel, codebook_bits=bits, scale_bits_per_codebook=32)
                storage = breakdown["storage_total_bits"]
                add_row(
                    rows,
                    dataset=args.dataset,
                    variant=variant,
                    method="branch",
                    clusters=clusters,
                    base_clusters=base_clusters,
                    stage=f"clustered_hwq_w{bits}",
                    codebook_bits=bits,
                    test_acc=acc,
                    storage_kib=bits_to_kib(storage),
                    compression_vs_dense=dense_storage / max(storage, 1),
                    **breakdown,
                )

    pd.DataFrame(rows).to_csv(variant_dir / "summary.csv", index=False)
    return rows


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    data = get_cifar_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        val_size=args.val_size,
        seed=args.seed,
        num_workers=args.num_workers,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        augment=args.augment,
    )

    print("=" * 80)
    print("FuncCode-KAN CIFAR experiment")
    print("Dataset:", args.dataset)
    print("Run dir:", run_dir)
    print("Device:", device)
    print("Variants:", args.variants)
    print("Methods:", args.methods)
    print("=" * 80)

    all_rows = []
    for variant in args.variants:
        print("\n" + "#" * 80)
        print("Variant:", variant)
        print("#" * 80)
        all_rows.extend(run_variant(args, data, device, run_dir, variant))

    df = pd.DataFrame(all_rows)
    out_csv = run_dir / "combined_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
