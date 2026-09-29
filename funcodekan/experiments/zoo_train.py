"""Train / evaluate external zoo models (funcodekan.models.zoo) on any
dataset in funcodekan.data.registry.

Requires the external `kans` package (see funcodekan/models/zoo.py). Supports:

- classification datasets (tabular, synthetic, image): CrossEntropy + accuracy
- `traffic_california`: forecasting regression (MSE), VIKIN / PDR-KAN protocol

Examples
--------
python -m funcodekan.experiments.zoo_train --dataset mnist --arch kan_mlp_mnist
python -m funcodekan.experiments.zoo_train --dataset wine --arch kan_wine
python -m funcodekan.experiments.zoo_train --dataset cifar10 --arch kagn_simple_cifar10
python -m funcodekan.experiments.zoo_train --dataset traffic_california \
    --arch kan_traffic_3layer --sensor 0
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F

from ..data.bundles import get_dataset_bundle, get_traffic_regression_loaders
from ..utils.training import set_seed, get_device, train_one_epoch, evaluate
from ..models import zoo

# Archs whose forward expects [B, C, H, W] images (everything convolutional).
_CONV_PREFIXES = ("kan_lenet", "kan_convnet", "kan_resnet", "kagn_", "vgg")


def build_zoo_model(dataset: str, arch: str):
    """Thin adapter around the zoo's create_model(args) dispatcher."""
    if not zoo.KANS_AVAILABLE:
        raise ImportError(
            "The model zoo requires the external 'kans' package "
            f"(original error: {zoo.KANS_IMPORT_ERROR}). Vendor your kans/ "
            "package at the repository root or install it, then re-run."
        )
    args = SimpleNamespace(
        dataloader=SimpleNamespace(dataset=dataset),
        arch=arch,
        pre_trained=False,
    )
    return zoo.create_model(args)


def is_conv_arch(arch: str) -> bool:
    return any(arch.startswith(p) or p in arch for p in _CONV_PREFIXES)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--arch", type=str, required=True)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs_zoo")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=2)
    # traffic-specific (VIKIN / PDR-KAN single-sensor protocol)
    p.add_argument("--sensor", type=int, default=0,
                   help="traffic_california: sensor index (univariate window)")
    p.add_argument("--input-len", type=int, default=72)
    p.add_argument("--horizon", type=int, default=96)
    return p.parse_args()


@torch.no_grad()
def eval_mse(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device).float()
        pred = model(x)
        total += F.mse_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return total / n


def train_regression(model, loaders, device, args, run_dir):
    tr, va, te, _, _ = loaders
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    best_va, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device).float(), y.to(device).float()
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        va_mse = eval_mse(model, va, device)
        print(f"epoch {epoch:3d}  val MSE {va_mse:.6e}")
        if va_mse < best_va:
            best_va = va_mse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    te_mse = eval_mse(model, te, device)
    print(f"TEST MSE: {te_mse:.6e}  (best val {best_va:.6e})")
    return {"val_mse": best_va, "test_mse": te_mse}


def train_classification(model, data, device, args, run_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    best_va, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, data.train_loader, opt, device, epoch,
                        desc=f"{args.arch}")
        _, va_acc = evaluate(model, data.val_loader, device, desc="val")
        print(f"epoch {epoch:3d}  val acc {va_acc:.2f}%")
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    _, te_acc = evaluate(model, data.test_loader, device, desc="test")
    print(f"TEST ACC: {te_acc:.2f}%  (best val {best_va:.2f}%)")
    return {"val_acc": best_va, "test_acc": te_acc}


def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"zoo_{args.dataset}_{args.arch}_seed{args.seed}"
    set_seed(args.seed)
    device = get_device(args.device)

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = build_zoo_model(args.dataset, args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args.arch}: {n_params:,} parameters on {device}")

    if args.dataset == "traffic_california":
        loaders = get_traffic_regression_loaders(
            data_dir=args.data_dir, sensors=[args.sensor],
            input_len=args.input_len, horizon=args.horizon,
            batch_size=args.batch_size, test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
        )
        metrics = train_regression(model, loaders, device, args, run_dir)
    else:
        data = get_dataset_bundle(
            args.dataset, data_dir=args.data_dir,
            batch_size=args.batch_size, test_batch_size=args.test_batch_size,
            seed=args.seed, num_workers=args.num_workers,
            flatten=not is_conv_arch(args.arch),
        )
        metrics = train_classification(model, data, device, args, run_dir)

    row = {"dataset": args.dataset, "arch": args.arch, "seed": args.seed,
           "params": n_params, **metrics}
    pd.DataFrame([row]).to_csv(run_dir / "summary.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    print("Saved:", run_dir / "summary.csv")


if __name__ == "__main__":
    main()
