"""Section-4 analysis: why function space?

Trains a dense KAN variant briefly on a dataset (or loads a state_dict),
then per layer compares FUNCTION-space signatures against raw COEFFICIENT
vectors of the same edges:

  1. singular-value spectrum + effective rank (entropy of normalized
     singular values) in both spaces,
  2. k-means inertia vs K in both spaces (normalized by K=1 inertia),
  3. signature reconstruction error of the K-center codebook.

Writes tidy CSVs ready for the paper's F2 figure.

Usage:
  python tools/analyze_edge_redundancy.py \
      --dataset mnist --variant spline --epochs 3 \
      --out-dir runs_paper/redundancy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from funcodekan.data.bundles import get_dataset_bundle
from funcodekan.models.variants import DirectKANVariant
from funcodekan.compression.cross_variant import make_grid_domain, kmeans_fit
from funcodekan.utils.training import set_seed, get_device, train_one_epoch, evaluate


def effective_rank(s: np.ndarray) -> float:
    """Roy & Vetterli effective rank: exp(entropy of s / sum(s))."""
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def spectrum_rows(mat: np.ndarray, space: str, layer: int, dataset: str,
                  variant: str):
    mat = mat - mat.mean(axis=0, keepdims=True)
    s = np.linalg.svd(mat, compute_uv=False)
    er = effective_rank(s)
    rows = []
    for i, v in enumerate(s):
        rows.append(dict(dataset=dataset, variant=variant, layer=layer,
                         space=space, sv_index=i, singular_value=float(v),
                         effective_rank=er, n_edges=mat.shape[0]))
    return rows


def inertia_rows(mat: np.ndarray, space: str, layer: int, dataset: str,
                 variant: str, ks, seed: int):
    base = float(((mat - mat.mean(axis=0, keepdims=True)) ** 2).sum())
    rows = []
    for k in ks:
        if k >= mat.shape[0]:
            continue
        centers, labels = kmeans_fit(mat.astype(np.float32), k, seed)
        assigned = centers[labels]
        inertia = float(((mat.astype(np.float32) - assigned) ** 2).sum())
        rows.append(dict(dataset=dataset, variant=variant, layer=layer,
                         space=space, k=k,
                         relative_inertia=inertia / max(base, 1e-12)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="spline",
                    choices=["spline", "fast", "gram"])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--spline-order", type=int, default=3)
    ap.add_argument("--num-grids", type=int, default=8)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--ks", nargs="+", type=int,
                    default=[2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--state-dict", default=None,
                    help="optional pretrained dense state_dict (.pt) to skip training")
    ap.add_argument("--out-dir", default="runs_paper/redundancy")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    data = get_dataset_bundle(args.dataset, data_dir=args.data_dir,
                              batch_size=args.batch_size, seed=args.seed,
                              num_workers=args.num_workers)

    model = DirectKANVariant(
        args.variant, data.input_dim, args.width, data.num_classes,
        grid_size=args.grid_size, spline_order=args.spline_order,
        num_grids=args.num_grids, degree=args.degree,
    ).to(device)

    if args.state_dict:
        model.load_state_dict(torch.load(args.state_dict, map_location=device))
        test_acc = evaluate(model, data.test_loader, device)[1]
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for ep in range(1, args.epochs + 1):
            train_one_epoch(model, data.train_loader, opt, device, ep,
                            desc=f"{args.dataset}/{args.variant}")
        test_acc = evaluate(model, data.test_loader, device)[1]
    print(f"dense {args.variant} on {args.dataset}: test acc {test_acc:.2f}%")

    model = model.cpu().eval()
    x_dom = make_grid_domain(samples=args.samples, device="cpu")

    spec_rows, inrt_rows = [], []
    for li, (layer, w) in enumerate(zip(model.layers, model.get_edge_weights())):
        coeff = w.reshape(w.shape[0] * w.shape[1], -1).detach().numpy()
        sig = layer.edge_function_signatures(
            w, x_dom, include_base=True, normalize=True).detach().numpy()
        for mat, space in ((sig, "function"), (coeff, "coefficient")):
            spec_rows += spectrum_rows(mat, space, li, args.dataset, args.variant)
            inrt_rows += inertia_rows(mat, space, li, args.dataset,
                                      args.variant, args.ks, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.variant}_seed{args.seed}"
    pd.DataFrame(spec_rows).to_csv(out / f"spectrum_{tag}.csv", index=False)
    pd.DataFrame(inrt_rows).to_csv(out / f"inertia_{tag}.csv", index=False)

    er = (pd.DataFrame(spec_rows)
          .groupby(["layer", "space"])["effective_rank"].first().unstack())
    er["n_edges"] = (pd.DataFrame(spec_rows)
                     .groupby("layer")["n_edges"].first())
    er["test_acc"] = test_acc
    er.to_csv(out / f"effective_rank_{tag}.csv")
    print(er)
    print(f"Saved spectrum/inertia/effective-rank CSVs -> {out}/*_{tag}.csv")


if __name__ == "__main__":
    main()
