#!/usr/bin/env python
"""Aggregate conv-CIFAR runs into the paper table.

Reads every ``summary.csv`` under the run root, deduplicates the dense rows that
each job re-emits, aggregates over seeds, and writes:

  conv_cifar_all.csv        every row, tidy
  conv_cifar_agg.csv        mean/std over seeds per (dataset, method, config)
  table_main_conv.tex       the replacement for the CIFAR half of Table 1
  table_iso_storage.tex     accuracy at matched bits/edge -- the headline claim
  pareto_conv.csv           accuracy vs storage, for fig_pareto

The iso-storage table is the one that carries the argument. It groups every
method by bits/edge so a reader compares like with like: at 5 bits/edge FuncCode
K=32 sits against uniform W1 (which does not exist), PQ (10 bits, so it is not
even in the row), and a dense net shrunk to the same budget.

  python tools/summarize_conv_cifar.py --root runs_conv_cifar --out-dir runs_conv_cifar/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRETTY = {
    "dense": "dense FP32",
    "funccode_function": r"\funccode{} function",
    "funccode_branch": r"\funccode{} branch",
    "funccode_coefficient": "coefficient (ablation)",
    "uniform_ptq": "uniform PTQ",
    "lsq_qat": "LSQ QAT",
    "product_quant": "product quant.",
    "prune_w4": "magnitude prune + W4",
    "iso_dense": "iso-storage dense",
}
ORDER = ["dense", "funccode_function", "funccode_branch", "funccode_coefficient",
         "uniform_ptq", "lsq_qat", "product_quant", "prune_w4", "iso_dense"]


def load(root: Path) -> pd.DataFrame:
    frames = []
    for csv in sorted(root.rglob("summary.csv")):
        if "_queue" in csv.parts:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"  ! skipping {csv}: {e}")
            continue
        if df.empty:
            continue
        df["source"] = str(csv.relative_to(root))
        frames.append(df)
    if not frames:
        raise SystemExit(f"no summary.csv found under {root}")
    df = pd.concat(frames, ignore_index=True)

    # every job re-emits the dense row from the shared checkpoint
    key = ["dataset", "preset", "seed", "method", "config"]
    df = df.sort_values("source").drop_duplicates(subset=key, keep="first")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "preset", "method", "config"], dropna=False)
    agg = g.agg(
        acc_mean=("test_acc", "mean"),
        acc_std=("test_acc", "std"),
        n_seeds=("test_acc", "size"),
        bits_per_edge=("bits_per_edge", "mean"),
        storage_kib=("storage_kib", "mean"),
        compression=("compression", "mean"),
    ).reset_index()
    agg["acc_std"] = agg["acc_std"].fillna(0.0)
    agg["rank"] = agg["method"].apply(lambda m: ORDER.index(m) if m in ORDER else 99)
    return agg.sort_values(["dataset", "rank", "bits_per_edge"])


def fmt(mean, std, n, bold=False):
    s = f"{mean:.2f}" if n < 2 else f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def table_main(agg: pd.DataFrame, datasets, out: Path):
    """Main table: one block per method, columns are the datasets."""
    configs = (agg[["method", "config"]].drop_duplicates()
               .assign(rank=lambda d: d["method"].map(lambda m: ORDER.index(m) if m in ORDER else 99))
               .sort_values(["rank", "config"]))

    lines = [
        r"\begin{table*}[!ht]", r"\centering",
        r"\caption{\textbf{Convolutional KAGN backbones on CIFAR, W4 codebooks} "
        r"(mean\,$\pm$\,std over seeds). Storage is bit-exact and includes "
        r"normalisation parameters and folded BatchNorm statistics for every "
        r"method. \emph{bits/edge} is total storage divided by the number of KAN "
        r"edges and is the axis on which the methods are comparable: scalar "
        r"quantisation pays per coefficient, \funccode{} pays per edge.}",
        r"\label{tab:conv_main}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll r" + " cc" * len(datasets) + "}", r"\toprule",
        "& & & " + " ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{d}}}}} &" for d in datasets).rstrip("&") + r" \\",
    ]
    cm = " ".join(rf"\cmidrule(lr){{{4+2*i}-{5+2*i}}}" for i in range(len(datasets)))
    lines.append(cm)
    lines.append("Method & Config & bits/edge & " +
                 " & ".join(r"Acc.\ (\%) & Comp." for _ in datasets) + r" \\")
    lines.append(r"\midrule")

    best = {d: agg[(agg.dataset == d) & (agg.method != "dense")]["acc_mean"].max() for d in datasets}
    last_method = None

    for _, cfg in configs.iterrows():
        m, c = cfg["method"], cfg["config"]
        cells, bpe = [], None
        for d in datasets:
            r = agg[(agg.dataset == d) & (agg.method == m) & (agg.config == c)]
            if r.empty:
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            bpe = r["bits_per_edge"] if bpe is None else bpe
            is_best = (m != "dense") and np.isclose(r["acc_mean"], best[d])
            cells += [fmt(r["acc_mean"], r["acc_std"], r["n_seeds"], is_best),
                      f"{r['compression']:.1f}$\\times$"]
        if all(x == "--" for x in cells):
            continue
        if last_method is not None and m != last_method:
            lines.append(r"\midrule")
        last_method = m
        # hoisted out of the f-string: a backslash in the expression part is a
        # SyntaxError before Python 3.12
        cfg_tex = str(c).replace("_", r"\_")
        lines.append(f"{PRETTY.get(m, m)} & {cfg_tex} & "
                     f"{bpe:.1f} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def table_iso(agg: pd.DataFrame, datasets, out: Path, tol: float = 0.6):
    """Accuracy grouped by storage budget -- the like-for-like comparison."""
    budgets = [4, 5, 6, 8, 10, 20]
    lines = [
        r"\begin{table}[!ht]", r"\centering",
        r"\caption{\textbf{Accuracy at matched storage.} Rows are storage "
        r"budgets in bits per KAN edge; entries are the best configuration each "
        r"method achieves within the budget. A dash means the method cannot "
        r"reach that budget at all: uniform quantisation of $C{=}5$ coefficients "
        r"per edge bottoms out at 10 bits/edge (W2), below which no scalar "
        r"scheme has bits left to spend.}",
        r"\label{tab:conv_iso}", r"\small",
        r"\begin{tabular}{r" + "l" * (len(datasets) * 2) + "}", r"\toprule",
        "bits/edge & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{d}}}" for d in datasets) + r" \\",
        "& " + " & ".join("method & acc." for _ in datasets) + r" \\", r"\midrule",
    ]
    for b in budgets:
        cells = []
        for d in datasets:
            sub = agg[(agg.dataset == d) & (agg.method != "dense") &
                      (agg.bits_per_edge <= b + tol)]
            if sub.empty:
                cells += ["--", "--"]
                continue
            r = sub.loc[sub["acc_mean"].idxmax()]
            cells += [PRETTY.get(r["method"], r["method"]), f"{r['acc_mean']:.2f}"]
        lines.append(f"{b} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs_conv_cifar")
    ap.add_argument("--out-dir", default="runs_conv_cifar/tables")
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    args = ap.parse_args()

    root, out_dir = Path(args.root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(root)
    agg = aggregate(df)
    df.to_csv(out_dir / "conv_cifar_all.csv", index=False)
    agg.to_csv(out_dir / "conv_cifar_agg.csv", index=False)

    present = [d for d in args.datasets if d in set(agg["dataset"])]
    table_main(agg, present, out_dir / "table_main_conv.tex")
    table_iso(agg, present, out_dir / "table_iso_storage.tex")

    pareto = agg[["dataset", "method", "config", "bits_per_edge", "storage_kib",
                  "acc_mean", "acc_std", "compression"]]
    pareto.to_csv(out_dir / "pareto_conv.csv", index=False)

    print(f"\n{len(df)} rows, {df['seed'].nunique()} seed(s), "
          f"{agg['method'].nunique()} methods")
    for d in present:
        sub = agg[agg.dataset == d]
        dn = sub[sub.method == "dense"]
        print(f"\n=== {d} ===")
        if not dn.empty:
            print(f"  dense FP32: {dn.iloc[0]['acc_mean']:.2f}%")
        top = sub[sub.method != "dense"].nlargest(5, "acc_mean")
        for _, r in top.iterrows():
            print(f"  {PRETTY.get(r['method'], r['method']):24s} {r['config']:14s} "
                  f"{r['bits_per_edge']:6.2f} b/edge  {r['acc_mean']:6.2f}%  "
                  f"{r['compression']:5.1f}x")


if __name__ == "__main__":
    main()
