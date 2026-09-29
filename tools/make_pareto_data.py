"""Extract accuracy-vs-storage Pareto frontiers from experiment summaries.

Scans one or more run roots for combined_summary.csv / summary.csv files,
pools all rows (any method / K / bit width / stage), and emits per-group
Pareto-optimal points (maximum accuracy for a given storage budget).

Usage:
  python tools/make_pareto_data.py \
      --roots runs_paper/mnist runs_paper/fashion_mnist \
      --out runs_paper/pareto_mnist_family.csv [--plot out.png]

Grouping default is (dataset?, variant); pass --group-cols to change.
"""

import argparse
from pathlib import Path

import pandas as pd


def collect(roots):
    frames = []
    for root in roots:
        for name in ("combined_summary.csv", "summary.csv"):
            for f in Path(root).rglob(name):
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue
                if {"test_acc"}.issubset(df.columns) and (
                    "storage_kib" in df.columns or "storage_total_kib" in df.columns
                ):
                    df = df.copy()
                    if "storage_kib" not in df.columns:
                        df["storage_kib"] = df["storage_total_kib"]
                    df["source"] = str(f)
                    frames.append(df)
    if not frames:
        raise SystemExit("No summary CSVs with test_acc + storage found.")
    return pd.concat(frames, ignore_index=True)


def pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Rows not dominated by any other row (higher acc AND lower storage)."""
    d = df.sort_values(["storage_kib", "test_acc"], ascending=[True, False])
    keep, best = [], -float("inf")
    for _, r in d.iterrows():
        if r["test_acc"] > best:
            keep.append(r)
            best = r["test_acc"]
    return pd.DataFrame(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-cols", nargs="*", default=None,
                    help="default: whichever of [dataset, variant] exist")
    ap.add_argument("--stage-filter", default=None,
                    help="e.g. clustered_hwq_w4 to restrict to one stage")
    ap.add_argument("--plot", default=None, help="optional PNG path")
    args = ap.parse_args()

    df = collect(args.roots)
    if args.stage_filter and "stage" in df.columns:
        keep_dense = df["stage"] == "dense_fp32"
        df = df[(df["stage"] == args.stage_filter) | keep_dense]

    groups = args.group_cols
    if groups is None:
        groups = [c for c in ("dataset", "variant") if c in df.columns]

    out_frames = []
    if groups:
        for key, g in df.groupby(groups):
            p = pareto(g)
            out_frames.append(p)
    else:
        out_frames.append(pareto(df))
    out = pd.concat(out_frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"Pareto points: {len(out)} rows -> {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        if groups:
            for key, g in out.groupby(groups):
                label = key if isinstance(key, str) else "/".join(map(str, key))
                g = g.sort_values("storage_kib")
                ax.plot(g["storage_kib"], g["test_acc"], "o-", ms=3, lw=1,
                        label=label)
            ax.legend(fontsize=6, ncol=2)
        else:
            g = out.sort_values("storage_kib")
            ax.plot(g["storage_kib"], g["test_acc"], "o-", ms=3, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Storage (KiB, log)")
        ax.set_ylabel("Test accuracy (%)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"Plot -> {args.plot}")


if __name__ == "__main__":
    main()
