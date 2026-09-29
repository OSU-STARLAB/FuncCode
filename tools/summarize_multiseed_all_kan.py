import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np


def infer_seed_from_path(path: Path):
    text = str(path)
    m = re.search(r"seed(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def collect_summaries(root: Path):
    rows = []
    for summary_path in sorted(root.rglob("combined_summary.csv")):
        df = pd.read_csv(summary_path)
        seed = infer_seed_from_path(summary_path)
        run_name = summary_path.parent.name
        df["seed"] = seed
        df["run_name"] = run_name
        df["summary_path"] = str(summary_path)
        rows.append(df)

    # Also support per-variant summary.csv if combined_summary.csv is missing.
    if not rows:
        for summary_path in sorted(root.rglob("summary.csv")):
            if summary_path.name != "summary.csv":
                continue
            df = pd.read_csv(summary_path)
            seed = infer_seed_from_path(summary_path)
            run_name = summary_path.parents[1].name if len(summary_path.parents) > 1 else summary_path.parent.name
            df["seed"] = seed
            df["run_name"] = run_name
            df["summary_path"] = str(summary_path)
            rows.append(df)

    if not rows:
        raise RuntimeError(f"No combined_summary.csv or summary.csv files found under {root}")

    return pd.concat(rows, ignore_index=True)


def summarize(df, stage_filter):
    if stage_filter:
        df = df[df["stage"].astype(str).isin(stage_filter)].copy()

    group_cols = [
        "variant",
        "method",
        "clusters",
        "base_clusters",
        "stage",
        "codebook_bits",
    ]

    # Keep only columns that exist.
    group_cols = [c for c in group_cols if c in df.columns]

    metrics = [
        "test_acc",
        "storage_kib",
        "compression_vs_dense",
        "storage_total_bits",
        "storage_total_kib",
        "shared_codebook_bits",
        "shared_index_bits",
        "spline_codebook_bits",
        "spline_index_bits",
        "base_codebook_bits",
        "base_index_bits",
        "scale_bits",
    ]
    metrics = [m for m in metrics if m in df.columns]

    agg_dict = {}
    for m in metrics:
        agg_dict[m] = ["mean", "std", "min", "max"]

    out = df.groupby(group_cols, dropna=False).agg(agg_dict)
    out.columns = ["_".join(col).strip() for col in out.columns.values]
    out = out.reset_index()

    # Add number of seeds successfully found.
    ndf = df.groupby(group_cols, dropna=False)["seed"].nunique().reset_index(name="num_seeds")
    out = out.merge(ndf, on=group_cols, how="left")

    # Add seed list.
    seed_df = df.groupby(group_cols, dropna=False)["seed"].apply(
        lambda x: ",".join(str(int(v)) for v in sorted(set(x.dropna())))
    ).reset_index(name="seeds")
    out = out.merge(seed_df, on=group_cols, how="left")

    # Sort for readability.
    sort_cols = [c for c in ["variant", "stage", "test_acc_mean"] if c in out.columns]
    if sort_cols:
        ascending = [True, True, False][:len(sort_cols)]
        out = out.sort_values(sort_cols, ascending=ascending)

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root containing seed runs, e.g. runs_multiseed")
    p.add_argument("--out", required=True, help="Output CSV")
    p.add_argument(
        "--stages",
        nargs="*",
        default=["dense_fp32", "clustered_hwq_w4", "dense_uniform_ptq_w4"],
        help="Stages to include. Use empty list by passing --stages all to include all.",
    )
    args = p.parse_args()

    root = Path(args.root)
    df = collect_summaries(root)

    if args.stages == ["all"]:
        stages = None
    else:
        stages = args.stages

    out = summarize(df, stages)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    show_cols = [
        "variant", "method", "clusters", "base_clusters", "stage", "codebook_bits",
        "num_seeds", "seeds",
        "test_acc_mean", "test_acc_std",
        "storage_kib_mean", "compression_vs_dense_mean",
    ]
    show_cols = [c for c in show_cols if c in out.columns]

    print(out[show_cols].to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
