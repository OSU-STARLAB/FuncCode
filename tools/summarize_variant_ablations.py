import argparse
from pathlib import Path
import re
import pandas as pd


def infer_seed(path: Path):
    m = re.search(r"seed(\d+)", str(path))
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default="hwq_w4")
    args = p.parse_args()

    root = Path(args.root)
    files = list(root.rglob("combined_variant_ablation_summary.csv"))
    if not files:
        files = list(root.rglob("variant_ablation_summary.csv"))
    if not files:
        raise RuntimeError(f"No variant ablation summaries found under {root}")

    frames = []
    for f in sorted(files):
        df = pd.read_csv(f)
        df["seed"] = infer_seed(f)
        df["run_name"] = f.parent.name
        df["summary_path"] = str(f)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    # Keep dense and requested W4 stage by default.
    if args.stage != "all":
        raw = raw[(raw["stage"] == args.stage) | (raw["stage"] == "dense_fp32")].copy()

    group_cols = [
        "variant", "method", "clusters", "residual_fraction",
        "stage", "codebook_bits",
    ]
    group_cols = [c for c in group_cols if c in raw.columns]

    metrics = [
        "test_acc", "storage_kib", "compression_vs_dense",
        "storage_total_kib", "spline_index_bits", "conditional_base_bits",
        "residual_index_bits", "residual_mask_bits",
    ]
    metrics = [m for m in metrics if m in raw.columns]

    agg = raw.groupby(group_cols, dropna=False).agg({m: ["mean", "std", "min", "max"] for m in metrics})
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    ndf = raw.groupby(group_cols, dropna=False)["seed"].nunique().reset_index(name="num_seeds")
    agg = agg.merge(ndf, on=group_cols, how="left")

    seed_df = raw.groupby(group_cols, dropna=False)["seed"].apply(
        lambda x: ",".join(str(int(v)) for v in sorted(set(x.dropna())))
    ).reset_index(name="seeds")
    agg = agg.merge(seed_df, on=group_cols, how="left")

    agg = agg.sort_values(["variant", "stage", "test_acc_mean"], ascending=[True, True, False])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)

    show_cols = [
        "variant", "method", "clusters", "residual_fraction", "stage",
        "num_seeds", "test_acc_mean", "test_acc_std",
        "storage_kib_mean", "compression_vs_dense_mean",
    ]
    show_cols = [c for c in show_cols if c in agg.columns]
    print(agg[show_cols].to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
