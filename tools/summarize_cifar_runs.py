import argparse, re
from pathlib import Path
import pandas as pd


def infer_seed(path):
    m = re.search(r"seed(\d+)", str(path))
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default=None)
    args = p.parse_args()

    frames = []
    for f in sorted(Path(args.root).rglob("combined_summary.csv")):
        df = pd.read_csv(f)
        df["seed"] = infer_seed(f)
        df["run_name"] = f.parent.name
        df["summary_path"] = str(f)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No combined_summary.csv found under {args.root}")

    raw = pd.concat(frames, ignore_index=True)
    if args.stage:
        raw = raw[(raw["stage"] == args.stage) | (raw["stage"] == "dense_fp32") | (raw["stage"] == "dense_uniform_ptq_w4")].copy()

    group_cols = ["dataset", "variant", "method", "clusters", "base_clusters", "stage", "codebook_bits"]
    group_cols = [c for c in group_cols if c in raw.columns]
    metrics = ["test_acc", "storage_kib", "compression_vs_dense"]
    out = raw.groupby(group_cols, dropna=False).agg({m: ["mean", "std", "min", "max"] for m in metrics})
    out.columns = ["_".join(c) for c in out.columns]
    out = out.reset_index()

    ndf = raw.groupby(group_cols, dropna=False)["seed"].nunique().reset_index(name="num_seeds")
    out = out.merge(ndf, on=group_cols, how="left")
    seed_df = raw.groupby(group_cols, dropna=False)["seed"].apply(lambda x: ",".join(str(int(v)) for v in sorted(set(x.dropna())))).reset_index(name="seeds")
    out = out.merge(seed_df, on=group_cols, how="left")

    out = out.sort_values(["dataset", "variant", "stage", "test_acc_mean"], ascending=[True, True, True, False])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    show = ["dataset", "variant", "method", "clusters", "base_clusters", "stage", "num_seeds", "test_acc_mean", "test_acc_std", "storage_kib_mean", "compression_vs_dense_mean"]
    print(out[[c for c in show if c in out.columns]].to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
