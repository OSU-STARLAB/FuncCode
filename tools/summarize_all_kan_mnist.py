import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Run root directory, e.g., runs/all_kan_mnist_main")
    p.add_argument("--out", required=True, help="Output combined CSV")
    p.add_argument("--prefer-stage", default="clustered_hwq_w4")
    args = p.parse_args()

    root = Path(args.root)
    rows = []

    for summary in root.glob("*/summary.csv"):
        df = pd.read_csv(summary)
        df["source_summary"] = str(summary)

        preferred = df[df["stage"].astype(str) == args.prefer_stage].copy()
        dense = df[df["stage"].astype(str) == "dense_fp32"].copy()

        rows.append(dense)
        if len(preferred) > 0:
            rows.append(preferred)
        else:
            rows.append(df)

    if not rows:
        raise RuntimeError(f"No summary.csv files found under {root}")

    out_df = pd.concat(rows, ignore_index=True)
    out_df = out_df.sort_values(["variant", "stage", "test_acc"], ascending=[True, True, False])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    show_cols = [
        "variant", "method", "clusters", "base_clusters", "stage", "codebook_bits",
        "test_acc", "storage_kib", "compression_vs_dense",
    ]
    show_cols = [c for c in show_cols if c in out_df.columns]
    print(out_df[show_cols].to_string(index=False))
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
