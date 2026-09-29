import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV from summarize_multiseed_all_kan.py")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default="clustered_hwq_w4")
    p.add_argument("--include-dense", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    if args.include_dense:
        keep = df[(df["stage"] == args.stage) | (df["stage"] == "dense_fp32") | (df["stage"] == "dense_uniform_ptq_w4")].copy()
    else:
        keep = df[df["stage"] == args.stage].copy()

    if len(keep) == 0:
        raise RuntimeError(f"No rows found for stage={args.stage}")

    # Best compressed method by accuracy per variant.
    best = []
    for variant, sub in keep.groupby("variant"):
        # Exclude dense when choosing compressed method unless only dense exists.
        sub_comp = sub[~sub["method"].astype(str).eq("dense")]
        if len(sub_comp) == 0:
            sub_comp = sub
        row = sub_comp.sort_values(
            ["test_acc_mean", "compression_vs_dense_mean"],
            ascending=[False, False],
        ).iloc[0]
        best.append(row)

    out = pd.DataFrame(best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    show_cols = [
        "variant", "method", "clusters", "base_clusters", "stage",
        "num_seeds", "test_acc_mean", "test_acc_std",
        "storage_kib_mean", "compression_vs_dense_mean",
    ]
    show_cols = [c for c in show_cols if c in out.columns]
    print(out[show_cols].to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
