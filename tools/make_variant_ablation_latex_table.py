import argparse
from pathlib import Path
import pandas as pd


def fmt(x, n=2):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{n}f}"


def pm(row):
    mean = row.get("test_acc_mean", float("nan"))
    std = row.get("test_acc_std", float("nan"))
    if pd.isna(std):
        std = 0.0
    return f"${float(mean):.2f}\\pm{float(std):.2f}$"


def name(row):
    variant = str(row["variant"])
    method = str(row["method"])
    k = int(row["clusters"]) if not pd.isna(row["clusters"]) else 0
    rf = float(row["residual_fraction"]) if "residual_fraction" in row and not pd.isna(row["residual_fraction"]) else 0.0

    if method == "dense":
        return f"{variant} dense"
    if method == "srb":
        return f"{variant} SRB $K={k},\\rho={rf:.2f}$"
    if method == "index":
        return f"{variant} index-efficient $K={k}$"
    if method == "soft_hard":
        return f"{variant} soft-to-hard $K={k}$"
    return f"{variant} {method} $K={k}$"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-tex", required=True)
    p.add_argument("--caption", default="Extended ablation results across KAN variants under W4 codebook quantization.")
    p.add_argument("--label", default="tab:variant_ablations")
    p.add_argument("--best-only", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["stage"].isin(["hwq_w4", "dense_fp32"])].copy()

    if args.best_only:
        rows = []
        for (variant, method), sub in df[df["method"] != "dense"].groupby(["variant", "method"]):
            rows.append(sub.sort_values(["test_acc_mean", "compression_vs_dense_mean"], ascending=[False, False]).iloc[0])
        # add dense rows
        dense = df[df["method"] == "dense"]
        if len(dense):
            rows.extend([r for _, r in dense.iterrows()])
        df = pd.DataFrame(rows)

    df = df.sort_values(["variant", "method", "test_acc_mean"], ascending=[True, True, False])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{" + args.caption + r"}")
    lines.append(r"\label{" + args.label + r"}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Seeds & Acc. (\%) & KiB & Comp. \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{name(row)} & {int(row.get('num_seeds', 0))} & {pm(row)} & "
            f"{fmt(row.get('storage_kib_mean', 0), 2)} & "
            f"{fmt(row.get('compression_vs_dense_mean', 0), 2)}$\\times$ \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
