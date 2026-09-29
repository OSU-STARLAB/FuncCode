import argparse
from pathlib import Path
import pandas as pd
import math


def fnum(x, nd=2):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"


def acc_pm(row):
    mean = row.get("test_acc_mean", float("nan"))
    std = row.get("test_acc_std", float("nan"))
    if pd.isna(std):
        std = 0.0
    return f"${float(mean):.2f} \\pm {float(std):.2f}$"


def method_name(row):
    variant = str(row.get("variant", ""))
    method = str(row.get("method", ""))
    stage = str(row.get("stage", ""))
    k = int(row.get("clusters", 0)) if not pd.isna(row.get("clusters", 0)) else 0
    kb = int(row.get("base_clusters", 0)) if not pd.isna(row.get("base_clusters", 0)) else 0
    bits = int(row.get("codebook_bits", 0)) if not pd.isna(row.get("codebook_bits", 0)) else 0

    if method == "dense":
        return f"{variant} dense"
    if method == "branch":
        return f"{variant} branch $K_s={k},K_b={kb}$"
    if method == "function":
        return f"{variant} function $K={k}$"
    if method == "coefficient":
        return f"{variant} coeff. $K={k}$"
    if method == "uniform_weight_ptq":
        return f"{variant} uniform W{bits}"
    return f"{variant} {method}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-tex", required=True)
    p.add_argument("--stage", default=None, help="Optional stage filter, e.g. clustered_hwq_w4")
    p.add_argument("--best-only", action="store_true", help="Use best method per variant from the provided CSV")
    p.add_argument("--caption", default="Multi-seed MNIST results across KAN variants under W4 codebook quantization.")
    p.add_argument("--label", default="tab:multiseed_mnist")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    if args.stage is not None:
        df = df[df["stage"].astype(str) == args.stage].copy()

    if args.best_only:
        rows = []
        for variant, sub in df.groupby("variant"):
            sub_comp = sub[~sub["method"].astype(str).eq("dense")]
            if len(sub_comp) == 0:
                sub_comp = sub
            rows.append(sub_comp.sort_values(["test_acc_mean", "compression_vs_dense_mean"], ascending=[False, False]).iloc[0])
        df = pd.DataFrame(rows)

    df = df.sort_values(["variant", "test_acc_mean"], ascending=[True, False])

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
        name = method_name(row)
        seeds = int(row.get("num_seeds", 0)) if not pd.isna(row.get("num_seeds", 0)) else 0
        acc = acc_pm(row)
        kib = fnum(row.get("storage_kib_mean", float("nan")), 2)
        comp = fnum(row.get("compression_vs_dense_mean", float("nan")), 2) + r"$\times$"
        lines.append(f"{name} & {seeds} & {acc} & {kib} & {comp} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
