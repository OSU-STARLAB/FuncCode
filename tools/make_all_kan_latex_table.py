import argparse
from pathlib import Path
import pandas as pd

def fmt(x, n=2):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)

def method_name(row):
    method = str(row.get("method", ""))
    variant = str(row.get("variant", ""))
    k = int(row.get("clusters", 0)) if not pd.isna(row.get("clusters", 0)) else 0
    kb = int(row.get("base_clusters", 0)) if not pd.isna(row.get("base_clusters", 0)) else 0

    if method == "dense":
        return f"{variant} dense"
    if method == "branch":
        return f"{variant} branch $K_s={k},K_b={kb}$"
    if method == "function":
        return f"{variant} function $K={k}$"
    if method == "coefficient":
        return f"{variant} coeff. $K={k}$"
    if method == "uniform_weight_ptq":
        return f"{variant} uniform W{int(row.get('codebook_bits', 0))}"
    return f"{variant} {method}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-tex", required=True)
    p.add_argument("--stage", default="clustered_hwq_w4")
    p.add_argument("--caption", default="MNIST results across KAN variants under W4 compression.")
    p.add_argument("--label", default="tab:all_kan_mnist")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    # Keep dense and requested compressed stage.
    keep = df[(df["stage"] == "dense_fp32") | (df["stage"] == args.stage)].copy()
    keep = keep.sort_values(["variant", "test_acc"], ascending=[True, False])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{" + args.caption + r"}")
    lines.append(r"\label{" + args.label + r"}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Bits & Acc. & KiB & Comp. \\")
    lines.append(r"\midrule")

    for _, row in keep.iterrows():
        name = method_name(row)
        bits = int(row.get("codebook_bits", 32))
        acc = fmt(row.get("test_acc", 0), 2)
        kib = fmt(row.get("storage_kib", 0), 2)
        comp = fmt(row.get("compression_vs_dense", 0), 2) + r"$\times$"
        lines.append(f"{name} & {bits} & {acc} & {kib} & {comp} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
