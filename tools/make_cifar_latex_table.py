import argparse
from pathlib import Path
import pandas as pd


def fmt(x, n=2):
    return "--" if pd.isna(x) else f"{float(x):.{n}f}"


def acc(row):
    std = row.get("test_acc_std", 0.0)
    std = 0.0 if pd.isna(std) else float(std)
    return f"${float(row['test_acc_mean']):.2f}\\pm{std:.2f}$"


def name(row):
    dataset = str(row.get("dataset", "")).upper()
    vmap = {"spline": "SplineKAN", "fast": "FastKAN", "gram": "GRAM/KAGN", "mlp": "MLP"}
    v = vmap.get(str(row["variant"]), str(row["variant"]))
    m = str(row["method"])
    k = int(row.get("clusters", 0)) if not pd.isna(row.get("clusters", 0)) else 0
    kb = int(row.get("base_clusters", 0)) if not pd.isna(row.get("base_clusters", 0)) else 0
    if m == "dense":
        return f"{dataset} {v} dense"
    if m == "function":
        return f"{dataset} {v} function $K={k}$"
    if m == "branch":
        return f"{dataset} {v} branch $K_s={k},K_b={kb}$"
    if m == "uniform_weight_ptq":
        return f"{dataset} {v} uniform W{int(row.get('codebook_bits', 0))}"
    return f"{dataset} {v} {m}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-tex", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--stage", default=None)
    p.add_argument("--best-only", action="store_true")
    p.add_argument("--caption", default="CIFAR compressed KAN results under W4 codebook quantization.")
    p.add_argument("--label", default="tab:cifar_results")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.dataset:
        df = df[df["dataset"].astype(str).eq(args.dataset)].copy()
    if args.stage:
        df = df[(df["stage"].astype(str).eq(args.stage)) | (df["stage"].astype(str).eq("dense_fp32"))].copy()

    if args.best_only:
        rows = []
        for (dataset, variant), sub in df.groupby(["dataset", "variant"]):
            dense = sub[sub["method"] == "dense"]
            if len(dense):
                rows.append(dense.iloc[0])
            comp = sub[sub["method"] != "dense"]
            if len(comp):
                rows.append(comp.sort_values(["test_acc_mean", "compression_vs_dense_mean"], ascending=[False, False]).iloc[0])
        df = pd.DataFrame(rows)

    df = df.sort_values(["dataset", "variant", "method", "test_acc_mean"], ascending=[True, True, True, False])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + args.caption + r"}",
        r"\label{" + args.label + r"}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Seeds & Acc. (\%) & KiB & Comp. \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(f"{name(row)} & {int(row.get('num_seeds', 0))} & {acc(row)} & {fmt(row.get('storage_kib_mean', 0), 2)} & {fmt(row.get('compression_vs_dense_mean', 0), 2)}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
