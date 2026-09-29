import argparse
from pathlib import Path
import pandas as pd

def fmt(x, n=2):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hw-csv", required=True, help="CSV from compare_hw_estimates.py")
    p.add_argument("--out-tex", required=True, help="Output LaTeX table path")
    p.add_argument("--caption", default="Hardware-memory estimates for compressed KAN variants.")
    p.add_argument("--label", default="tab:hw_memory_estimates")
    args = p.parse_args()

    df = pd.read_csv(args.hw_csv)

    # Keep the most useful paper-facing columns.
    rows = []
    for _, r in df.iterrows():
        rows.append([
            str(r.get("run_name", "")),
            fmt(r.get("test_acc", 0), 2),
            fmt(r.get("storage_total_kib", 0), 2),
            fmt(r.get("compression_vs_dense", 0), 2),
            str(int(round(float(r.get("bram18_count", 0))))),
            str(int(round(float(r.get("bram36_count", 0))))),
            fmt(100 * float(r.get("index_fraction", 0)), 1),
            fmt(r.get("traffic_reduction_vs_dense", 0), 2),
        ])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{" + args.caption + r"}")
    lines.append(r"\label{" + args.label + r"}")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Acc. & KiB & Comp. & BRAM18 & BRAM36 & Index \% & Traffic Red. \\")
    lines.append(r"\midrule")

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
