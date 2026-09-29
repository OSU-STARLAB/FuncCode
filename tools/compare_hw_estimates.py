import argparse
from pathlib import Path
import pandas as pd
import subprocess
import sys

# Import helper functions from estimate_fpga_bram.py without requiring a package install.
TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
from estimate_fpga_bram import choose_rows, summarize_row

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", nargs="+", required=True, help="List of summary.csv files")
    p.add_argument("--out", required=True, help="Output comparison CSV")
    p.add_argument("--stage", nargs="*", default=None, help="Optional stage names to include")
    args = p.parse_args()

    all_rows = []

    for summary in args.summaries:
        path = Path(summary)
        if not path.exists():
            print(f"[skip missing] {summary}")
            continue

        df = pd.read_csv(path)
        rows = choose_rows(df, args.stage)

        for _, row in rows.iterrows():
            all_rows.append(summarize_row(row, path))

    out_df = pd.DataFrame(all_rows)

    if len(out_df) == 0:
        raise RuntimeError("No rows found. Check summary paths.")

    # Sort by W4 accuracy descending, then storage ascending.
    if "test_acc" in out_df.columns:
        out_df = out_df.sort_values(
            by=["test_acc", "storage_total_kib"],
            ascending=[False, True],
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    show_cols = [
        "run_name",
        "stage",
        "cluster_method",
        "codebook_bits",
        "test_acc",
        "storage_total_kib",
        "compression_vs_dense",
        "bram18_count",
        "bram36_count",
        "index_kib",
        "codebook_kib",
        "scale_kib",
        "index_fraction",
        "traffic_reduction_vs_dense",
    ]
    show_cols = [c for c in show_cols if c in out_df.columns]

    print(out_df[show_cols].to_string(index=False))
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
