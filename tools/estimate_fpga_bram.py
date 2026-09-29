import argparse
import math
from pathlib import Path
import pandas as pd

BRAM18_BITS = 18 * 1024
BRAM36_BITS = 36 * 1024

BIT_COLUMNS = [
    "storage_total_bits",
    "shared_codebook_bits",
    "shared_index_bits",
    "spline_codebook_bits",
    "spline_index_bits",
    "base_codebook_bits",
    "base_index_bits",
    "conditional_base_codebook_bits",
    "residual_base_codebook_bits",
    "residual_index_bits",
    "residual_position_bits",
    "scale_bits",
    "metadata_bits",
]

def bits_to_kib(bits):
    return float(bits) / 8.0 / 1024.0

def safe_get(row, col, default=0.0):
    if col not in row.index:
        return default
    v = row[col]
    if pd.isna(v):
        return default
    return float(v)

def bram_count(bits, bram_bits):
    return int(math.ceil(max(float(bits), 0.0) / bram_bits))

def infer_run_name(summary_path):
    p = Path(summary_path)
    if p.parent.name:
        return p.parent.name
    return p.stem

def summarize_row(row, summary_path):
    total_bits = safe_get(row, "storage_total_bits")
    if total_bits <= 0 and "storage_kib" in row.index:
        total_bits = safe_get(row, "storage_kib") * 1024.0 * 8.0

    shared_codebook = safe_get(row, "shared_codebook_bits")
    spline_codebook = safe_get(row, "spline_codebook_bits")
    base_codebook = safe_get(row, "base_codebook_bits")
    conditional_base_codebook = safe_get(row, "conditional_base_codebook_bits")
    residual_base_codebook = safe_get(row, "residual_base_codebook_bits")

    shared_index = safe_get(row, "shared_index_bits")
    spline_index = safe_get(row, "spline_index_bits")
    base_index = safe_get(row, "base_index_bits")
    residual_index = safe_get(row, "residual_index_bits")
    residual_position = safe_get(row, "residual_position_bits")

    scale_bits = safe_get(row, "scale_bits")
    metadata_bits = safe_get(row, "metadata_bits")

    codebook_bits = (
        shared_codebook
        + spline_codebook
        + base_codebook
        + conditional_base_codebook
        + residual_base_codebook
    )

    index_bits = shared_index + spline_index + base_index + residual_index + residual_position

    dense_bits = total_bits * safe_get(row, "compression_vs_dense", 1.0)
    if row.get("stage", "") == "dense_fp32":
        dense_bits = total_bits

    compressed_traffic_bits = total_bits
    dense_traffic_bits = dense_bits
    traffic_reduction = dense_traffic_bits / max(compressed_traffic_bits, 1.0)

    return {
        "run_name": infer_run_name(summary_path),
        "stage": row.get("stage", ""),
        "cluster_method": row.get("cluster_method", ""),
        "codebook_bits": safe_get(row, "codebook_bits"),
        "clusters": safe_get(row, "clusters"),
        "test_acc": safe_get(row, "test_acc"),
        "compression_vs_dense": safe_get(row, "compression_vs_dense"),
        "storage_total_kib": bits_to_kib(total_bits),
        "storage_total_bits": total_bits,
        "bram18_count": bram_count(total_bits, BRAM18_BITS),
        "bram36_count": bram_count(total_bits, BRAM36_BITS),

        "codebook_kib": bits_to_kib(codebook_bits),
        "index_kib": bits_to_kib(index_bits),
        "scale_kib": bits_to_kib(scale_bits),
        "metadata_kib": bits_to_kib(metadata_bits),

        "codebook_fraction": codebook_bits / max(total_bits, 1.0),
        "index_fraction": index_bits / max(total_bits, 1.0),
        "scale_fraction": scale_bits / max(total_bits, 1.0),

        "shared_codebook_kib": bits_to_kib(shared_codebook),
        "shared_index_kib": bits_to_kib(shared_index),
        "spline_codebook_kib": bits_to_kib(spline_codebook),
        "spline_index_kib": bits_to_kib(spline_index),
        "base_codebook_kib": bits_to_kib(base_codebook),
        "base_index_kib": bits_to_kib(base_index),
        "conditional_base_codebook_kib": bits_to_kib(conditional_base_codebook),
        "residual_base_codebook_kib": bits_to_kib(residual_base_codebook),
        "residual_index_kib": bits_to_kib(residual_index),
        "residual_position_kib": bits_to_kib(residual_position),

        "dense_traffic_per_inference_kib": bits_to_kib(dense_traffic_bits),
        "compressed_traffic_per_inference_kib": bits_to_kib(compressed_traffic_bits),
        "traffic_reduction_vs_dense": traffic_reduction,
    }

def choose_rows(df, stage_filter):
    if stage_filter is not None:
        return df[df["stage"].astype(str).isin(stage_filter)].copy()

    # Prefer W4 rows because that is the hardware-facing setting we have been using.
    if "stage" in df.columns:
        w4 = df[df["stage"].astype(str).str.contains("hwq_w4", regex=False)].copy()
        if len(w4) > 0:
            return w4

        ft = df[df["stage"].astype(str).str.contains("finetuned_fp32", regex=False)].copy()
        if len(ft) > 0:
            return ft

    return df.tail(1).copy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, help="Path to one experiment summary.csv")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--stage", nargs="*", default=None, help="Optional stage names to include")
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    rows = choose_rows(df, args.stage)

    out_rows = [summarize_row(row, args.summary) for _, row in rows.iterrows()]
    out_df = pd.DataFrame(out_rows)

    show_cols = [
        "run_name", "stage", "cluster_method", "codebook_bits", "test_acc",
        "storage_total_kib", "compression_vs_dense", "bram18_count", "bram36_count",
        "index_kib", "codebook_kib", "scale_kib", "index_fraction",
        "traffic_reduction_vs_dense",
    ]

    print(out_df[show_cols].to_string(index=False))

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
