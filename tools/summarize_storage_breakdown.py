import argparse
import pandas as pd

def kib(bits): return bits / 8.0 / 1024.0

def main():
    p = argparse.ArgumentParser(); p.add_argument("summary_csv"); args = p.parse_args()
    df = pd.read_csv(args.summary_csv)
    bit_cols = ["shared_codebook_bits","shared_index_bits","spline_codebook_bits","spline_index_bits","base_codebook_bits","base_index_bits","conditional_base_codebook_bits","residual_base_codebook_bits","residual_index_bits","residual_position_bits","scale_bits","metadata_bits"]
    for c in bit_cols:
        if c not in df.columns: df[c] = 0
    show = df.copy()
    for c in bit_cols + ["storage_total_bits"]:
        if c in show.columns: show[c.replace("_bits", "_kib")] = show[c].apply(kib)
    cols = ["stage","cluster_method","codebook_bits","clusters","test_acc","storage_total_kib","compression_vs_dense","shared_index_kib","spline_index_kib","base_index_kib","residual_position_kib","residual_index_kib","shared_codebook_kib","spline_codebook_kib","base_codebook_kib","conditional_base_codebook_kib","residual_base_codebook_kib","scale_kib"]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].to_string(index=False))
if __name__ == "__main__": main()
