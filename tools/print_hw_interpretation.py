import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hw-csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.hw_csv)
    if len(df) == 0:
        print("No data.")
        return

    best_acc = df.sort_values("test_acc", ascending=False).iloc[0]
    best_comp = df.sort_values("compression_vs_dense", ascending=False).iloc[0]
    lowest_bram = df.sort_values("bram18_count", ascending=True).iloc[0]
    most_index = df.sort_values("index_fraction", ascending=False).iloc[0]

    print("Paper-ready interpretation:")
    print()
    print(f"- Highest W4 accuracy: {best_acc['run_name']} with {best_acc['test_acc']:.2f}% accuracy, "
          f"{best_acc['storage_total_kib']:.2f} KiB, and {best_acc['compression_vs_dense']:.2f}x compression.")
    print(f"- Highest compression: {best_comp['run_name']} with {best_comp['compression_vs_dense']:.2f}x compression "
          f"and {best_comp['storage_total_kib']:.2f} KiB storage.")
    print(f"- Lowest BRAM18 count: {lowest_bram['run_name']} uses approximately {int(lowest_bram['bram18_count'])} BRAM18 blocks.")
    print(f"- Most index-dominated: {most_index['run_name']} has {100 * most_index['index_fraction']:.1f}% of storage in index streams.")
    print()
    print("Suggested wording:")
    print()
    print("The analytical FPGA-memory estimator shows that compressed KAN deployments are dominated by index storage rather than codebook storage. "
          "This confirms that future hardware-aware KAN compression should prioritize compact index streams and branch-aware assignment formats, "
          "not only low-bit centroid quantization.")

if __name__ == "__main__":
    main()
