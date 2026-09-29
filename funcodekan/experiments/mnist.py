import argparse, json
from pathlib import Path
import pandas as pd
import torch
import torch.optim as optim
from ..data.mnist import get_mnist_loaders
from ..models.spline import DenseSplineKAN
from ..utils.training import set_seed, get_device, train_one_epoch, evaluate
from ..compression.clustering import build_clustered_from_dense
from ..quantization.hwq_pipeline import quantize_clustered_model, export_hwq_state
from ..analysis.storage import dense_fp32_bits_from_model, clustered_fp32_bits_from_model, hwq_bits_from_model, bits_to_kib, compression_ratio, dense_storage_breakdown, compressed_storage_breakdown, add_kib_columns

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, default="mnist_splinekan"); p.add_argument("--out-dir", type=str, default="runs"); p.add_argument("--seed", type=int, default=42); p.add_argument("--device", type=str, default="auto")
    p.add_argument("--width", type=int, default=64); p.add_argument("--grid-size", type=int, default=5); p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--finetune-epochs", type=int, default=3); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--finetune-lr", type=float, default=5e-4)
    p.add_argument("--clusters", type=int, default=16); p.add_argument("--bits-list", type=int, nargs="+", default=[8,6,4,3,2])
    p.add_argument("--cluster-method", type=str, default="coefficient", choices=["coefficient","function","branch","branch_index","branch_residual"])
    p.add_argument("--function-samples", type=int, default=128); p.add_argument("--function-domain", type=str, default="grid", choices=["grid","activation"]); p.add_argument("--activation-sample-batches", type=int, default=8)
    p.add_argument("--include-base-in-function", action="store_true"); p.add_argument("--no-normalize-function-signatures", action="store_true")
    p.add_argument("--branch-spline-clusters", type=int, default=None); p.add_argument("--branch-base-clusters", type=int, default=8); p.add_argument("--branch-spline-method", type=str, default="function", choices=["function","coefficient"])
    p.add_argument("--branch-function-samples", type=int, default=128); p.add_argument("--branch-function-domain", type=str, default="grid", choices=["grid","activation"])
    p.add_argument("--residual-fraction", type=float, default=0.25); p.add_argument("--residual-base-clusters", type=int, default=8); p.add_argument("--residual-selection", type=str, default="error", choices=["error","magnitude"])
    p.add_argument("--batch-size", type=int, default=1024); p.add_argument("--test-batch-size", type=int, default=2048); p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()

def append_result(rows, **kwargs): rows.append(kwargs); print(kwargs, flush=True)

def main():
    args = parse_args(); set_seed(args.seed); device = get_device(args.device)
    run_dir = Path(args.out_dir) / args.run_name; run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f: json.dump(vars(args), f, indent=2)
    print("="*60); print("FuncCode-KAN MNIST experiment"); print("Run directory:", run_dir); print("Device:", device); print("Cluster method:", args.cluster_method)
    if args.cluster_method in ["branch","branch_index","branch_residual"]:
        print("Branch spline clusters:", args.branch_spline_clusters if args.branch_spline_clusters is not None else args.clusters)
        print("Branch spline method:", args.branch_spline_method); print("Branch function domain:", args.branch_function_domain)
    if args.cluster_method == "branch_residual":
        print("Residual fraction:", args.residual_fraction); print("Residual base clusters:", args.residual_base_clusters); print("Residual selection:", args.residual_selection)
    data = get_mnist_loaders(batch_size=args.batch_size, test_batch_size=args.test_batch_size, seed=args.seed, num_workers=args.num_workers)
    rows = []
    print("\n[1/5] Training dense SplineKAN from scratch")
    dense = DenseSplineKAN(input_dim=data.input_dim, hidden_width=args.width, output_dim=data.num_classes, grid_size=args.grid_size, spline_order=args.spline_order).to(device)
    opt = optim.AdamW(dense.parameters(), lr=args.lr, weight_decay=1e-4); best_val_acc = -1.0; best_state = None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(dense, data.train_loader, opt, device, epoch, desc="dense train")
        val_loss, val_acc = evaluate(dense, data.val_loader, device, desc="dense val")
        print(f"Dense epoch {epoch}: train_acc={tr_acc:.2f}, val_acc={val_acc:.2f}")
        if val_acc > best_val_acc: best_val_acc = val_acc; best_state = {k:v.detach().cpu().clone() for k,v in dense.state_dict().items()}
    dense.load_state_dict(best_state); _, dense_test_acc = evaluate(dense, data.test_loader, device, desc="dense test")
    torch.save({"model": dense.state_dict(), "args": vars(args)}, run_dir / "dense.pt")
    dense_bits = dense_fp32_bits_from_model(dense); dense_breakdown = add_kib_columns(dense_storage_breakdown(dense))
    append_result(rows, stage="dense_fp32", cluster_method=args.cluster_method, codebook_bits=32, clusters=0, test_acc=dense_test_acc, storage_kib=bits_to_kib(dense_bits), compression_vs_dense=1.0, **dense_breakdown)
    print(f"\n[2/5] {args.cluster_method} clustering")
    clustered = build_clustered_from_dense(dense_model=dense.cpu(), input_dim=data.input_dim, hidden_width=args.width, output_dim=data.num_classes, grid_size=args.grid_size, spline_order=args.spline_order, num_clusters=args.clusters, seed=args.seed, train_codebooks=True, cluster_method=args.cluster_method, function_samples=args.function_samples, function_domain=args.function_domain, train_loader=data.train_loader, device=device, activation_sample_batches=args.activation_sample_batches, include_base_in_function=args.include_base_in_function, normalize_function_signatures=not args.no_normalize_function_signatures, branch_spline_clusters=args.branch_spline_clusters, branch_base_clusters=args.branch_base_clusters, branch_spline_method=args.branch_spline_method, branch_function_samples=args.branch_function_samples, branch_function_domain=args.branch_function_domain, residual_fraction=args.residual_fraction, residual_base_clusters=args.residual_base_clusters, residual_selection=args.residual_selection).to(device)
    _, clust_acc = evaluate(clustered, data.test_loader, device, desc="clustered test before FT")
    clust_bits = clustered_fp32_bits_from_model(clustered); clust_breakdown = add_kib_columns(compressed_storage_breakdown(clustered, 32, 0))
    append_result(rows, stage="clustered_before_finetune_fp32_codebook", cluster_method=args.cluster_method, codebook_bits=32, clusters=args.clusters, test_acc=clust_acc, storage_kib=bits_to_kib(clust_bits), compression_vs_dense=compression_ratio(dense_bits, clust_bits), **clust_breakdown)
    print("\n[3/5] Fine-tuning clustered codebooks")
    opt_c = optim.AdamW(clustered.parameters(), lr=args.finetune_lr, weight_decay=0.0); best_c_val_acc = -1.0; best_c_state = None
    for epoch in range(1, args.finetune_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(clustered, data.train_loader, opt_c, device, epoch, desc="clustered FT")
        val_loss, val_acc = evaluate(clustered, data.val_loader, device, desc="clustered val")
        print(f"Clustered FT epoch {epoch}: train_acc={tr_acc:.2f}, val_acc={val_acc:.2f}")
        if val_acc > best_c_val_acc: best_c_val_acc = val_acc; best_c_state = {k:v.detach().cpu().clone() for k,v in clustered.state_dict().items()}
    clustered.load_state_dict(best_c_state); _, c_test_acc = evaluate(clustered, data.test_loader, device, desc="clustered FT test")
    torch.save({"model": clustered.state_dict(), "clustered_export": clustered.export_clustered_state(), "args": vars(args)}, run_dir / "clustered_finetuned.pt")
    c_bits = clustered_fp32_bits_from_model(clustered); c_breakdown = add_kib_columns(compressed_storage_breakdown(clustered, 32, 0))
    append_result(rows, stage="clustered_finetuned_fp32_codebook", cluster_method=args.cluster_method, codebook_bits=32, clusters=args.clusters, test_acc=c_test_acc, storage_kib=bits_to_kib(c_bits), compression_vs_dense=compression_ratio(dense_bits, c_bits), **c_breakdown)
    print("\n[4/5] Hardware-aware codebook quantization and evaluation")
    for bits in args.bits_list:
        qmodel = quantize_clustered_model(clustered.cpu(), bits=bits, input_dim=data.input_dim, hidden_width=args.width, output_dim=data.num_classes, grid_size=args.grid_size, spline_order=args.spline_order, per_vector=True).to(device)
        _, q_acc = evaluate(qmodel, data.test_loader, device, desc=f"HWQ W{bits} test")
        q_bits = hwq_bits_from_model(qmodel.cpu(), codebook_bits=bits, scale_bits=32)
        export_hwq_state(qmodel.cpu(), bits=bits, save_path=str(run_dir / f"hwq_w{bits}.pt"))
        q_breakdown = add_kib_columns(compressed_storage_breakdown(qmodel.cpu(), bits, 32))
        append_result(rows, stage=f"clustered_hwq_w{bits}", cluster_method=args.cluster_method, codebook_bits=bits, clusters=args.clusters, test_acc=q_acc, storage_kib=bits_to_kib(q_bits), compression_vs_dense=compression_ratio(dense_bits, q_bits), **q_breakdown)
    print("\n[5/5] Saving summary"); df = pd.DataFrame(rows); df.to_csv(run_dir / "summary.csv", index=False)
    print(df.to_string(index=False)); print("\nSaved to:", run_dir / "summary.csv")
if __name__ == "__main__": main()
