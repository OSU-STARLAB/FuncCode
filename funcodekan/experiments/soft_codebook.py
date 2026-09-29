import argparse, json, math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from ..data.mnist import get_mnist_loaders
from ..models.spline import DenseSplineKAN
from ..models.soft_codebook import build_soft_index_branch_from_dense
from ..utils.training import set_seed, get_device, train_one_epoch, evaluate
from ..quantization.hwq_pipeline import quantize_clustered_model, export_hwq_state
from ..analysis.storage import (
    dense_fp32_bits_from_model, clustered_fp32_bits_from_model, hwq_bits_from_model,
    bits_to_kib, compression_ratio, dense_storage_breakdown,
    compressed_storage_breakdown, add_kib_columns,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, default="soft_k32")
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--soft-epochs", type=int, default=10)
    p.add_argument("--hard-finetune-epochs", type=int, default=10)
    p.add_argument("--soft-lr", type=float, default=3e-4)
    p.add_argument("--hard-lr", type=float, default=5e-4)
    p.add_argument("--spline-clusters", type=int, default=32)
    p.add_argument("--spline-method", type=str, default="function", choices=["function", "coefficient"])
    p.add_argument("--function-domain", type=str, default="grid", choices=["grid", "activation"])
    p.add_argument("--function-samples", type=int, default=128)
    p.add_argument("--activation-sample-batches", type=int, default=8)
    p.add_argument("--init-logit-strength", type=float, default=6.0)
    p.add_argument("--temp-start", type=float, default=2.0)
    p.add_argument("--temp-end", type=float, default=0.25)
    p.add_argument("--distill-alpha", type=float, default=0.5)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--entropy-lambda", type=float, default=1e-3)
    p.add_argument("--balance-lambda", type=float, default=1e-3)
    p.add_argument("--bits-list", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--test-batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def append_result(rows, **kwargs):
    rows.append(kwargs)
    print(kwargs, flush=True)


def anneal_temperature(epoch, total_epochs, start, end):
    if total_epochs <= 1:
        return end
    t = (epoch - 1) / float(total_epochs - 1)
    return float(start * ((end / start) ** t))


def train_soft_epoch(model, teacher, loader, optimizer, device, epoch, args):
    model.train(); teacher.eval()
    temperature = anneal_temperature(epoch, args.soft_epochs, args.temp_start, args.temp_end)
    total_loss = total_correct = total_samples = 0
    pbar = tqdm(loader, desc=f"soft epoch {epoch} T={temperature:.3f}", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(x, temperature=temperature, hard=False)
        ce = F.cross_entropy(student_logits, y)
        with torch.no_grad():
            teacher_logits = teacher(x)
        tau = args.distill_temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / tau, dim=-1),
            F.softmax(teacher_logits / tau, dim=-1),
            reduction="batchmean",
        ) * (tau * tau)
        entropy = model.assignment_entropy_loss()
        balance = model.assignment_balance_loss()
        loss = (1.0 - args.distill_alpha) * ce + args.distill_alpha * kd + args.entropy_lambda * entropy + args.balance_lambda * balance
        loss.backward(); optimizer.step()
        total_loss += float(loss.item()) * y.numel()
        total_correct += int((student_logits.argmax(dim=-1) == y).sum().item())
        total_samples += int(y.numel())
        pbar.set_postfix(loss=loss.item(), ce=ce.item(), kd=kd.item(), ent=entropy.item())
    return total_loss / total_samples, 100.0 * total_correct / total_samples, temperature


@torch.no_grad()
def evaluate_soft(model, loader, device, temperature=0.25, hard=False, desc="soft eval"):
    model.eval(); total_correct = total_samples = 0; total_loss = 0.0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x, temperature=temperature, hard=hard)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * y.numel()
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_samples += int(y.numel())
    return total_loss / total_samples, 100.0 * total_correct / total_samples


def main():
    args = parse_args(); set_seed(args.seed); device = get_device(args.device)
    run_dir = Path(args.out_dir) / args.run_name; run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f: json.dump(vars(args), f, indent=2)
    print("="*70); print("Differentiable Soft-to-Hard Codebook MNIST experiment")
    print("Run directory:", run_dir); print("Device:", device); print("Spline clusters:", args.spline_clusters)

    data = get_mnist_loaders(batch_size=args.batch_size, test_batch_size=args.test_batch_size, seed=args.seed, num_workers=args.num_workers)
    rows = []

    print("\n[1/6] Train dense SplineKAN teacher")
    dense = DenseSplineKAN(data.input_dim, args.width, data.num_classes, args.grid_size, args.spline_order).to(device)
    opt = optim.AdamW(dense.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = -1.0; best_state = None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(dense, data.train_loader, opt, device, epoch, desc="dense train")
        val_loss, val_acc = evaluate(dense, data.val_loader, device, desc="dense val")
        print(f"Dense epoch {epoch}: train_acc={tr_acc:.2f}, val_acc={val_acc:.2f}")
        if val_acc > best_val:
            best_val = val_acc; best_state = {k: v.detach().cpu().clone() for k, v in dense.state_dict().items()}
    dense.load_state_dict(best_state)
    _, dense_test = evaluate(dense, data.test_loader, device, desc="dense test")
    torch.save({"model": dense.state_dict(), "args": vars(args)}, run_dir / "dense.pt")
    dense_bits = dense_fp32_bits_from_model(dense)
    append_result(rows, stage="dense_fp32", method="dense", codebook_bits=32, clusters=0,
                  test_acc=dense_test, storage_kib=bits_to_kib(dense_bits), compression_vs_dense=1.0,
                  **add_kib_columns(dense_storage_breakdown(dense)))

    print("\n[2/6] Initialize soft branch-index codebooks from function-space clustering")
    soft = build_soft_index_branch_from_dense(
        dense_model=dense.cpu(), input_dim=data.input_dim, hidden_width=args.width, output_dim=data.num_classes,
        grid_size=args.grid_size, spline_order=args.spline_order, spline_clusters=args.spline_clusters,
        seed=args.seed, spline_method=args.spline_method, function_samples=args.function_samples,
        function_domain=args.function_domain, train_loader=data.train_loader, device=device,
        activation_sample_batches=args.activation_sample_batches, init_logit_strength=args.init_logit_strength,
    ).to(device)
    dense = dense.to(device)
    _, soft_init_acc = evaluate_soft(soft, data.test_loader, device, temperature=args.temp_start, hard=False, desc="soft init test")
    hard_init = soft.export_hard_model(data.input_dim, args.width, data.num_classes, args.grid_size, args.spline_order, train_codebooks=True).to(device)
    _, hard_init_acc = evaluate(hard_init, data.test_loader, device, desc="hard init test")
    hard_bits = clustered_fp32_bits_from_model(hard_init.cpu())
    append_result(rows, stage="soft_initialized", method="soft_branch_index", codebook_bits=32, clusters=args.spline_clusters,
                  test_acc=soft_init_acc, storage_kib=bits_to_kib(hard_bits), compression_vs_dense=compression_ratio(dense_bits, hard_bits),
                  **add_kib_columns(compressed_storage_breakdown(hard_init.cpu(), 32, 0)))
    append_result(rows, stage="hard_argmax_before_soft_training", method="soft_branch_index", codebook_bits=32, clusters=args.spline_clusters,
                  test_acc=hard_init_acc, storage_kib=bits_to_kib(hard_bits), compression_vs_dense=compression_ratio(dense_bits, hard_bits),
                  **add_kib_columns(compressed_storage_breakdown(hard_init.cpu(), 32, 0)))

    print("\n[3/6] Differentiable soft-to-hard assignment learning")
    soft = soft.to(device); opt_s = optim.AdamW(soft.parameters(), lr=args.soft_lr, weight_decay=0.0)
    best_soft_val = -1.0; best_soft_state = None
    for epoch in range(1, args.soft_epochs + 1):
        tr_loss, tr_acc, temp = train_soft_epoch(soft, dense, data.train_loader, opt_s, device, epoch, args)
        _, val_acc = evaluate_soft(soft, data.val_loader, device, temperature=temp, hard=False, desc="soft val")
        _, hard_val_acc = evaluate_soft(soft, data.val_loader, device, temperature=temp, hard=True, desc="hard val")
        print(f"Soft epoch {epoch}: train_acc={tr_acc:.2f}, soft_val={val_acc:.2f}, hard_val={hard_val_acc:.2f}, T={temp:.3f}")
        # select by hard validation, because deployment uses hard indices
        if hard_val_acc > best_soft_val:
            best_soft_val = hard_val_acc
            best_soft_state = {k: v.detach().cpu().clone() for k, v in soft.state_dict().items()}
    soft.load_state_dict(best_soft_state)
    _, soft_test = evaluate_soft(soft, data.test_loader, device, temperature=args.temp_end, hard=False, desc="soft final test")
    _, hard_argmax_test = evaluate_soft(soft, data.test_loader, device, temperature=args.temp_end, hard=True, desc="hard argmax test")
    hard = soft.export_hard_model(data.input_dim, args.width, data.num_classes, args.grid_size, args.spline_order, train_codebooks=True).to(device)
    hard_bits = clustered_fp32_bits_from_model(hard.cpu())
    append_result(rows, stage="soft_trained", method="soft_branch_index", codebook_bits=32, clusters=args.spline_clusters,
                  test_acc=soft_test, storage_kib=bits_to_kib(hard_bits), compression_vs_dense=compression_ratio(dense_bits, hard_bits),
                  **add_kib_columns(compressed_storage_breakdown(hard.cpu(), 32, 0)))
    append_result(rows, stage="hard_argmax_after_soft_training", method="soft_branch_index", codebook_bits=32, clusters=args.spline_clusters,
                  test_acc=hard_argmax_test, storage_kib=bits_to_kib(hard_bits), compression_vs_dense=compression_ratio(dense_bits, hard_bits),
                  **add_kib_columns(compressed_storage_breakdown(hard.cpu(), 32, 0)))

    print("\n[4/6] Fine-tune hardened codebooks only")
    opt_h = optim.AdamW(hard.parameters(), lr=args.hard_lr, weight_decay=0.0)
    best_hard_val = -1.0; best_hard_state = None
    for epoch in range(1, args.hard_finetune_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(hard, data.train_loader, opt_h, device, epoch, desc="hard FT")
        _, val_acc = evaluate(hard, data.val_loader, device, desc="hard val")
        print(f"Hard FT epoch {epoch}: train_acc={tr_acc:.2f}, val_acc={val_acc:.2f}")
        if val_acc > best_hard_val:
            best_hard_val = val_acc; best_hard_state = {k: v.detach().cpu().clone() for k, v in hard.state_dict().items()}
    hard.load_state_dict(best_hard_state)
    _, hard_test = evaluate(hard, data.test_loader, device, desc="hard FT test")
    torch.save({"model": hard.state_dict(), "clustered_export": hard.export_clustered_state(), "args": vars(args)}, run_dir / "hard_finetuned.pt")
    hard_bits = clustered_fp32_bits_from_model(hard.cpu())
    append_result(rows, stage="hard_finetuned_fp32_codebook", method="soft_branch_index", codebook_bits=32, clusters=args.spline_clusters,
                  test_acc=hard_test, storage_kib=bits_to_kib(hard_bits), compression_vs_dense=compression_ratio(dense_bits, hard_bits),
                  **add_kib_columns(compressed_storage_breakdown(hard.cpu(), 32, 0)))

    print("\n[5/6] Hardware-aware codebook quantization")
    for bits in args.bits_list:
        qmodel = quantize_clustered_model(hard.cpu(), bits=bits, input_dim=data.input_dim, hidden_width=args.width,
                                          output_dim=data.num_classes, grid_size=args.grid_size, spline_order=args.spline_order,
                                          per_vector=True).to(device)
        _, q_acc = evaluate(qmodel, data.test_loader, device, desc=f"HWQ W{bits} test")
        q_bits = hwq_bits_from_model(qmodel.cpu(), codebook_bits=bits, scale_bits=32)
        export_hwq_state(qmodel.cpu(), bits=bits, save_path=str(run_dir / f"hwq_w{bits}.pt"))
        append_result(rows, stage=f"hard_finetuned_hwq_w{bits}", method="soft_branch_index", codebook_bits=bits, clusters=args.spline_clusters,
                      test_acc=q_acc, storage_kib=bits_to_kib(q_bits), compression_vs_dense=compression_ratio(dense_bits, q_bits),
                      **add_kib_columns(compressed_storage_breakdown(qmodel.cpu(), bits, 32)))

    print("\n[6/6] Save summary")
    df = pd.DataFrame(rows); df.to_csv(run_dir / "summary.csv", index=False)
    print(df.to_string(index=False)); print("\nSaved to:", run_dir / "summary.csv")


if __name__ == "__main__":
    main()
