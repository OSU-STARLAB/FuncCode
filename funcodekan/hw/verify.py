"""Verification-ladder drivers (L1, L4 here; L2/L3 compare HLS outputs).

L1: PyTorch fake-quant wrapper <-> integer emulator, full 10k MNIST test set,
    argmax must match 10,000/10,000 (logit representations differ by design:
    float vs INT32 fixed-point; see HW_DESIGN_CONTRACT.md).
    D1: torch FP32 <-> numpy FP32, logits within 1e-4.
L4: emulator full-test-set accuracy within 0.3 pp of the FP32 dense
    reference.

Every check appends an entry to runs_hw/verification_log.json.
"""

from __future__ import annotations

import numpy as np
import torch

from .veriflog import append_verification_entry


@torch.no_grad()
def _collect_test_set(test_loader) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys)


@torch.no_grad()
def _torch_logits(model, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    model = model.cpu().eval()
    outs = []
    for i in range(0, len(x), batch):
        outs.append(model(torch.from_numpy(x[i:i + batch]).float()).numpy())
    return np.concatenate(outs)


def run_l1_quant(design: str, wrapper, fp_model, test_loader,
                 log_path="runs_hw/verification_log.json") -> dict:
    """L1 for D2/D3: fake-quant torch vs integer emulator on the full test set."""
    x, y = _collect_test_set(test_loader)
    torch_pred = _torch_logits(wrapper, x).argmax(axis=1)
    emu_logits = fp_model.forward_float(x)
    emu_pred = emu_logits.argmax(axis=1)
    n = len(y)
    agree = int((torch_pred == emu_pred).sum())
    emu_acc = 100.0 * float((emu_pred == y).mean())
    torch_acc = 100.0 * float((torch_pred == y).mean())
    result = {
        "milestone": "M3", "rung": "L1", "design": design,
        "n_images": n, "argmax_agree": agree,
        "argmax_mismatch_indices": np.nonzero(torch_pred != emu_pred)[0][:50].tolist(),
        "torch_fake_quant_acc": torch_acc, "emulator_acc": emu_acc,
        "passed": agree == n,
    }
    append_verification_entry(result, log_path)
    return result


def run_l1_fp32(dense, np_model, test_loader,
                log_path="runs_hw/verification_log.json") -> dict:
    """L1 for D1: torch FP32 vs numpy FP32 path, logits <= 1e-4."""
    x, y = _collect_test_set(test_loader)
    t_logits = _torch_logits(dense, x)
    n_logits = np_model.forward(x)
    max_abs = float(np.max(np.abs(t_logits - n_logits)))
    agree = int((t_logits.argmax(1) == n_logits.argmax(1)).sum())
    acc = 100.0 * float((n_logits.argmax(1) == y).mean())
    result = {
        "milestone": "M3", "rung": "L1", "design": "d1_fp32",
        "n_images": len(y), "max_abs_logit_diff": max_abs,
        "argmax_agree": agree, "numpy_acc": acc,
        "passed": max_abs <= 1e-4 and agree == len(y),
    }
    append_verification_entry(result, log_path)
    return result


def run_l4(design: str, emulator_acc: float, dense_fp32_acc: float,
           log_path="runs_hw/verification_log.json") -> dict:
    """L4: near-lossless claim — emulator accuracy within 0.3 pp of dense."""
    delta = dense_fp32_acc - emulator_acc
    result = {
        "milestone": "M3", "rung": "L4", "design": design,
        "emulator_acc": emulator_acc, "dense_fp32_acc": dense_fp32_acc,
        "delta_pp": delta, "passed": delta <= 0.3,
    }
    append_verification_entry(result, log_path)
    return result
