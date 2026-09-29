"""QAT wrappers for the HW contract (see docs/HW_DESIGN_CONTRACT.md).

`QuantSplineKAN` composes an UNTOUCHED `DenseSplineKAN` (D2). The quantized
forward — shared with the D3 wrapper in act_quant.py — applies fake-quant at
exactly the four contract points: (a) layer-1 input, (b) spline coefficients,
(c) base weights, (d) inter-layer activation, plus the Q4.12 LUT rounding of
basis/SiLU values that the HW bakes into its 16-entry LUTs. Training-time
forward therefore equals the HW forward up to float accumulation order.
"""

from __future__ import annotations

import copy
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from funcodekan.models.spline import DenseSplineKAN
from funcodekan.utils.training import evaluate, train_one_epoch

from .hw_spec import (ACT_BITS, WEIGHT_BITS, lut_quantize_basis_ste,
                      lut_quantize_silu_ste)
from .lsq import LsqQuantizer


def quant_spline_forward(
    layers: Sequence[nn.Module],
    x: torch.Tensor,
    weight_fn: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    act_quants: Sequence[LsqQuantizer],
) -> torch.Tensor:
    """The HW-contract forward, shared by D2 and D3 wrappers.

    weight_fn(l) must return the FAKE-QUANT (spline_weight [out,in,8],
    base_weight [out,in]) for layer l; how they are produced (dense arrays vs
    codebook lookup) is the only D2/D3 difference.
    """
    for l, layer in enumerate(layers):
        x = act_quants[l](x)
        spline_w, base_w = weight_fn(l)
        basis = lut_quantize_basis_ste(layer.b_splines(x))
        silu = lut_quantize_silu_ste(F.silu(x))
        x = (torch.einsum("nik,oik->no", basis, spline_w)
             + F.linear(silu, base_w))
    return x  # logits stay unquantized (INT32 fixed-point in HW)


class QuantSplineKAN(nn.Module):
    """D2: LSQ W4A4 fake-quant wrapper around a dense spline KAN."""

    def __init__(self, dense: DenseSplineKAN,
                 weight_bits: int = WEIGHT_BITS, act_bits: int = ACT_BITS):
        super().__init__()
        self.dense = dense
        n = len(dense.layers)
        self.act_quants = nn.ModuleList(
            [LsqQuantizer(act_bits, signed=True) for _ in range(n)])
        self.w_spline_quants = nn.ModuleList(
            [LsqQuantizer(weight_bits, signed=True) for _ in range(n)])
        self.w_base_quants = nn.ModuleList(
            [LsqQuantizer(weight_bits, signed=True) for _ in range(n)])
        for l, w in enumerate(dense.weights):
            self.w_spline_quants[l].init_from(w[..., :-1])
            self.w_base_quants[l].init_from(w[..., -1])

    def quant_weights(self, l: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.dense.weights[l]
        return (self.w_spline_quants[l](w[..., :-1]),
                self.w_base_quants[l](w[..., -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quant_spline_forward(self.dense.layers, x,
                                    self.quant_weights, self.act_quants)

    @torch.no_grad()
    def int_weights(self, l: int):
        """Frozen INT4 codes + steps for export/emulator."""
        w = self.dense.weights[l].detach().cpu()
        sq, bq = self.w_spline_quants[l], self.w_base_quants[l]
        return {
            "spline_q": sq.quantize_int(w[..., :-1]),
            "base_q": bq.quantize_int(w[..., -1]),
            "spline_step": sq.step_size(),
            "base_step": bq.step_size(),
        }

    def act_steps(self) -> list[float]:
        return [q.step_size() for q in self.act_quants]


def step_parameters(module: nn.Module) -> list[nn.Parameter]:
    """All LSQ step-size parameters inside a wrapper."""
    return [m.s for m in module.modules() if isinstance(m, LsqQuantizer)]


def run_qat(model: nn.Module, bundle, device, epochs: int,
            lr: float = 5e-4, weight_decay: float = 1e-4,
            step_lr: float | None = None, log=print) -> dict:
    """Cosine-LR QAT loop reusing the verified train_one_epoch/evaluate.

    Tracks the best-val state (same pattern as the verified mnist driver)
    and reloads it before the final test evaluation.
    """
    steps = step_parameters(model)
    step_ids = {id(p) for p in steps}
    others = [p for p in model.parameters()
              if p.requires_grad and id(p) not in step_ids]
    optimizer = torch.optim.AdamW([
        {"params": others, "lr": lr, "weight_decay": weight_decay},
        {"params": steps, "lr": step_lr if step_lr is not None else lr,
         "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))

    model.to(device)
    best_val, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, bundle.train_loader, optimizer, device, epoch, desc="qat")
        scheduler.step()
        val_loss, val_acc = evaluate(model, bundle.val_loader, device,
                                     desc="qat-val")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss,
                        "val_acc": val_acc})
        log(f"[qat] epoch {epoch}/{epochs} train {train_acc:.2f}% "
            f"val {val_acc:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_acc = evaluate(model, bundle.test_loader, device, desc="qat-test")
    return {"best_val_acc": best_val, "test_acc": test_acc,
            "history": history}
