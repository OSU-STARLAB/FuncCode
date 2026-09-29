"""D3: activation quantization on top of the verified branch-aware model.

Wraps an UNTOUCHED `BranchAwareClusteredSplineKAN` (Ks=32, Kb=16 from the
verified pipeline) with the SAME LSQ quantizers at the SAME points as the D2
wrapper (funcodekan/hw/qat.py). Weight quantization is symmetric INT4 with a
per-tensor step on each codebook, matching D2's per-tensor weight steps, so
D2 vs D3 isolates codebook sharing only. Cluster indices are frozen buffers
— only codebook entries and LSQ steps train during the short finetune.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from funcodekan.models.spline import BranchAwareClusteredSplineKAN

from .hw_spec import ACT_BITS, WEIGHT_BITS
from .lsq import LsqQuantizer
from .qat import quant_spline_forward


def rebuild_branch_model(clustered_export: dict, input_dim: int,
                         hidden_width: int, output_dim: int,
                         grid_size: int, spline_order: int,
                         train_codebooks: bool = True
                         ) -> BranchAwareClusteredSplineKAN:
    """Reconstruct the verified branch model from a clustered_finetuned.pt
    'clustered_export' dict (keys spline_clustered_weights.{i}, ...)."""
    n = len([k for k in clustered_export if k.startswith("spline_clustered_weights.")])
    return BranchAwareClusteredSplineKAN(
        input_dim, hidden_width, output_dim, grid_size, spline_order,
        spline_codebooks=[clustered_export[f"spline_clustered_weights.{i}"] for i in range(n)],
        spline_cluster_ids=[clustered_export[f"spline_cluster_ids.{i}"] for i in range(n)],
        base_codebooks=[clustered_export[f"base_clustered_weights.{i}"] for i in range(n)],
        base_cluster_ids=[clustered_export[f"base_cluster_ids.{i}"] for i in range(n)],
        train_codebooks=train_codebooks,
    )


class QuantBranchKAN(nn.Module):
    """FuncCode W4A4: fake-quant codebooks + LSQ activations, frozen indices."""

    def __init__(self, branch: BranchAwareClusteredSplineKAN,
                 weight_bits: int = WEIGHT_BITS, act_bits: int = ACT_BITS):
        super().__init__()
        self.branch = branch
        n = len(branch.layers)
        self.act_quants = nn.ModuleList(
            [LsqQuantizer(act_bits, signed=True) for _ in range(n)])
        self.cb_spline_quants = nn.ModuleList(
            [LsqQuantizer(weight_bits, signed=True) for _ in range(n)])
        self.cb_base_quants = nn.ModuleList(
            [LsqQuantizer(weight_bits, signed=True) for _ in range(n)])
        for l in range(n):
            self.cb_spline_quants[l].init_from(branch.spline_codebooks[l])
            self.cb_base_quants[l].init_from(branch.base_codebooks[l])

    def quant_weights(self, l: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Fake-quant the CODEBOOKS, then expand through the frozen indices —
        # exactly the HW decode order (codebook BRAM holds INT4 entries).
        spline_cb = self.cb_spline_quants[l](self.branch.spline_codebooks[l])
        base_cb = self.cb_base_quants[l](self.branch.base_codebooks[l])
        sids = self.branch.get_spline_cluster_ids(l)
        bids = self.branch.get_base_cluster_ids(l)
        return spline_cb[sids], base_cb[bids].squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quant_spline_forward(self.branch.layers, x,
                                    self.quant_weights, self.act_quants)

    @torch.no_grad()
    def int_codebooks(self, l: int):
        """Frozen INT4 codebooks + indices + steps for export/emulator."""
        scb = self.branch.spline_codebooks[l].detach().cpu()
        bcb = self.branch.base_codebooks[l].detach().cpu()
        sq, bq = self.cb_spline_quants[l], self.cb_base_quants[l]
        return {
            "spline_codebook_q": sq.quantize_int(scb),            # [Ks, 8]
            "base_codebook_q": bq.quantize_int(bcb).squeeze(-1),  # [Kb]
            "spline_ids": self.branch.get_spline_cluster_ids(l).detach().cpu(),
            "base_ids": self.branch.get_base_cluster_ids(l).detach().cpu(),
            "spline_step": sq.step_size(),
            "base_step": bq.step_size(),
        }

    def act_steps(self) -> list[float]:
        return [q.step_size() for q in self.act_quants]
