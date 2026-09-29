"""
Differentiable soft-to-hard codebook learning for SplineKAN compression.

This module is intentionally separate from the static K-means clustering code.
It implements a train-time continuous relaxation of index assignment:

    hard edge index  z_ij in {1,...,K}
    soft assignment  a_ij = softmax(logits_ij / temperature)
    edge function    w_ij = sum_k a_ijk * codebook_k

After training, the soft assignments are hardened with argmax and exported as
an ordinary IndexEfficientBranchSplineKAN. Therefore, deployment storage and
inference format remain hardware-friendly: codebooks + compact integer indices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spline import SplineLayer, IndexEfficientBranchSplineKAN

# NOTE: ``..compression.clustering`` imports ``..models``, so importing it at
# module scope makes ``import funcodekan.compression`` fail with a partially
# initialised module -- it only ever worked because every existing entry point
# happened to import ``funcodekan.models`` first. Deferred to call sites.


class SoftIndexBranchSplineKAN(nn.Module):
    """
    Soft-to-hard branch-index SplineKAN.

    This is the differentiable version of the index-efficient branch-aware
    model. During training, each edge uses a soft mixture over spline codebook
    entries. The base coefficient is tied to the same index distribution via a
    conditional base codebook. At export time, each soft vector is replaced by
    its argmax index, producing an IndexEfficientBranchSplineKAN.
    """

    compression_type = "soft_branch_index"

    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        output_dim: int,
        grid_size: int,
        spline_order: int,
        spline_codebooks,
        spline_cluster_ids,
        conditional_base_codebooks,
        init_logit_strength: float = 6.0,
        train_codebooks: bool = True,
        train_assignments: bool = True,
    ):
        super().__init__()
        widths = [input_dim, hidden_width, output_dim]
        self.layers = nn.ModuleList(
            [SplineLayer(din, dout, grid_size=grid_size, spline_order=spline_order)
             for din, dout in zip(widths[:-1], widths[1:])]
        )
        self.spline_codebooks = nn.ParameterList([
            nn.Parameter(cb.clone().float(), requires_grad=train_codebooks)
            for cb in spline_codebooks
        ])
        self.conditional_base_codebooks = nn.ParameterList([
            nn.Parameter((cb[:, None] if cb.dim() == 1 else cb).clone().float(), requires_grad=train_codebooks)
            for cb in conditional_base_codebooks
        ])
        self.assignment_logits = nn.ParameterList()
        for ids, cb in zip(spline_cluster_ids, spline_codebooks):
            ids = ids.clone().long()
            k = int(cb.shape[0])
            logits = torch.zeros(*ids.shape, k, dtype=torch.float32)
            logits.scatter_(-1, ids.unsqueeze(-1), float(init_logit_strength))
            self.assignment_logits.append(nn.Parameter(logits, requires_grad=train_assignments))

    def soft_assignments(self, idx: int, temperature: float = 1.0):
        temperature = max(float(temperature), 1e-4)
        return F.softmax(self.assignment_logits[idx] / temperature, dim=-1)

    def hard_cluster_ids(self, idx: int):
        return torch.argmax(self.assignment_logits[idx].detach(), dim=-1).long()

    def reconstruct_weight_soft(self, idx: int, temperature: float = 1.0):
        alpha = self.soft_assignments(idx, temperature=temperature)  # [out,in,K]
        spline_weight = torch.einsum("oik,kd->oid", alpha, self.spline_codebooks[idx])
        base_weight = torch.einsum("oik,kd->oid", alpha, self.conditional_base_codebooks[idx]).squeeze(-1)
        return torch.cat([spline_weight, base_weight.unsqueeze(-1)], dim=-1)

    def reconstruct_weight_hard(self, idx: int):
        ids = self.hard_cluster_ids(idx)
        spline_weight = self.spline_codebooks[idx][ids]
        base_weight = self.conditional_base_codebooks[idx][ids].squeeze(-1)
        return torch.cat([spline_weight, base_weight.unsqueeze(-1)], dim=-1)

    def forward(self, x, temperature: float = 1.0, hard: bool = False):
        for idx, layer in enumerate(self.layers):
            w = self.reconstruct_weight_hard(idx) if hard else self.reconstruct_weight_soft(idx, temperature)
            x = layer.forward_from_weight(x, w)
        return x

    def assignment_entropy_loss(self):
        """Normalized entropy. Minimize this to encourage hard assignments."""
        losses = []
        for logits in self.assignment_logits:
            p = F.softmax(logits, dim=-1)
            ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)
            ent = ent / torch.log(torch.tensor(float(logits.shape[-1]), device=logits.device))
            losses.append(ent.mean())
        return torch.stack(losses).mean()

    def assignment_balance_loss(self):
        """
        Optional anti-collapse loss. Minimize KL(mean assignment || uniform).
        A small positive weight prevents all edges from falling into one centroid.
        """
        losses = []
        for logits in self.assignment_logits:
            p_mean = F.softmax(logits, dim=-1).mean(dim=(0, 1))
            k = p_mean.numel()
            uniform = torch.full_like(p_mean, 1.0 / k)
            losses.append((p_mean * torch.log((p_mean / uniform).clamp_min(1e-8))).sum())
        return torch.stack(losses).mean()

    def export_hard_model(
        self,
        input_dim: int,
        hidden_width: int,
        output_dim: int,
        grid_size: int,
        spline_order: int,
        train_codebooks: bool = True,
    ):
        ids = [self.hard_cluster_ids(i).cpu() for i in range(len(self.spline_codebooks))]
        spline_cbs = [cb.detach().cpu().clone() for cb in self.spline_codebooks]
        base_cbs = [cb.detach().cpu().clone() for cb in self.conditional_base_codebooks]
        return IndexEfficientBranchSplineKAN(
            input_dim, hidden_width, output_dim, grid_size, spline_order,
            spline_cbs, ids, base_cbs, train_codebooks=train_codebooks,
        )


def build_soft_index_branch_from_dense(
    dense_model,
    input_dim: int,
    hidden_width: int,
    output_dim: int,
    grid_size: int,
    spline_order: int,
    spline_clusters: int = 32,
    seed: int = 42,
    spline_method: str = "function",
    function_samples: int = 128,
    function_domain: str = "grid",
    train_loader=None,
    device=None,
    activation_sample_batches: int = 8,
    normalize_signatures: bool = True,
    init_logit_strength: float = 6.0,
):
    from ..compression.clustering import cluster_spline_branch, _conditional_base_from_spline

    dense_cpu, spline_cbs, spline_ids = cluster_spline_branch(
        dense_model=dense_model,
        spline_clusters=spline_clusters,
        seed=seed,
        spline_method=spline_method,
        function_samples=function_samples,
        function_domain=function_domain,
        train_loader=train_loader,
        device=device,
        activation_sample_batches=activation_sample_batches,
        normalize_signatures=normalize_signatures,
    )
    conditional_base_cbs = _conditional_base_from_spline(dense_cpu, spline_cbs, spline_ids)
    return SoftIndexBranchSplineKAN(
        input_dim=input_dim,
        hidden_width=hidden_width,
        output_dim=output_dim,
        grid_size=grid_size,
        spline_order=spline_order,
        spline_codebooks=spline_cbs,
        spline_cluster_ids=spline_ids,
        conditional_base_codebooks=conditional_base_cbs,
        init_logit_strength=init_logit_strength,
        train_codebooks=True,
        train_assignments=True,
    )
