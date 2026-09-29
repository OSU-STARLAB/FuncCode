"""Competing compression methods for the conv-KAGN benchmark.

Every baseline here operates on the *same* trained dense checkpoint, gets the
*same* fine-tuning budget as FuncCode, and is charged storage with the *same*
accountant (``conv_compression.storage_breakdown`` conventions: norm parameters
and folded BN statistics always counted, scales always counted). Any budget
asymmetry between a baseline and FuncCode would show up directly in the
bits-per-edge column, which is why that column is reported for every arm.

Methods
-------
uniform    per-output-channel symmetric PTQ at W-bits  -> C*b bits/edge
lsq        learned step size QAT (Esser et al. 2020)    -> C*b bits/edge
prune      magnitude prune + W4, CSR index cost charged -> varies
pq         product quantisation of coefficient sub-vectors, the strongest
           non-functional codebook baseline             -> m*log2(K) bits/edge
iso        a narrower dense network at matched storage  -> C*32 bits/edge
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.conv_kagn import compressible_layers
from .conv_compression import bits_to_kib, index_bits_for, kmeans_large

__all__ = ["UniformFakeQuant", "apply_uniform_ptq_", "LSQEdgeWeights", "attach_lsq_",
           "apply_magnitude_prune_", "apply_product_quantization_", "baseline_storage_bits"]


# --------------------------------------------------------------------------
# Uniform PTQ
# --------------------------------------------------------------------------

def _sym_quant(x: torch.Tensor, bits: int, dim: int = 0) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    other = [d for d in range(x.dim()) if d != dim]
    scale = (x.abs().amax(dim=other, keepdim=True) / qmax).clamp(min=1e-12)
    return torch.round(x / scale).clamp(-qmax - 1, qmax) * scale


@torch.no_grad()
def apply_uniform_ptq_(model: nn.Module, bits: int) -> nn.Module:
    """Per-output-channel symmetric PTQ on both weight branches."""
    for _, layer in compressible_layers(model):
        p = layer.wprov
        if hasattr(p, "poly_weight"):
            p.poly_weight.copy_(_sym_quant(p.poly_weight.data, bits, dim=0))
            p.base_weight.copy_(_sym_quant(p.base_weight.data, bits, dim=0))
    return model


class UniformFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qn, qp):
        ctx.save_for_backward(x, scale)
        ctx.other = (qn, qp)
        return torch.round((x / scale).clamp(qn, qp)) * scale

    @staticmethod
    def backward(ctx, g):
        x, scale = ctx.saved_tensors
        qn, qp = ctx.other
        q = x / scale
        inside = (q >= qn) & (q <= qp)
        return g * inside, None, None, None


# --------------------------------------------------------------------------
# LSQ (learned step size quantization)
# --------------------------------------------------------------------------

class _LSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step, qn, qp, grad_scale):
        ctx.save_for_backward(x, step)
        ctx.other = (qn, qp, grad_scale)
        q = (x / step).clamp(qn, qp)
        return torch.round(q) * step

    @staticmethod
    def backward(ctx, g):
        x, step = ctx.saved_tensors
        qn, qp, gs = ctx.other
        q = x / step
        lower, upper = q < qn, q > qp
        inside = ~(lower | upper)
        g_step = torch.where(lower, torch.full_like(q, qn),
                             torch.where(upper, torch.full_like(q, qp),
                                         torch.round(q) - q))
        g_step = (g_step * g * gs).sum().view(step.shape)
        return g * inside, g_step, None, None, None


class LSQEdgeWeights(nn.Module):
    """Wraps a dense edge provider with learnable per-tensor step sizes."""

    kind = "lsq"

    def __init__(self, dense, bits: int = 4):
        super().__init__()
        self.inner = dense
        self.bits = bits
        self.qn, self.qp = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        for attr, w in (("poly", dense.poly_weight), ("base", dense.base_weight)):
            init = 2 * w.detach().abs().mean() / math.sqrt(self.qp)
            self.register_parameter(f"step_{attr}", nn.Parameter(init.clamp(min=1e-8)))
            self.register_buffer(f"gs_{attr}", torch.tensor(1.0 / math.sqrt(w.numel() * self.qp)))
        # forwarded metadata so the storage accountant can treat it like any other
        self.n_edges = dense.n_edges
        self.coeff_dim = dense.coeff_dim
        self.degree = dense.degree

    def forward(self):
        p = _LSQ.apply(self.inner.poly_weight, self.step_poly.abs().clamp(min=1e-8),
                       self.qn, self.qp, self.gs_poly.item())
        b = _LSQ.apply(self.inner.base_weight, self.step_base.abs().clamp(min=1e-8),
                       self.qn, self.qp, self.gs_base.item())
        return p, b

    def edge_matrix(self):
        return self.inner.edge_matrix()


def attach_lsq_(model: nn.Module, bits: int = 4, skip_first: bool = False,
                skip_head: bool = False) -> nn.Module:
    layers = compressible_layers(model)
    for i, (_, layer) in enumerate(layers):
        if (skip_first and i == 0) or (skip_head and i == len(layers) - 1):
            continue
        layer.wprov = LSQEdgeWeights(layer.wprov, bits)
    return model


# --------------------------------------------------------------------------
# Magnitude pruning
# --------------------------------------------------------------------------

@torch.no_grad()
def apply_magnitude_prune_(model: nn.Module, sparsity: float, bits: Optional[int] = 4
                           ) -> Tuple[nn.Module, Dict[str, float]]:
    """Global magnitude prune of edge coefficients, then optional PTQ.

    Storage is charged honestly: a bitmask of one bit per coefficient plus
    ``bits`` per surviving value. This is what kills naive pruning as a
    compression baseline at high bit-widths, and saying so is part of the
    argument for indexing whole edges instead.
    """
    allw = torch.cat([layer.wprov.edge_matrix().detach().abs().flatten()
                      for _, layer in compressible_layers(model)])
    thresh = torch.quantile(allw.float()[torch.randperm(allw.numel())[:2_000_000]], sparsity)

    kept = total = 0
    for _, layer in compressible_layers(model):
        p = layer.wprov
        for w in (p.poly_weight, p.base_weight):
            mask = w.abs() > thresh
            w.mul_(mask)
            kept += int(mask.sum())
            total += w.numel()
    if bits:
        apply_uniform_ptq_(model, bits)
    return model, {"kept": kept, "total": total, "actual_sparsity": 1 - kept / max(total, 1)}


# --------------------------------------------------------------------------
# Product quantization (the strongest non-functional codebook baseline)
# --------------------------------------------------------------------------

@torch.no_grad()
def apply_product_quantization_(model: nn.Module, subvectors: int = 2, clusters: int = 32,
                                seed: int = 42, device=None) -> nn.Module:
    """Split each edge's C coefficients into ``subvectors`` chunks, k-means each.

    This is the honest ablation for "is the *function-space* metric doing the
    work, or would any codebook do?". PQ costs ``subvectors * log2(K)``
    bits/edge, so at m=2, K=32 it is 10 bits/edge against FuncCode's 5.
    """
    for _, layer in compressible_layers(model):
        p = layer.wprov
        W = p.edge_matrix().detach().float()
        C = W.shape[1]
        recon = torch.empty_like(W)
        for b in np.array_split(np.arange(C), subvectors):
            sl = slice(int(b[0]), int(b[-1]) + 1)
            centers, labels = kmeans_large(W[:, sl].contiguous(), clusters, seed=seed, device=device)
            recon[:, sl] = centers[labels]
        # A PQ model is not a single shared codebook, so we write the
        # reconstruction back into the dense weights and charge the true PQ bit
        # cost in baseline_storage_bits(method="pq").
        poly, base = p._edges_to_conv(recon)
        p.poly_weight.copy_(poly)
        p.base_weight.copy_(base)
        layer.pq_bits_per_edge = subvectors * index_bits_for(clusters)
    return model


# --------------------------------------------------------------------------
# Shared storage accountant for the baselines
# --------------------------------------------------------------------------

def baseline_storage_bits(model: nn.Module, method: str, bits: int = 4,
                          sparsity: float = 0.0, subvectors: int = 2,
                          clusters: int = 32, other_bits: int = 16,
                          scale_bits: int = 16) -> Dict[str, float]:
    """Storage under the same conventions as ``storage_breakdown``."""
    out = {"weight_bits": 0, "index_bits": 0, "scale_bits": 0, "norm_bits": 0}

    edge_ids = set()
    for _, layer in compressible_layers(model):
        p = layer.wprov
        inner = getattr(p, "inner", p)
        edge_ids.update(id(t) for t in p.parameters())
        E, C = p.n_edges, p.coeff_dim
        n_out = inner.out_features

        if method in ("uniform", "lsq"):
            out["weight_bits"] += E * C * bits
            out["scale_bits"] += 2 * n_out * scale_bits          # one scale per branch per out-channel
        elif method == "prune":
            out["weight_bits"] += int(E * C * (1 - sparsity) * bits)
            out["index_bits"] += E * C                            # 1-bit mask per coefficient
            out["scale_bits"] += 2 * n_out * scale_bits
        elif method == "pq":
            out["weight_bits"] += subvectors * clusters * (C / subvectors) * 32
            out["index_bits"] += E * subvectors * index_bits_for(clusters)
            out["scale_bits"] += subvectors * clusters * scale_bits
        elif method == "dense":
            out["weight_bits"] += E * C * 32
        else:
            raise ValueError(method)

    for name, prm in model.named_parameters():
        if id(prm) not in edge_ids:
            out["norm_bits"] += prm.numel() * other_bits
    for name, buf in model.named_buffers():
        if name.endswith("running_mean") or name.endswith("running_var"):
            out["norm_bits"] += buf.numel() * other_bits

    total = sum(out.values())
    out["storage_total_bits"] = int(total)
    out["storage_total_kib"] = bits_to_kib(total)
    return out
