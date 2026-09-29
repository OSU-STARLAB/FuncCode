"""KAGN convolutional MNIST model + FuncCode branch compression (ADDITIVE).

The external `kans` package (zoo KAGNConv2DLayer) is absent from this
environment and the verified pipeline does not support conv compression
("future work" per README). This module is therefore NEW research code —
NOT paper-verified-pipeline pedigree — implementing:

  GramConv2D: the conv realization of the SAME layer math as the verified
      FC gram variant (funcodekan/models/variants.py GramPolynomialLayer):
      basis = [1, z, z*P1-0.1*P0, ...] on z = tanh(x) per pixel;
      out = SiLU(InstanceNorm2d(base_conv(SiLU(x)) + poly_conv(basis(x))))
      (InstanceNorm2d(affine=True) per the zoo's VGG-KAGN norm choice —
      no train/eval gap, and its per-channel spatial normalization reuses
      the integer-LayerNorm hardware).

  KagnConvNet (MNIST): GramConv2D 1->16 s2 -> 14x14, GramConv2D 16->32 s2
      -> 7x7, global average pool, gram FC head 32->10 (the verified
      GramPolynomialLayer math, LayerNorm+SiLU output).

  Branch-aware compression: each conv kernel position (out_ch, in_ch, kh,
      kw) is an "edge" carrying 4 poly coefficients + 1 base weight —
      clustered exactly like FC edges (function-space signatures on a grid
      domain -> KMeans Ks; scalar base KMeans Kb), frozen index buffers,
      trainable codebooks. Storage accounting reuses
      funcodekan.analysis.storage.branch_storage_breakdown via the same
      attribute contract (spline_codebooks/base_codebooks/get_*_cluster_ids).

Every claim about these models is backed by the same bit-exact
verification ladder as the other families (see kagn_qat / kagn_fixed_point
/ kagn_export).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import order guard: models before compression (pre-existing circular dep)
import funcodekan.models  # noqa: F401
from funcodekan.models.variants import GramPolynomialLayer

DEGREE = 3
BASIS_DIM = DEGREE + 1
COEFF_DIM = BASIS_DIM + 1


def gram_basis(x: torch.Tensor, degree: int = DEGREE) -> torch.Tensor:
    """The verified gram recurrence (variants.py), stacked on dim=-1."""
    z = torch.tanh(x)
    terms = [torch.ones_like(z)]
    if degree >= 1:
        terms.append(z)
    for _ in range(2, degree + 1):
        terms.append(z * terms[-1] - 0.1 * terms[-2])
    return torch.stack(terms, dim=-1)


class GramConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 degree=DEGREE):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.degree = degree
        self.basis_dim = degree + 1
        self.coeff_dim = self.basis_dim + 1
        w = torch.empty(out_ch, in_ch, kernel_size, kernel_size,
                        self.coeff_dim)
        nn.init.trunc_normal_(w, mean=0.0, std=0.05)
        self.weight = nn.Parameter(w)
        # InstanceNorm2d (affine), as in the zoo's VGG-KAGN variants:
        # BatchNorm was structurally unstable on this small net (train 95%+
        # / eval collapse — running stats compound across layers), while
        # instance norm has ZERO train/eval gap and maps directly onto the
        # verified integer-LayerNorm hardware (per-channel over spatial
        # positions). track_running_stats defaults to False.
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)

    def forward_from_weight(self, x: torch.Tensor,
                            weight: torch.Tensor) -> torch.Tensor:
        """Pre-activation z = base_conv(SiLU(x)) + poly_conv(basis(x)) and
        the full output SiLU(BN(z)) — split so wrappers can intercept."""
        return F.silu(self.norm(self.pre_norm_from_weight(x, weight)))

    def pre_norm_from_weight(self, x, weight, basis_x=None, silu_x=None):
        n, c, h, w = x.shape
        k = self.kernel_size
        poly_w = weight[..., :-1]                    # [O, I, k, k, K]
        base_w = weight[..., -1]                     # [O, I, k, k]
        basis = gram_basis(x, self.degree) if basis_x is None else basis_x
        silu = F.silu(x) if silu_x is None else silu_x
        # [N, C, H, W, K] -> [N, C*K, H, W]; conv weight [O, C*K, k, k]
        basis_flat = basis.permute(0, 1, 4, 2, 3).reshape(
            n, c * self.basis_dim, h, w)
        poly_flat = poly_w.permute(0, 1, 4, 2, 3).reshape(
            self.out_ch, self.in_ch * self.basis_dim, k, k)
        nonlinear = F.conv2d(basis_flat, poly_flat, stride=self.stride,
                             padding=self.padding)
        base = F.conv2d(silu, base_w, stride=self.stride,
                        padding=self.padding)
        return base + nonlinear

    def forward(self, x):
        return self.forward_from_weight(x, self.weight)


class KagnConvNet(nn.Module):
    """Dense KAGN conv MNIST model (K-series FP32 reference)."""

    def __init__(self, channels=(16, 32), num_classes=10, degree=DEGREE):
        super().__init__()
        c1, c2 = channels
        self.conv1 = GramConv2D(1, c1, 3, stride=2, padding=1, degree=degree)
        self.conv2 = GramConv2D(c1, c2, 3, stride=2, padding=1, degree=degree)
        self.head_layer = GramPolynomialLayer(c2, num_classes, degree=degree)
        hw = torch.empty(num_classes, c2, self.head_layer.coeff_dim)
        nn.init.trunc_normal_(hw, mean=0.0, std=0.05)
        self.head_weight = nn.Parameter(hw)

    def conv_weight(self, i: int) -> torch.Tensor:
        return (self.conv1, self.conv2)[i].weight

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean(dim=(2, 3))                       # global average pool
        return self.head_layer.forward_from_weight(x, self.head_weight)

    def dense_storage_bits(self) -> int:
        return int(sum(p.numel() * 32 for p in
                       (self.conv1.weight, self.conv2.weight,
                        self.head_weight)))


# ---------------------------------------------------------------------------
# Branch-aware compression (conv edges + FC head edges)
# ---------------------------------------------------------------------------

def _edge_signatures(poly_vecs: torch.Tensor, samples: int = 128,
                     lo: float = -2.5, hi: float = 2.5) -> torch.Tensor:
    """Function-space signatures of poly-only edge functions on a grid
    domain (same domain/normalization convention as the verified
    cross-variant clustering)."""
    x = torch.linspace(lo, hi, samples)
    basis = gram_basis(x)                            # [S, K]
    sig = poly_vecs @ basis.T                        # [E, S]
    sig = sig - sig.mean(dim=1, keepdim=True)
    sig = sig / (sig.std(dim=1, keepdim=True) + 1e-6)
    return sig


def _kmeans(vectors: np.ndarray, k: int, seed: int):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=min(k, len(vectors)), random_state=seed,
                n_init=10)
    ids = km.fit_predict(vectors)
    return ids


def cluster_branch_conv(weight: torch.Tensor, ks: int, kb: int, seed: int):
    """weight [..., COEFF_DIM] -> (poly codebook [ks,4], poly ids [...],
    base codebook [kb,1], base ids [...]). Codebook centers = mean of the
    raw coefficient vectors per cluster (verified convention)."""
    shape = weight.shape[:-1]
    flat = weight.detach().cpu().reshape(-1, weight.shape[-1])
    poly = flat[:, :-1]
    base = flat[:, -1:]
    sig = _edge_signatures(poly).numpy()
    sids = _kmeans(sig, ks, seed)
    n_s = sids.max() + 1
    poly_cb = torch.stack([poly[sids == c].mean(dim=0)
                           for c in range(n_s)])
    bids = _kmeans(base.numpy(), kb, seed)
    n_b = bids.max() + 1
    base_cb = torch.stack([base[bids == c].mean(dim=0)
                           for c in range(n_b)])
    return (poly_cb, torch.from_numpy(sids).long().reshape(shape),
            base_cb, torch.from_numpy(bids).long().reshape(shape))


class CompressedKagnConvNet(nn.Module):
    """Branch-compressed KAGN conv net: per layer a poly codebook + base
    codebook + FROZEN index buffers. Attribute contract matches
    funcodekan.analysis.storage.branch_storage_breakdown."""

    compression_type = "branch"

    def __init__(self, dense: KagnConvNet, ks=32, kb=16, seed=42,
                 train_codebooks=True):
        super().__init__()
        self.conv1, self.conv2 = dense.conv1, dense.conv2
        self.head_layer = dense.head_layer
        self.spline_codebooks = nn.ParameterList()
        self.base_codebooks = nn.ParameterList()
        self.spline_id_names, self.base_id_names = [], []
        weights = [dense.conv1.weight, dense.conv2.weight,
                   dense.head_weight]
        for i, w in enumerate(weights):
            pcb, pid, bcb, bid = cluster_branch_conv(w, ks, kb, seed)
            self.spline_codebooks.append(
                nn.Parameter(pcb.clone().float(),
                             requires_grad=train_codebooks))
            self.base_codebooks.append(
                nn.Parameter(bcb.clone().float(),
                             requires_grad=train_codebooks))
            self.register_buffer(f"spline_cluster_ids_{i}", pid)
            self.spline_id_names.append(f"spline_cluster_ids_{i}")
            self.register_buffer(f"base_cluster_ids_{i}", bid)
            self.base_id_names.append(f"base_cluster_ids_{i}")

    def get_spline_cluster_ids(self, i):
        return getattr(self, self.spline_id_names[i])

    def get_base_cluster_ids(self, i):
        return getattr(self, self.base_id_names[i])

    def reconstruct_weight(self, i):
        sids = self.get_spline_cluster_ids(i)
        bids = self.get_base_cluster_ids(i)
        poly = self.spline_codebooks[i][sids]
        base = self.base_codebooks[i][bids].squeeze(-1)
        return torch.cat([poly, base.unsqueeze(-1)], dim=-1)

    def forward(self, x):
        x = self.conv1.forward_from_weight(x, self.reconstruct_weight(0))
        x = self.conv2.forward_from_weight(x, self.reconstruct_weight(1))
        x = x.mean(dim=(2, 3))
        return self.head_layer.forward_from_weight(
            x, self.reconstruct_weight(2))
