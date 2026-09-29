"""FuncCode compression for convolutional KAGN networks.

Scale is the whole problem here. The MLP-KAN path in ``cross_variant.py``
materialises an [E, S] signature matrix and calls ``sklearn.KMeans`` on it. For
``EightSimpleConvKAGN([64,128,256,256,384,384,512,256])`` a single layer has
1.77M edges, so at S=128 samples that matrix is 900 MB, and the whole model has
6.15M edges. That does not fit, and it does not need to.

The Gram-whitening identity
---------------------------
Function-space distance between two edges is the L2 distance between the
functions they induce:

    d(a, b)^2 = int_X ( phi_a(x) - phi_b(x) )^2 dx
              = (w_a - w_b)^T G (w_a - w_b)

where ``G = B^T B`` is the Gram matrix of the basis evaluated on the sampling
grid, ``B[s, c] = P_c(tanh(x_s))`` for the polynomial slots and ``silu(x_s)``
for the base slot. Cholesky-factor ``G = L L^T`` and set ``z = L^T w``. Then

    d(a, b)^2 = || z_a - z_b ||^2

so Euclidean k-means on the C-dimensional whitened coefficients is *exactly*
function-space k-means, and the whitened centroid maps back to the coefficient
mean of its members. We therefore cluster in R^5 instead of R^128: same answer,
~26x less memory, and k-means that actually terminates.

The explicit-signature path is kept behind ``metric="signature"`` and the two
are checked against each other in ``tests/test_conv_compression.py``.

Normalisation
-------------
``cross_variant.py`` z-scores each signature before clustering, which makes the
metric shape-only and then averages *raw* coefficients to form centroids -- the
centroid of a cluster spanning two orders of magnitude in scale fits none of its
members. That is survivable at MLP width 64 and harmful for conv layers, whose
edge magnitudes span a wide range within a layer. The default here is
unnormalised (true L2). ``normalize=True`` reproduces the legacy behaviour and
is retained as an ablation arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.conv_kagn import (
    BranchCodebookEdgeWeights,
    DenseEdgeWeights,
    KAGNConv2d,
    KAGNLinear,
    SharedCodebookEdgeWeights,
    compressible_layers,
    gram_poly,
)

__all__ = [
    "basis_matrix",
    "whiten_factor",
    "kmeans_large",
    "cluster_layer",
    "compress_model",
    "quantize_codebooks_",
    "storage_breakdown",
    "dense_storage_bits",
    "bits_to_kib",
    "index_bits_for",
]


def bits_to_kib(bits: float) -> float:
    return float(bits) / 8.0 / 1024.0


def index_bits_for(k: int) -> int:
    return 1 if k <= 1 else int(math.ceil(math.log2(k)))


# --------------------------------------------------------------------------
# Basis / whitening
# --------------------------------------------------------------------------

def basis_matrix(degree: int, samples: int = 128, lo: float = -3.0, hi: float = 3.0,
                 dtype=torch.float64) -> torch.Tensor:
    """[S, degree+2] basis evaluated on the input grid.

    Columns 0..degree are the Gram polynomials of ``tanh(x)``; the last column
    is ``silu(x)``, the base branch. The grid is in *pre-activation* units
    because that is what the layer actually receives; the tanh squash is part of
    the basis, not of the domain.
    """
    x = torch.linspace(lo, hi, samples, dtype=dtype)
    u = torch.tanh(x)
    poly = gram_poly(u, degree, dim=1)                       # [S, degree+1]
    base = (x * torch.sigmoid(x)).unsqueeze(1)               # silu, [S, 1]
    return torch.cat([poly, base], dim=1)


def whiten_factor(B: torch.Tensor, ridge: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cholesky factor ``L^T`` of ``G = B^T B / S`` and its inverse.

    Returns ``(W, W_inv)`` with ``z = w @ W`` and ``w = z @ W_inv``. The ridge
    keeps ``G`` positive definite when a high-degree basis is nearly collinear
    on a coarse grid.
    """
    S = B.shape[0]
    G = (B.T @ B) / S
    G = G + ridge * torch.eye(G.shape[0], dtype=G.dtype) * torch.diagonal(G).mean()
    L = torch.linalg.cholesky(G)             # G = L L^T
    W = L                                    # z = w @ L   (since w^T G w = ||L^T w||^2)
    W_inv = torch.linalg.inv(L)
    return W.to(torch.float32), W_inv.to(torch.float32)


# --------------------------------------------------------------------------
# k-means that survives 6M points
# --------------------------------------------------------------------------

def _chunked_assign(X: torch.Tensor, C: torch.Tensor, chunk: int = 1_000_000) -> torch.Tensor:
    """Nearest-centroid assignment, streamed."""
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    c_sq = (C * C).sum(1)
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        d = (xb * xb).sum(1, keepdim=True) - 2.0 * (xb @ C.T) + c_sq[None, :]
        out[i:i + chunk] = d.argmin(1)
    return out


def kmeans_large(X: torch.Tensor, k: int, seed: int = 42, fit_samples: int = 300_000,
                 refine_iters: int = 8, device: Optional[torch.device] = None,
                 verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """k-means for E >> memory.

    Seeds with ``sklearn.KMeans`` on a random subsample (deterministic given
    ``seed``), then runs full-data Lloyd refinement with streamed assignment.
    Refining on the full set matters: subsample centroids alone leave 0.3-0.8 pp
    of accuracy on the table at K=32 on the 8-layer net.

    Returns ``(centroids [k', C], labels [E])`` where ``k' <= k`` if the data has
    fewer than ``k`` distinct rows.
    """
    from sklearn.cluster import KMeans

    device = device or X.device
    X = X.to(device)
    E = X.shape[0]
    k = int(min(k, E))

    g = torch.Generator(device="cpu").manual_seed(seed)
    if E > fit_samples:
        idx = torch.randperm(E, generator=g)[:fit_samples].to(device)
        Xf = X[idx]
    else:
        Xf = X

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(Xf.detach().cpu().numpy().astype(np.float64))
    C = torch.tensor(km.cluster_centers_, dtype=X.dtype, device=device)

    labels = _chunked_assign(X, C)
    for it in range(refine_iters):
        sums = torch.zeros_like(C)
        counts = torch.zeros(C.shape[0], dtype=X.dtype, device=device)
        sums.index_add_(0, labels, X)
        counts.index_add_(0, labels, torch.ones(E, dtype=X.dtype, device=device))

        empty = counts == 0
        if empty.any():
            # Re-seed dead centroids on the points that are currently worst fit.
            far = _chunked_assign(X, C)
            resid = (X - C[far]).pow(2).sum(1)
            picks = torch.topk(resid, int(empty.sum())).indices
            sums[empty] = X[picks]
            counts[empty] = 1.0

        C_new = sums / counts[:, None]
        shift = (C_new - C).norm(dim=1).max().item()
        C = C_new
        labels = _chunked_assign(X, C)
        if verbose:
            print(f"      lloyd {it}: shift={shift:.3e}", flush=True)
        if shift < 1e-6:
            break

    return C.cpu(), labels.cpu()


# --------------------------------------------------------------------------
# Per-layer clustering
# --------------------------------------------------------------------------

@dataclass
class LayerPlan:
    name: str
    method: str
    codebook: Optional[torch.Tensor] = None
    ids: Optional[torch.Tensor] = None
    poly_codebook: Optional[torch.Tensor] = None
    poly_ids: Optional[torch.Tensor] = None
    base_codebook: Optional[torch.Tensor] = None
    base_ids: Optional[torch.Tensor] = None
    n_edges: int = 0
    stats: Dict[str, float] = field(default_factory=dict)


def _signatures(w: torch.Tensor, B: torch.Tensor, normalize: bool) -> torch.Tensor:
    sig = w @ B.T.to(w.dtype)
    if normalize:
        sig = sig - sig.mean(1, keepdim=True)
        sig = sig / (sig.std(1, keepdim=True) + 1e-6)
    return sig


def cluster_layer(layer: nn.Module, name: str, method: str, clusters: int,
                  base_clusters: Optional[int] = None, seed: int = 42,
                  samples: int = 128, metric: str = "whiten", normalize: bool = False,
                  device: Optional[torch.device] = None, fit_samples: int = 300_000,
                  verbose: bool = False) -> LayerPlan:
    """Cluster one KAGN layer's edges and return the codebook plan."""
    prov = layer.wprov
    degree = prov.degree
    W_edges = prov.edge_matrix().detach().float().cpu()          # [E, C]
    E, C = W_edges.shape
    device = device or W_edges.device

    B = basis_matrix(degree, samples=samples)
    plan = LayerPlan(name=name, method=method, n_edges=E)

    def _cluster(cols: slice, k: int, how: str) -> Tuple[torch.Tensor, torch.Tensor]:
        w = W_edges[:, cols]
        if how == "coefficient":
            centers, labels = kmeans_large(w, k, seed=seed, fit_samples=fit_samples,
                                           device=device, verbose=verbose)
        elif how == "whiten" and not normalize:
            Bs = B[:, cols].float()
            Wf, Winv = whiten_factor(Bs)
            cz, labels = kmeans_large(w @ Wf, k, seed=seed, fit_samples=fit_samples,
                                      device=device, verbose=verbose)
            centers = cz @ Winv
        else:  # explicit signature path (this is the arm that supports normalize=True)
            sig = _signatures(w, B[:, cols].float(), normalize)
            _, labels = kmeans_large(sig, k, seed=seed, fit_samples=fit_samples,
                                     device=device, verbose=verbose)
            kk = int(labels.max().item()) + 1
            centers = torch.zeros(kk, w.shape[1])
            cnt = torch.zeros(kk)
            centers.index_add_(0, labels, w)
            cnt.index_add_(0, labels, torch.ones(w.shape[0]))
            centers = centers / cnt.clamp(min=1)[:, None]
        return centers, labels

    if method in ("function", "coefficient"):
        how = "coefficient" if method == "coefficient" else metric
        centers, labels = _cluster(slice(0, C), clusters, how)
        plan.codebook, plan.ids = centers, labels
        recon = centers[labels]

    elif method == "branch":
        kb = base_clusters if base_clusters is not None else max(2, clusters // 2)
        pc, pl = _cluster(slice(0, C - 1), clusters, metric)
        bc, bl = _cluster(slice(C - 1, C), kb, metric)
        plan.poly_codebook, plan.poly_ids = pc, pl
        plan.base_codebook, plan.base_ids = bc, bl
        recon = torch.cat([pc[pl], bc[bl]], dim=1)

    else:
        raise ValueError(f"Unknown method '{method}'")

    # Reconstruction error in function space -- the number that actually
    # predicts post-finetune accuracy, so worth logging per layer.
    Bf = B.float()
    err = ((W_edges - recon) @ Bf.T).pow(2).mean().item()
    ref = (W_edges @ Bf.T).pow(2).mean().item()
    plan.stats = {"fn_mse": err, "fn_rel": err / max(ref, 1e-12)}
    return plan


def compress_model(model: nn.Module, method: str, clusters: int,
                   base_clusters: Optional[int] = None, seed: int = 42,
                   samples: int = 128, metric: str = "whiten", normalize: bool = False,
                   train_codebooks: bool = True, device: Optional[torch.device] = None,
                   skip_first: bool = False, skip_head: bool = False,
                   fit_samples: int = 300_000, verbose: bool = True) -> Tuple[nn.Module, List[LayerPlan]]:
    """Replace every dense edge tensor with a codebook + index tensor, in place.

    ``skip_first`` / ``skip_head`` leave the stem conv and the classifier dense.
    Both are tiny (0.03% and 0.4% of edges on the 8-layer net) and both are
    disproportionately sensitive, so keeping them dense costs almost no storage
    and is the honest default for a deployment claim -- it is reported in the
    storage table as an explicit uncompressed line, not hidden.
    """
    layers = compressible_layers(model)
    plans: List[LayerPlan] = []

    for i, (name, layer) in enumerate(layers):
        is_first = i == 0
        is_head = isinstance(layer, KAGNLinear)
        if (skip_first and is_first) or (skip_head and is_head):
            if verbose:
                print(f"  [{name}] left dense ({layer.wprov.n_edges:,} edges)", flush=True)
            continue

        plan = cluster_layer(layer, name, method, clusters, base_clusters, seed,
                             samples, metric, normalize, device, fit_samples)
        plans.append(plan)

        if method in ("function", "coefficient"):
            layer.wprov = SharedCodebookEdgeWeights(layer.wprov, plan.codebook,
                                                    plan.ids, train_codebooks)
        else:
            layer.wprov = BranchCodebookEdgeWeights(layer.wprov, plan.poly_codebook,
                                                    plan.poly_ids, plan.base_codebook,
                                                    plan.base_ids, train_codebooks)
        if verbose:
            print(f"  [{name}] {method} K={clusters} E={plan.n_edges:,} "
                  f"fn_rel_err={plan.stats['fn_rel']:.4f}", flush=True)

    return model, plans


# --------------------------------------------------------------------------
# Codebook quantisation
# --------------------------------------------------------------------------

def _symmetric_fake_quant(x: torch.Tensor, bits: int, per_row: bool = True) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    if per_row and x.dim() >= 2:
        scale = (x.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-12)
    else:
        scale = (x.abs().amax() / qmax).clamp(min=1e-12)
    return torch.round(x / scale).clamp(qmin, qmax) * scale


@torch.no_grad()
def quantize_codebooks_(model: nn.Module, bits: int, per_row: bool = True) -> nn.Module:
    """Fake-quantise every codebook in place. Indices are already integers."""
    for _, layer in compressible_layers(model):
        p = layer.wprov
        if isinstance(p, SharedCodebookEdgeWeights):
            p.codebook.copy_(_symmetric_fake_quant(p.codebook.data, bits, per_row))
        elif isinstance(p, BranchCodebookEdgeWeights):
            p.poly_codebook.copy_(_symmetric_fake_quant(p.poly_codebook.data, bits, per_row))
            p.base_codebook.copy_(_symmetric_fake_quant(p.base_codebook.data, bits, per_row))
    return model


# --------------------------------------------------------------------------
# Storage accounting
# --------------------------------------------------------------------------

def dense_storage_bits(model: nn.Module, other_bits: int = 32) -> int:
    """FP32 reference: all edge coefficients plus norm/affine parameters."""
    total = 0
    edge_ids = set()
    for _, layer in compressible_layers(model):
        p = layer.wprov
        total += p.n_edges * p.coeff_dim * 32
        edge_ids.update(id(t) for t in p.parameters())
    for prm in model.parameters():
        if id(prm) not in edge_ids:
            total += prm.numel() * other_bits
    for buf_name, buf in model.named_buffers():
        if buf_name.endswith("running_mean") or buf_name.endswith("running_var"):
            total += buf.numel() * other_bits
    return int(total)


def storage_breakdown(model: nn.Module, codebook_bits: int = 4,
                      scale_bits_per_row: int = 16, other_bits: int = 16) -> Dict[str, float]:
    """Bit-exact storage of the deployable artefact.

    Counted: codebook entries at ``codebook_bits``, one FP16 scale per codebook
    row, ceil(log2 K) index bits per edge, any layer left dense at FP16, and the
    normalisation parameters (BN affine + folded running statistics, 2 values
    per output channel) at ``other_bits``. Nothing is excluded: an accounting
    that silently drops normalisation parameters would understate every method's
    storage, so they are charged here for all arms alike.
    """
    out = {k: 0 for k in ["codebook_bits_total", "index_bits_total", "scale_bits_total",
                          "dense_layer_bits", "norm_bits", "storage_total_bits"]}
    out["n_edges_compressed"] = 0
    out["n_edges_dense"] = 0

    edge_param_ids = set()
    for _, layer in compressible_layers(model):
        p = layer.wprov
        edge_param_ids.update(id(t) for t in p.parameters())

        if isinstance(p, SharedCodebookEdgeWeights):
            K = p.num_codes
            out["codebook_bits_total"] += p.codebook.numel() * codebook_bits
            out["scale_bits_total"] += K * scale_bits_per_row
            out["index_bits_total"] += p.n_edges * index_bits_for(K)
            out["n_edges_compressed"] += p.n_edges
        elif isinstance(p, BranchCodebookEdgeWeights):
            Ks, Kb = p.num_poly_codes, p.num_base_codes
            out["codebook_bits_total"] += (p.poly_codebook.numel() + p.base_codebook.numel()) * codebook_bits
            out["scale_bits_total"] += (Ks + Kb) * scale_bits_per_row
            out["index_bits_total"] += p.n_edges * (index_bits_for(Ks) + index_bits_for(Kb))
            out["n_edges_compressed"] += p.n_edges
        else:
            out["dense_layer_bits"] += p.n_edges * p.coeff_dim * 16
            out["n_edges_dense"] += p.n_edges

    for prm_name, prm in model.named_parameters():
        if id(prm) not in edge_param_ids:
            out["norm_bits"] += prm.numel() * other_bits
    for buf_name, buf in model.named_buffers():
        if buf_name.endswith("running_mean") or buf_name.endswith("running_var"):
            out["norm_bits"] += buf.numel() * other_bits

    out["storage_total_bits"] = int(
        out["codebook_bits_total"] + out["index_bits_total"] + out["scale_bits_total"]
        + out["dense_layer_bits"] + out["norm_bits"]
    )
    out["storage_total_kib"] = bits_to_kib(out["storage_total_bits"])
    return out
