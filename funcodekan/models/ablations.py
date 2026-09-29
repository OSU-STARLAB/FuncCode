import math
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .variants import DirectKANVariant
from ..compression.cross_variant import (
    cluster_branch_aware,
    symmetric_quantize_tensor,
    required_index_bits,
    bits_to_kib,
)


def _kmeans_1d(values: np.ndarray, k: int, seed: int):
    values = values.reshape(-1, 1).astype(np.float32)
    k = min(k, values.shape[0])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(values)
    centers = km.cluster_centers_.astype(np.float32).reshape(-1)
    return centers, labels


def _make_dense_like_from_layers(variant: str, layers):
    dense_like = DirectKANVariant(
        variant=variant,
        input_dim=layers[0].in_features,
        hidden_width=layers[0].out_features,
        output_dim=layers[-1].out_features,
    )
    dense_like.layers = layers
    return dense_like


class IndexEfficientBranchKAN(nn.Module):
    compression_type = "index"

    def __init__(
        self,
        dense_model: DirectKANVariant,
        spline_codebooks: List[torch.Tensor],
        spline_ids: List[torch.Tensor],
        conditional_base: List[torch.Tensor],
        train_codebooks: bool = True,
    ):
        super().__init__()
        self.variant = dense_model.variant
        self.layers = dense_model.layers
        self.spline_codebooks = nn.ParameterList([
            nn.Parameter(cb.clone().float(), requires_grad=train_codebooks)
            for cb in spline_codebooks
        ])
        self.conditional_base = nn.ParameterList([
            nn.Parameter(cb.clone().float().view(-1), requires_grad=train_codebooks)
            for cb in conditional_base
        ])

        self.spline_id_names = []
        for idx, ids in enumerate(spline_ids):
            self.register_buffer(f"spline_ids_{idx}", ids.clone().long())
            self.spline_id_names.append(f"spline_ids_{idx}")

    def get_spline_ids(self, idx):
        return getattr(self, self.spline_id_names[idx])

    def reconstruct_weight(self, idx):
        device = self.spline_codebooks[idx].device
        ids = self.get_spline_ids(idx).to(device)
        basis = self.spline_codebooks[idx][ids]
        base = self.conditional_base[idx][ids]
        return torch.cat([basis, base.unsqueeze(-1)], dim=-1)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer.forward_from_weight(x, self.reconstruct_weight(idx))
        return x


class SparseResidualBranchKAN(nn.Module):
    compression_type = "srb"

    def __init__(
        self,
        dense_model: DirectKANVariant,
        spline_codebooks: List[torch.Tensor],
        spline_ids: List[torch.Tensor],
        conditional_base: List[torch.Tensor],
        residual_codebooks: List[torch.Tensor],
        residual_ids: List[torch.Tensor],
        residual_masks: List[torch.Tensor],
        train_codebooks: bool = True,
    ):
        super().__init__()
        self.variant = dense_model.variant
        self.layers = dense_model.layers

        self.spline_codebooks = nn.ParameterList([
            nn.Parameter(cb.clone().float(), requires_grad=train_codebooks)
            for cb in spline_codebooks
        ])
        self.conditional_base = nn.ParameterList([
            nn.Parameter(cb.clone().float().view(-1), requires_grad=train_codebooks)
            for cb in conditional_base
        ])
        self.residual_codebooks = nn.ParameterList([
            nn.Parameter(cb.clone().float().view(-1), requires_grad=train_codebooks)
            for cb in residual_codebooks
        ])

        self.spline_id_names, self.residual_id_names, self.residual_mask_names = [], [], []
        for idx, ids in enumerate(spline_ids):
            self.register_buffer(f"spline_ids_{idx}", ids.clone().long())
            self.spline_id_names.append(f"spline_ids_{idx}")
        for idx, ids in enumerate(residual_ids):
            self.register_buffer(f"residual_ids_{idx}", ids.clone().long())
            self.residual_id_names.append(f"residual_ids_{idx}")
        for idx, mask in enumerate(residual_masks):
            self.register_buffer(f"residual_mask_{idx}", mask.clone().bool())
            self.residual_mask_names.append(f"residual_mask_{idx}")

    def get_spline_ids(self, idx):
        return getattr(self, self.spline_id_names[idx])

    def get_residual_ids(self, idx):
        return getattr(self, self.residual_id_names[idx])

    def get_residual_mask(self, idx):
        return getattr(self, self.residual_mask_names[idx])

    def reconstruct_weight(self, idx):
        device = self.spline_codebooks[idx].device
        sids = self.get_spline_ids(idx).to(device)
        rids = self.get_residual_ids(idx).to(device)
        mask = self.get_residual_mask(idx).to(device)

        basis = self.spline_codebooks[idx][sids]
        base = self.conditional_base[idx][sids]

        safe_rids = torch.clamp(rids, min=0)
        residual = self.residual_codebooks[idx][safe_rids]
        base = base + residual * mask.to(base.dtype)

        return torch.cat([basis, base.unsqueeze(-1)], dim=-1)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer.forward_from_weight(x, self.reconstruct_weight(idx))
        return x


class SoftToHardBranchIndexKAN(nn.Module):
    compression_type = "soft_index"

    def __init__(
        self,
        dense_model: DirectKANVariant,
        spline_codebooks: List[torch.Tensor],
        initial_spline_ids: List[torch.Tensor],
        conditional_base: List[torch.Tensor],
        tau: float = 1.0,
    ):
        super().__init__()
        self.variant = dense_model.variant
        self.layers = dense_model.layers
        self.tau = tau

        self.spline_codebooks = nn.ParameterList([
            nn.Parameter(cb.clone().float(), requires_grad=True)
            for cb in spline_codebooks
        ])
        self.conditional_base = nn.ParameterList([
            nn.Parameter(cb.clone().float().view(-1), requires_grad=True)
            for cb in conditional_base
        ])

        self.logits = nn.ParameterList()
        for ids, cb in zip(initial_spline_ids, spline_codebooks):
            k = cb.shape[0]
            logits = torch.full((*ids.shape, k), -4.0)
            logits.scatter_(-1, ids.unsqueeze(-1), 4.0)
            self.logits.append(nn.Parameter(logits.float(), requires_grad=True))

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def reconstruct_weight(self, idx, hard: bool = False):
        logits = self.logits[idx]
        cb = self.spline_codebooks[idx]
        base_cb = self.conditional_base[idx]

        if hard:
            ids = torch.argmax(logits, dim=-1)
            basis = cb[ids]
            base = base_cb[ids]
        else:
            alpha = torch.softmax(logits / max(self.tau, 1e-6), dim=-1)
            basis = torch.einsum("oik,kd->oid", alpha, cb)
            base = torch.einsum("oik,k->oi", alpha, base_cb)

        return torch.cat([basis, base.unsqueeze(-1)], dim=-1)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer.forward_from_weight(x, self.reconstruct_weight(idx, hard=False))
        return x

    def forward_hard(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer.forward_from_weight(x, self.reconstruct_weight(idx, hard=True))
        return x

    def harden(self):
        ids_all = [torch.argmax(logits.detach().cpu(), dim=-1).long() for logits in self.logits]
        cbs = [cb.detach().cpu() for cb in self.spline_codebooks]
        cond = [cb.detach().cpu() for cb in self.conditional_base]
        dense_like = _make_dense_like_from_layers(self.variant, self.layers)
        return IndexEfficientBranchKAN(dense_like, cbs, ids_all, cond, train_codebooks=True)


def build_index_efficient_from_dense(
    dense_model: DirectKANVariant,
    clusters: int,
    seed: int,
    function_samples: int = 128,
):
    scb, sid, _, _ = cluster_branch_aware(
        dense_model=dense_model,
        spline_clusters=clusters,
        base_clusters=max(2, clusters // 2),
        seed=seed,
        samples=function_samples,
        spline_method="function",
    )

    cond_base = []
    weights = dense_model.get_edge_weights()

    for w, ids, cb in zip(weights, sid, scb):
        k = cb.shape[0]
        base = w[..., -1].reshape(-1)
        flat_ids = ids.reshape(-1)
        vals = torch.zeros(k, dtype=torch.float32)
        for j in range(k):
            mask = flat_ids == j
            if mask.any():
                vals[j] = base[mask].mean()
            else:
                vals[j] = 0.0
        cond_base.append(vals)

    return IndexEfficientBranchKAN(dense_model, scb, sid, cond_base, train_codebooks=True)


def build_srb_from_dense(
    dense_model: DirectKANVariant,
    clusters: int,
    residual_fraction: float,
    residual_clusters: int,
    seed: int,
    function_samples: int = 128,
):
    scb, sid, _, _ = cluster_branch_aware(
        dense_model=dense_model,
        spline_clusters=clusters,
        base_clusters=max(2, clusters // 2),
        seed=seed,
        samples=function_samples,
        spline_method="function",
    )

    cond_base = []
    residual_codebooks = []
    residual_ids_all = []
    residual_masks = []

    weights = dense_model.get_edge_weights()

    for layer_idx, (w, ids, cb) in enumerate(zip(weights, sid, scb)):
        k = cb.shape[0]
        out_features, in_features, _ = w.shape
        base = w[..., -1]
        flat_base = base.reshape(-1)
        flat_ids = ids.reshape(-1)

        vals = torch.zeros(k, dtype=torch.float32)
        for j in range(k):
            mask = flat_ids == j
            vals[j] = flat_base[mask].mean() if mask.any() else 0.0
        cond_base.append(vals)

        pred = vals[flat_ids]
        residual = flat_base - pred

        n_edges = residual.numel()
        n_select = max(1, int(round(float(residual_fraction) * n_edges)))
        top_idx = torch.topk(residual.abs(), k=n_select, largest=True).indices

        mask_flat = torch.zeros(n_edges, dtype=torch.bool)
        mask_flat[top_idx] = True

        selected_residual = residual[top_idx].numpy().reshape(-1, 1).astype(np.float32)
        centers, labels = _kmeans_1d(selected_residual, residual_clusters, seed + layer_idx)

        rid_flat = torch.full((n_edges,), -1, dtype=torch.long)
        rid_flat[top_idx] = torch.tensor(labels, dtype=torch.long)

        residual_codebooks.append(torch.tensor(centers, dtype=torch.float32))
        residual_ids_all.append(rid_flat.reshape(out_features, in_features))
        residual_masks.append(mask_flat.reshape(out_features, in_features))

    return SparseResidualBranchKAN(
        dense_model,
        scb,
        sid,
        cond_base,
        residual_codebooks,
        residual_ids_all,
        residual_masks,
        train_codebooks=True,
    )


def build_soft_to_hard_from_dense(
    dense_model: DirectKANVariant,
    clusters: int,
    seed: int,
    function_samples: int = 128,
    tau: float = 1.0,
):
    index_model = build_index_efficient_from_dense(
        dense_model=dense_model,
        clusters=clusters,
        seed=seed,
        function_samples=function_samples,
    )
    cbs = [cb.detach().cpu() for cb in index_model.spline_codebooks]
    ids = [index_model.get_spline_ids(i).detach().cpu() for i in range(len(cbs))]
    cond = [cb.detach().cpu() for cb in index_model.conditional_base]
    return SoftToHardBranchIndexKAN(dense_model, cbs, ids, cond, tau=tau)


def quantize_variant_ablation_model(model, bits: int):
    ctype = getattr(model, "compression_type", "")

    dense_like = _make_dense_like_from_layers(model.variant, model.layers)

    if ctype == "index":
        cbs = [symmetric_quantize_tensor(cb.detach().cpu(), bits) for cb in model.spline_codebooks]
        cond = [symmetric_quantize_tensor(cb.detach().cpu(), bits, per_vector=False) for cb in model.conditional_base]
        ids = [model.get_spline_ids(i).detach().cpu() for i in range(len(cbs))]
        return IndexEfficientBranchKAN(dense_like, cbs, ids, cond, train_codebooks=False)

    if ctype == "srb":
        cbs = [symmetric_quantize_tensor(cb.detach().cpu(), bits) for cb in model.spline_codebooks]
        cond = [symmetric_quantize_tensor(cb.detach().cpu(), bits, per_vector=False) for cb in model.conditional_base]
        rcb = [symmetric_quantize_tensor(cb.detach().cpu(), bits, per_vector=False) for cb in model.residual_codebooks]
        sid = [model.get_spline_ids(i).detach().cpu() for i in range(len(cbs))]
        rid = [model.get_residual_ids(i).detach().cpu() for i in range(len(rcb))]
        msk = [model.get_residual_mask(i).detach().cpu() for i in range(len(rcb))]
        return SparseResidualBranchKAN(dense_like, cbs, sid, cond, rcb, rid, msk, train_codebooks=False)

    if ctype == "soft_index":
        hard = model.harden()
        return quantize_variant_ablation_model(hard, bits=bits)

    raise ValueError(f"Unsupported model compression_type for quantization: {ctype}")


def variant_ablation_storage_breakdown(model, codebook_bits: int = 32, scale_bits_per_value: int = 0) -> Dict[str, float]:
    ctype = getattr(model, "compression_type", "")
    out = {
        "storage_total_bits": 0,
        "storage_total_kib": 0.0,
        "spline_codebook_bits": 0,
        "spline_index_bits": 0,
        "conditional_base_bits": 0,
        "residual_codebook_bits": 0,
        "residual_index_bits": 0,
        "residual_mask_bits": 0,
        "scale_bits": 0,
    }

    if ctype in ["index", "soft_index"]:
        # If soft, report deployable hard format.
        if ctype == "soft_index":
            model = model.harden()
        for idx, cb in enumerate(model.spline_codebooks):
            ids = model.get_spline_ids(idx)
            k = cb.shape[0]
            out["spline_codebook_bits"] += int(cb.numel() * codebook_bits)
            out["spline_index_bits"] += int(ids.numel() * required_index_bits(k))
            out["conditional_base_bits"] += int(model.conditional_base[idx].numel() * codebook_bits)
            out["scale_bits"] += int((k + model.conditional_base[idx].numel()) * scale_bits_per_value)

    elif ctype == "srb":
        for idx, cb in enumerate(model.spline_codebooks):
            ids = model.get_spline_ids(idx)
            k = cb.shape[0]
            out["spline_codebook_bits"] += int(cb.numel() * codebook_bits)
            out["spline_index_bits"] += int(ids.numel() * required_index_bits(k))
            out["conditional_base_bits"] += int(model.conditional_base[idx].numel() * codebook_bits)

            rcb = model.residual_codebooks[idx]
            rid = model.get_residual_ids(idx)
            mask = model.get_residual_mask(idx)
            rk = rcb.numel()

            out["residual_codebook_bits"] += int(rk * codebook_bits)
            out["residual_index_bits"] += int(mask.sum().item() * required_index_bits(max(rk, 1)))
            # Store 1 bit mask per edge. This is simple and honest; later we can use sparse coordinates/RLE.
            out["residual_mask_bits"] += int(mask.numel())
            out["scale_bits"] += int((k + model.conditional_base[idx].numel() + rk) * scale_bits_per_value)

    else:
        raise ValueError(f"Unsupported compression_type for storage breakdown: {ctype}")

    total = sum(v for k, v in out.items() if k.endswith("_bits"))
    out["storage_total_bits"] = int(total)
    out["storage_total_kib"] = bits_to_kib(total)
    return out
