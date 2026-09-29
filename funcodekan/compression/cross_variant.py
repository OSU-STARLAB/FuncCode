import math
from typing import List, Tuple, Dict

import numpy as np
import torch
from sklearn.cluster import KMeans

from ..models.variants import DirectKANVariant, SharedCodebookKAN, BranchCodebookKAN, MLPBaseline


def required_index_bits(k: int) -> int:
    if k <= 1:
        return 1
    return int(math.ceil(math.log2(k)))


def bits_to_kib(bits: float) -> float:
    return float(bits) / 8.0 / 1024.0


def kmeans_fit(vectors: np.ndarray, k: int, seed: int):
    k = min(k, vectors.shape[0])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(vectors)
    centers = km.cluster_centers_.astype(np.float32)
    return centers, labels


def make_grid_domain(samples=128, lo=-2.5, hi=2.5, device="cpu"):
    return torch.linspace(lo, hi, samples, device=device)


def cluster_coefficient_space(dense_model: DirectKANVariant, clusters: int, seed: int):
    codebooks, ids_all = [], []
    for w in dense_model.get_edge_weights():
        out_features, in_features, coeff_dim = w.shape
        vectors = w.reshape(-1, coeff_dim).numpy()
        centers, labels = kmeans_fit(vectors, clusters, seed)
        codebooks.append(torch.tensor(centers, dtype=torch.float32))
        ids_all.append(torch.tensor(labels, dtype=torch.long).reshape(out_features, in_features))
    return codebooks, ids_all


def cluster_function_space(dense_model: DirectKANVariant, clusters: int, seed: int, samples=128, include_base=True, normalize=True):
    dense_cpu = dense_model.cpu()
    dense_cpu.eval()
    x_domain = make_grid_domain(samples=samples, device="cpu")

    codebooks, ids_all = [], []
    for layer, w in zip(dense_cpu.layers, dense_cpu.get_edge_weights()):
        out_features, in_features, coeff_dim = w.shape
        coeff_vectors = w.reshape(-1, coeff_dim).numpy()
        signatures = layer.edge_function_signatures(w, x_domain, include_base=include_base, normalize=normalize).numpy()

        k = min(clusters, signatures.shape[0])
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(signatures)

        centers = np.zeros((k, coeff_dim), dtype=np.float32)
        rng = np.random.RandomState(seed)
        for j in range(k):
            mask = labels == j
            centers[j] = coeff_vectors[mask].mean(axis=0) if mask.any() else coeff_vectors[rng.randint(0, coeff_vectors.shape[0])]

        codebooks.append(torch.tensor(centers, dtype=torch.float32))
        ids_all.append(torch.tensor(labels, dtype=torch.long).reshape(out_features, in_features))

    return codebooks, ids_all


def cluster_branch_aware(
    dense_model: DirectKANVariant,
    spline_clusters: int,
    base_clusters: int,
    seed: int,
    samples=128,
    spline_method="function",
    normalize=True,
):
    dense_cpu = dense_model.cpu()
    dense_cpu.eval()
    x_domain = make_grid_domain(samples=samples, device="cpu")

    spline_codebooks, spline_ids_all = [], []
    base_codebooks, base_ids_all = [], []

    for layer, w in zip(dense_cpu.layers, dense_cpu.get_edge_weights()):
        out_features, in_features, coeff_dim = w.shape
        spline_dim = coeff_dim - 1

        spline_vectors = w[..., :-1].reshape(-1, spline_dim).numpy()
        base_vectors = w[..., -1].reshape(-1, 1).numpy()

        if spline_method == "function":
            signatures = layer.edge_function_signatures(w, x_domain, include_base=False, normalize=normalize).numpy()
            k = min(spline_clusters, signatures.shape[0])
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(signatures)

            centers = np.zeros((k, spline_dim), dtype=np.float32)
            rng = np.random.RandomState(seed)
            for j in range(k):
                mask = labels == j
                centers[j] = spline_vectors[mask].mean(axis=0) if mask.any() else spline_vectors[rng.randint(0, spline_vectors.shape[0])]
            spline_centers, spline_labels = centers, labels
        else:
            spline_centers, spline_labels = kmeans_fit(spline_vectors, spline_clusters, seed)

        base_centers, base_labels = kmeans_fit(base_vectors, base_clusters, seed)

        spline_codebooks.append(torch.tensor(spline_centers, dtype=torch.float32))
        spline_ids_all.append(torch.tensor(spline_labels, dtype=torch.long).reshape(out_features, in_features))
        base_codebooks.append(torch.tensor(base_centers, dtype=torch.float32))
        base_ids_all.append(torch.tensor(base_labels, dtype=torch.long).reshape(out_features, in_features))

    return spline_codebooks, spline_ids_all, base_codebooks, base_ids_all


def build_compressed_kan(
    dense_model: DirectKANVariant,
    method: str,
    clusters: int,
    seed: int,
    function_samples: int = 128,
    base_clusters: int = None,
    train_codebooks: bool = True,
):
    method = method.lower()

    if method == "coefficient":
        cbs, ids = cluster_coefficient_space(dense_model, clusters=clusters, seed=seed)
        return SharedCodebookKAN(dense_model, cbs, ids, train_codebooks=train_codebooks)

    if method == "function":
        cbs, ids = cluster_function_space(dense_model, clusters=clusters, seed=seed, samples=function_samples, include_base=True)
        return SharedCodebookKAN(dense_model, cbs, ids, train_codebooks=train_codebooks)

    if method == "branch":
        if base_clusters is None:
            base_clusters = max(2, clusters // 2)
        scb, sid, bcb, bid = cluster_branch_aware(
            dense_model,
            spline_clusters=clusters,
            base_clusters=base_clusters,
            seed=seed,
            samples=function_samples,
            spline_method="function",
        )
        return BranchCodebookKAN(dense_model, scb, sid, bcb, bid, train_codebooks=train_codebooks)

    raise ValueError(f"Unknown compression method for KAN: {method}")


def symmetric_quantize_tensor(x: torch.Tensor, bits: int, per_vector: bool = True):
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)

    if x.dim() >= 2 and per_vector:
        max_abs = x.abs().reshape(x.shape[0], -1).amax(dim=1, keepdim=True)
        view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
        scale = torch.clamp(max_abs.view(*view_shape) / qmax, min=1e-8)
    else:
        scale = torch.clamp(x.abs().amax() / qmax, min=1e-8)

    q = torch.round(x / scale).clamp(qmin, qmax)
    return q * scale


def quantize_compressed_kan(model, bits: int):
    ctype = getattr(model, "compression_type", "shared")

    if ctype == "branch":
        dense_like = DirectKANVariant(
            variant=model.variant,
            input_dim=model.layers[0].in_features,
            hidden_width=model.layers[0].out_features,
            output_dim=model.layers[-1].out_features,
        )
        dense_like.layers = model.layers

        scb = [symmetric_quantize_tensor(cb.detach().cpu(), bits) for cb in model.spline_codebooks]
        bcb = [symmetric_quantize_tensor(cb.detach().cpu(), bits) for cb in model.base_codebooks]
        sid = [model.get_spline_cluster_ids(i).detach().cpu() for i in range(len(scb))]
        bid = [model.get_base_cluster_ids(i).detach().cpu() for i in range(len(bcb))]
        return BranchCodebookKAN(dense_like, scb, sid, bcb, bid, train_codebooks=False)

    dense_like = DirectKANVariant(
        variant=model.variant,
        input_dim=model.layers[0].in_features,
        hidden_width=model.layers[0].out_features,
        output_dim=model.layers[-1].out_features,
    )
    dense_like.layers = model.layers

    cbs = [symmetric_quantize_tensor(cb.detach().cpu(), bits) for cb in model.codebooks]
    ids = [model.get_cluster_ids(i).detach().cpu() for i in range(len(cbs))]
    return SharedCodebookKAN(dense_like, cbs, ids, train_codebooks=False)


def quantize_mlp_copy(model: MLPBaseline, bits: int):
    import copy
    qmodel = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for p in qmodel.parameters():
            p.copy_(symmetric_quantize_tensor(p.data, bits, per_vector=False))
    return qmodel


def dense_bits(model) -> int:
    if hasattr(model, "dense_storage_bits"):
        return int(model.dense_storage_bits())
    return int(sum(p.numel() * 32 for p in model.parameters()))


def compressed_storage_breakdown(model, codebook_bits: int = 32, scale_bits_per_value: int = 0) -> Dict[str, float]:
    ctype = getattr(model, "compression_type", "dense")
    out = {
        "shared_codebook_bits": 0,
        "shared_index_bits": 0,
        "spline_codebook_bits": 0,
        "spline_index_bits": 0,
        "base_codebook_bits": 0,
        "base_index_bits": 0,
        "scale_bits": 0,
        "storage_total_bits": 0,
        "storage_total_kib": 0.0,
    }

    if ctype == "shared":
        for idx, cb in enumerate(model.codebooks):
            ids = model.get_cluster_ids(idx)
            k = cb.shape[0]
            out["shared_codebook_bits"] += int(cb.numel() * codebook_bits)
            out["shared_index_bits"] += int(ids.numel() * required_index_bits(k))
            out["scale_bits"] += int(k * scale_bits_per_value)

    elif ctype == "branch":
        for idx, cb in enumerate(model.spline_codebooks):
            ids = model.get_spline_cluster_ids(idx)
            k = cb.shape[0]
            out["spline_codebook_bits"] += int(cb.numel() * codebook_bits)
            out["spline_index_bits"] += int(ids.numel() * required_index_bits(k))
            out["scale_bits"] += int(k * scale_bits_per_value)

        for idx, cb in enumerate(model.base_codebooks):
            ids = model.get_base_cluster_ids(idx)
            k = cb.shape[0]
            out["base_codebook_bits"] += int(cb.numel() * codebook_bits)
            out["base_index_bits"] += int(ids.numel() * required_index_bits(k))
            out["scale_bits"] += int(k * scale_bits_per_value)

    else:
        out["storage_total_bits"] = dense_bits(model)
        out["storage_total_kib"] = bits_to_kib(out["storage_total_bits"])
        return out

    total = (
        out["shared_codebook_bits"] + out["shared_index_bits"]
        + out["spline_codebook_bits"] + out["spline_index_bits"]
        + out["base_codebook_bits"] + out["base_index_bits"]
        + out["scale_bits"]
    )
    out["storage_total_bits"] = int(total)
    out["storage_total_kib"] = bits_to_kib(total)
    return out
