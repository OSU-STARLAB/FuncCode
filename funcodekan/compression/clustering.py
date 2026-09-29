import numpy as np
import torch
from sklearn.cluster import KMeans
from ..models.spline import DenseSplineKAN, ClusteredSplineKAN, BranchAwareClusteredSplineKAN, IndexEfficientBranchSplineKAN, SparseResidualBranchSplineKAN
from .function_space import make_grid_domain, collect_layer_activation_domains, evaluate_spline_edge_functions

def _kmeans(vectors, k, seed):
    k = min(k, vectors.shape[0])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(vectors)
    return km.cluster_centers_.astype(np.float32), labels

def _domains_for_model(dense_model, domain, samples, train_loader, device, activation_batches):
    dense_cpu = dense_model.cpu(); dense_cpu.eval()
    if domain == "activation":
        if train_loader is None or device is None: raise ValueError("activation domain requires train_loader and device")
        domains = collect_layer_activation_domains(dense_model.to(device), train_loader, device, activation_batches, samples)
        dense_cpu = dense_model.cpu()
    elif domain == "grid":
        domains = [make_grid_domain(samples, -2.5, 2.5, "cpu") for _ in dense_cpu.layers]
    else:
        raise ValueError(f"Unknown function domain: {domain}")
    return dense_cpu, domains

def cluster_dense_weights_coefficient_space(edge_weights, num_clusters=16, seed=42):
    codebooks, cluster_ids = [], []
    for w in edge_weights:
        out_features, in_features, coeff_dim = w.shape
        centers, labels = _kmeans(w.reshape(-1, coeff_dim).numpy(), num_clusters, seed)
        codebooks.append(torch.tensor(centers, dtype=torch.float32))
        cluster_ids.append(torch.tensor(labels, dtype=torch.long).reshape(out_features, in_features))
    return codebooks, cluster_ids

def cluster_dense_weights_function_space(dense_model, num_clusters=16, seed=42, function_samples=128, function_domain="grid", train_loader=None, device=None, activation_sample_batches=8, include_base=True, normalize_signatures=True):
    dense_cpu, domains = _domains_for_model(dense_model, function_domain, function_samples, train_loader, device, activation_sample_batches)
    codebooks, cluster_ids = [], []
    for layer, w, x_domain in zip(dense_cpu.layers, dense_cpu.get_edge_weights(), domains):
        out_features, in_features, coeff_dim = w.shape
        coeff_vectors = w.reshape(-1, coeff_dim).numpy()
        signatures = evaluate_spline_edge_functions(layer, w, x_domain, include_base=include_base, normalize_signatures=normalize_signatures).numpy()
        k = min(num_clusters, signatures.shape[0])
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(signatures)
        centers = np.zeros((k, coeff_dim), dtype=np.float32)
        rng = np.random.RandomState(seed)
        for j in range(k):
            mask = labels == j
            centers[j] = coeff_vectors[mask].mean(axis=0) if mask.any() else coeff_vectors[rng.randint(0, coeff_vectors.shape[0])]
        codebooks.append(torch.tensor(centers, dtype=torch.float32))
        cluster_ids.append(torch.tensor(labels, dtype=torch.long).reshape(out_features, in_features))
    return codebooks, cluster_ids

def cluster_spline_branch(dense_model, spline_clusters, seed, spline_method, function_samples, function_domain, train_loader, device, activation_sample_batches, normalize_signatures):
    dense_cpu = dense_model.cpu(); dense_cpu.eval()
    if spline_method == "function":
        dense_cpu, domains = _domains_for_model(dense_model, function_domain, function_samples, train_loader, device, activation_sample_batches)
    elif spline_method == "coefficient":
        domains = [None for _ in dense_cpu.layers]
    else:
        raise ValueError(f"Unknown spline_method: {spline_method}")
    spline_codebooks, spline_ids_all = [], []
    for layer_idx, (layer, w, x_domain) in enumerate(zip(dense_cpu.layers, dense_cpu.get_edge_weights(), domains)):
        out_features, in_features, coeff_dim = w.shape
        spline_dim = coeff_dim - 1
        spline_vectors = w[..., :-1].reshape(-1, spline_dim).numpy()
        if spline_method == "function":
            signatures = evaluate_spline_edge_functions(layer, w, x_domain, include_base=False, normalize_signatures=normalize_signatures).numpy()
            k = min(spline_clusters, signatures.shape[0])
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(signatures)
            centers = np.zeros((k, spline_dim), dtype=np.float32)
            rng = np.random.RandomState(seed + layer_idx)
            for j in range(k):
                mask = labels == j
                centers[j] = spline_vectors[mask].mean(axis=0) if mask.any() else spline_vectors[rng.randint(0, spline_vectors.shape[0])]
        else:
            centers, labels = _kmeans(spline_vectors, spline_clusters, seed)
        spline_codebooks.append(torch.tensor(centers, dtype=torch.float32))
        spline_ids_all.append(torch.tensor(labels, dtype=torch.long).reshape(out_features, in_features))
    return dense_cpu, spline_codebooks, spline_ids_all

def _conditional_base_from_spline(dense_cpu, spline_codebooks, spline_ids_all):
    conditional_base_codebooks = []
    for layer_idx, w in enumerate(dense_cpu.get_edge_weights()):
        base_values = w[..., -1].reshape(-1, 1).numpy()
        labels = spline_ids_all[layer_idx].reshape(-1).numpy()
        k = spline_codebooks[layer_idx].shape[0]
        centers = np.zeros((k, 1), dtype=np.float32)
        global_mean = base_values.mean(axis=0)
        for j in range(k):
            mask = labels == j
            centers[j] = base_values[mask].mean(axis=0) if mask.any() else global_mean
        conditional_base_codebooks.append(torch.tensor(centers, dtype=torch.float32))
    return conditional_base_codebooks

def cluster_dense_weights_branch_aware(dense_model, spline_clusters=16, base_clusters=8, seed=42, spline_method="function", function_samples=128, function_domain="grid", train_loader=None, device=None, activation_sample_batches=8, normalize_signatures=True):
    dense_cpu, spline_cbs, spline_ids = cluster_spline_branch(dense_model, spline_clusters, seed, spline_method, function_samples, function_domain, train_loader, device, activation_sample_batches, normalize_signatures)
    base_cbs, base_ids_all = [], []
    for w in dense_cpu.get_edge_weights():
        out_features, in_features, _ = w.shape
        centers, labels = _kmeans(w[..., -1].reshape(-1, 1).numpy(), base_clusters, seed)
        base_cbs.append(torch.tensor(centers, dtype=torch.float32))
        base_ids_all.append(torch.tensor(labels, dtype=torch.long).reshape(out_features, in_features))
    return spline_cbs, spline_ids, base_cbs, base_ids_all

def cluster_dense_weights_index_efficient_branch(dense_model, spline_clusters=16, seed=42, spline_method="function", function_samples=128, function_domain="grid", train_loader=None, device=None, activation_sample_batches=8, normalize_signatures=True):
    dense_cpu, spline_cbs, spline_ids = cluster_spline_branch(dense_model, spline_clusters, seed, spline_method, function_samples, function_domain, train_loader, device, activation_sample_batches, normalize_signatures)
    cond_base_cbs = _conditional_base_from_spline(dense_cpu, spline_cbs, spline_ids)
    return spline_cbs, spline_ids, cond_base_cbs

def cluster_dense_weights_sparse_residual_branch(dense_model, spline_clusters=16, residual_fraction=0.25, residual_base_clusters=8, residual_selection="error", seed=42, spline_method="function", function_samples=128, function_domain="grid", train_loader=None, device=None, activation_sample_batches=8, normalize_signatures=True):
    dense_cpu, spline_cbs, spline_ids = cluster_spline_branch(dense_model, spline_clusters, seed, spline_method, function_samples, function_domain, train_loader, device, activation_sample_batches, normalize_signatures)
    cond_base_cbs = _conditional_base_from_spline(dense_cpu, spline_cbs, spline_ids)
    residual_cbs, residual_positions, residual_ids = [], [], []
    for layer_idx, w in enumerate(dense_cpu.get_edge_weights()):
        original_base = w[..., -1].reshape(-1, 1).numpy()
        sids_flat = spline_ids[layer_idx].reshape(-1).numpy()
        pred_base = cond_base_cbs[layer_idx].numpy()[sids_flat]
        residual = original_base - pred_base
        n_edges = original_base.shape[0]
        n_select = int(round(float(residual_fraction) * n_edges))
        n_select = max(0, min(n_edges, n_select))
        if n_select == 0:
            residual_cbs.append(torch.zeros((1, 1), dtype=torch.float32))
            residual_positions.append(torch.empty(0, dtype=torch.long))
            residual_ids.append(torch.empty(0, dtype=torch.long))
            continue
        if residual_selection == "magnitude":
            score = np.abs(original_base[:, 0])
        elif residual_selection == "error":
            score = np.abs(residual[:, 0])
        else:
            raise ValueError(f"Unknown residual_selection: {residual_selection}")
        selected = np.argpartition(-score, n_select - 1)[:n_select]
        selected = np.sort(selected)
        centers, labels = _kmeans(residual[selected], residual_base_clusters, seed + layer_idx)
        residual_cbs.append(torch.tensor(centers, dtype=torch.float32))
        residual_positions.append(torch.tensor(selected, dtype=torch.long))
        residual_ids.append(torch.tensor(labels, dtype=torch.long))
    return spline_cbs, spline_ids, cond_base_cbs, residual_cbs, residual_positions, residual_ids

def build_clustered_from_dense(dense_model, input_dim, hidden_width, output_dim, grid_size, spline_order, num_clusters, seed, train_codebooks=True, cluster_method="coefficient", function_samples=128, function_domain="grid", train_loader=None, device=None, activation_sample_batches=8, include_base_in_function=True, normalize_function_signatures=True, branch_spline_clusters=None, branch_base_clusters=8, branch_spline_method="function", branch_function_samples=128, branch_function_domain="grid", residual_fraction=0.25, residual_base_clusters=8, residual_selection="error"):
    if cluster_method == "coefficient":
        cbs, ids = cluster_dense_weights_coefficient_space(dense_model.get_edge_weights(), num_clusters, seed)
        return ClusteredSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order, cbs, ids, train_codebooks)
    if cluster_method == "function":
        cbs, ids = cluster_dense_weights_function_space(dense_model, num_clusters, seed, function_samples, function_domain, train_loader, device, activation_sample_batches, include_base_in_function, normalize_function_signatures)
        return ClusteredSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order, cbs, ids, train_codebooks)
    if branch_spline_clusters is None: branch_spline_clusters = num_clusters
    if cluster_method == "branch":
        scb, sids, bcb, bids = cluster_dense_weights_branch_aware(dense_model, branch_spline_clusters, branch_base_clusters, seed, branch_spline_method, branch_function_samples, branch_function_domain, train_loader, device, activation_sample_batches, normalize_function_signatures)
        return BranchAwareClusteredSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order, scb, sids, bcb, bids, train_codebooks)
    if cluster_method == "branch_index":
        scb, sids, cb = cluster_dense_weights_index_efficient_branch(dense_model, branch_spline_clusters, seed, branch_spline_method, branch_function_samples, branch_function_domain, train_loader, device, activation_sample_batches, normalize_function_signatures)
        return IndexEfficientBranchSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order, scb, sids, cb, train_codebooks)
    if cluster_method == "branch_residual":
        scb, sids, cb, rcb, rpos, rids = cluster_dense_weights_sparse_residual_branch(dense_model, branch_spline_clusters, residual_fraction, residual_base_clusters, residual_selection, seed, branch_spline_method, branch_function_samples, branch_function_domain, train_loader, device, activation_sample_batches, normalize_function_signatures)
        return SparseResidualBranchSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order, scb, sids, cb, rcb, rpos, rids, train_codebooks)
    raise ValueError(f"Unknown cluster_method: {cluster_method}")
