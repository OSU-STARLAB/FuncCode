import torch
import torch.nn as nn
from typing import List

@torch.no_grad()
def make_grid_domain(num_samples: int = 128, grid_min: float = -2.5, grid_max: float = 2.5, device: str = "cpu") -> torch.Tensor:
    return torch.linspace(grid_min, grid_max, num_samples, device=device)

@torch.no_grad()
def collect_layer_activation_domains(model: nn.Module, train_loader, device, max_batches: int = 8, num_samples: int = 128) -> List[torch.Tensor]:
    model.eval()
    collected = [[] for _ in model.layers]
    seen = 0

    for x, _ in train_loader:
        x = x.to(device)
        for layer_idx, layer in enumerate(model.layers):
            collected[layer_idx].append(x.detach().flatten().cpu())
            x = layer.forward_from_weight(x, model.weights[layer_idx])
        seen += 1
        if seen >= max_batches:
            break

    domains = []
    for vals in collected:
        vals = torch.cat(vals)
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            domains.append(torch.linspace(-2.5, 2.5, num_samples))
            continue
        lo = torch.quantile(vals, 0.005)
        hi = torch.quantile(vals, 0.995)
        vals = vals[(vals >= lo) & (vals <= hi)]
        if vals.numel() < num_samples:
            domains.append(torch.linspace(float(lo), float(hi), num_samples))
        else:
            domains.append(torch.quantile(vals, torch.linspace(0.0, 1.0, num_samples)))
    return domains

@torch.no_grad()
def evaluate_spline_edge_functions(layer, weight: torch.Tensor, x_domain: torch.Tensor, include_base: bool = True, normalize_signatures: bool = True, chunk_edges: int = 8192) -> torch.Tensor:
    """
    Convert layer edges into function signatures.

    weight: [out_features, in_features, coeff_dim]
    returns: [out_features * in_features, len(x_domain)]
    """
    device = weight.device
    x_domain = x_domain.to(device).float()
    out_features, in_features, coeff_dim = weight.shape
    spline_dim = coeff_dim - 1

    x_matrix = x_domain[:, None].repeat(1, in_features)
    basis = layer.b_splines(x_matrix)  # [T, I, D]

    spline_weight = weight[..., :-1]
    base_weight = weight[..., -1]

    edge_spline_weight = spline_weight.reshape(out_features * in_features, spline_dim)
    edge_base_weight = base_weight.reshape(out_features * in_features)

    basis_idt = basis.permute(1, 2, 0).contiguous()  # [I, D, T]
    feature_ids = torch.arange(in_features, device=device).repeat(out_features)
    edge_basis = basis_idt[feature_ids]  # [O*I, D, T]

    chunks = []
    n_edges = edge_spline_weight.shape[0]
    for start in range(0, n_edges, chunk_edges):
        end = min(start + chunk_edges, n_edges)
        sig = torch.einsum("bd,bdt->bt", edge_spline_weight[start:end], edge_basis[start:end])

        if include_base:
            base_curve = layer.base_activation(x_domain)
            sig = sig + edge_base_weight[start:end, None] * base_curve[None, :]

        if normalize_signatures:
            sig = (sig - sig.mean(dim=1, keepdim=True)) / sig.std(dim=1, keepdim=True).clamp_min(1e-6)

        chunks.append(sig.cpu())

    return torch.cat(chunks, dim=0)
