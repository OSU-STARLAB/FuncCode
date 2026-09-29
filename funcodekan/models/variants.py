import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name: str = "silu"):
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class SplineBasisLayer(nn.Module):
    variant_name = "spline"

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0), activation="silu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_dim = in_features
        self.output_dim = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.basis_dim = grid_size + spline_order
        self.coeff_dim = self.basis_dim + 1
        self.base_activation = make_activation(activation)

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

    def basis(self, x):
        grid = self.grid.to(device=x.device, dtype=x.dtype)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        eps = 1e-8

        for k in range(1, self.spline_order + 1):
            delta_prev = grid[:, k:-1] - grid[:, :-(k + 1)]
            delta_next = grid[:, k + 1:] - grid[:, 1:(-k)]
            term1 = (x - grid[:, :-(k + 1)]) / (delta_prev + eps) * bases[:, :, :-1]
            term2 = (grid[:, k + 1:] - x) / (delta_next + eps) * bases[:, :, 1:]
            bases = term1 + term2

        return bases.contiguous()

    def forward_from_weight(self, x, weight):
        weight = weight.to(device=x.device, dtype=x.dtype)
        basis_weight = weight[..., :-1]
        base_weight = weight[..., -1]
        base = F.linear(self.base_activation(x), base_weight)
        basis = self.basis(x)
        nonlinear = torch.einsum("nik,oik->no", basis, basis_weight)
        return base + nonlinear

    def edge_function_signatures(self, weight, x_domain, include_base=True, normalize=True):
        device = weight.device
        dtype = weight.dtype
        x_domain = x_domain.to(device=device, dtype=dtype)

        basis = self.basis(x_domain[:, None].expand(-1, self.in_features))
        # [samples, in, basis]
        basis_weight = weight[..., :-1]  # [out, in, basis]
        sig = torch.einsum("nib,oib->oin", basis, basis_weight)

        if include_base:
            base_x = self.base_activation(x_domain)
            sig = sig + weight[..., -1].unsqueeze(-1) * base_x.view(1, 1, -1)

        sig = sig.reshape(-1, x_domain.numel())
        if normalize:
            sig = sig - sig.mean(dim=1, keepdim=True)
            sig = sig / (sig.std(dim=1, keepdim=True) + 1e-6)
        return sig.detach().cpu()


class FastRBFLayer(nn.Module):
    variant_name = "fast"

    def __init__(self, in_features, out_features, num_grids=8, grid_min=-2.0, grid_max=2.0, activation="silu", use_layernorm=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_dim = in_features
        self.output_dim = out_features
        self.num_grids = num_grids
        self.basis_dim = num_grids
        self.coeff_dim = self.basis_dim + 1
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.denominator = (grid_max - grid_min) / max(num_grids - 1, 1)
        self.base_activation = make_activation(activation)
        self.layernorm = nn.LayerNorm(in_features) if use_layernorm and in_features > 1 else None
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_grids))

    def basis(self, x):
        if self.layernorm is not None:
            x = self.layernorm(x)
        grid = self.grid.to(device=x.device, dtype=x.dtype)
        return torch.exp(-((x[..., None] - grid) / self.denominator) ** 2)

    def forward_from_weight(self, x, weight):
        weight = weight.to(device=x.device, dtype=x.dtype)
        basis_weight = weight[..., :-1]
        base_weight = weight[..., -1]
        base = F.linear(self.base_activation(x), base_weight)
        basis = self.basis(x)
        nonlinear = torch.einsum("nik,oik->no", basis, basis_weight)
        return base + nonlinear

    def edge_function_signatures(self, weight, x_domain, include_base=True, normalize=True):
        device = weight.device
        dtype = weight.dtype
        x_domain = x_domain.to(device=device, dtype=dtype)
        x_eval = x_domain[:, None].expand(-1, self.in_features)
        basis = self.basis(x_eval)
        basis_weight = weight[..., :-1]
        sig = torch.einsum("nib,oib->oin", basis, basis_weight)

        if include_base:
            base_x = self.base_activation(x_domain)
            sig = sig + weight[..., -1].unsqueeze(-1) * base_x.view(1, 1, -1)

        sig = sig.reshape(-1, x_domain.numel())
        if normalize:
            sig = sig - sig.mean(dim=1, keepdim=True)
            sig = sig / (sig.std(dim=1, keepdim=True) + 1e-6)
        return sig.detach().cpu()


class GramPolynomialLayer(nn.Module):
    variant_name = "gram"

    def __init__(self, in_features, out_features, degree=3, activation="silu", use_norm=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_dim = in_features
        self.output_dim = out_features
        self.degree = degree
        self.basis_dim = degree + 1
        self.coeff_dim = self.basis_dim + 1
        self.base_activation = make_activation(activation)
        self.use_norm = use_norm
        self.norm = nn.LayerNorm(out_features) if use_norm else nn.Identity()

    def basis(self, x):
        # Stable polynomial basis over tanh-normalized inputs.
        z = torch.tanh(x)
        basis_terms = [torch.ones_like(z)]
        if self.degree >= 1:
            basis_terms.append(z)
        for d in range(2, self.degree + 1):
            # Simple recurrence-style polynomial features.
            basis_terms.append(z * basis_terms[-1] - 0.1 * basis_terms[-2])
        return torch.stack(basis_terms, dim=-1)

    def forward_from_weight(self, x, weight):
        weight = weight.to(device=x.device, dtype=x.dtype)
        basis_weight = weight[..., :-1]
        base_weight = weight[..., -1]
        base = F.linear(self.base_activation(x), base_weight)
        basis = self.basis(x)
        nonlinear = torch.einsum("nik,oik->no", basis, basis_weight)
        return self.base_activation(self.norm(base + nonlinear))

    def edge_function_signatures(self, weight, x_domain, include_base=True, normalize=True):
        device = weight.device
        dtype = weight.dtype
        x_domain = x_domain.to(device=device, dtype=dtype)
        x_eval = x_domain[:, None].expand(-1, self.in_features)
        basis = self.basis(x_eval)
        basis_weight = weight[..., :-1]
        sig = torch.einsum("nib,oib->oin", basis, basis_weight)

        if include_base:
            base_x = self.base_activation(x_domain)
            sig = sig + weight[..., -1].unsqueeze(-1) * base_x.view(1, 1, -1)

        sig = sig.reshape(-1, x_domain.numel())
        if normalize:
            sig = sig - sig.mean(dim=1, keepdim=True)
            sig = sig / (sig.std(dim=1, keepdim=True) + 1e-6)
        return sig.detach().cpu()


class DirectKANVariant(nn.Module):
    def __init__(
        self,
        variant: str,
        input_dim: int = 784,
        hidden_width: int = 64,
        output_dim: int = 10,
        grid_size: int = 5,
        spline_order: int = 3,
        num_grids: int = 8,
        degree: int = 3,
        activation: str = "silu",
    ):
        super().__init__()
        self.variant = variant
        self.input_dim = input_dim
        self.hidden_width = hidden_width
        self.output_dim = output_dim
        widths = [input_dim, hidden_width, output_dim]
        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()

        for din, dout in zip(widths[:-1], widths[1:]):
            if variant == "spline":
                layer = SplineBasisLayer(din, dout, grid_size=grid_size, spline_order=spline_order, activation=activation)
            elif variant == "fast":
                layer = FastRBFLayer(din, dout, num_grids=num_grids, activation=activation)
            elif variant == "gram":
                layer = GramPolynomialLayer(din, dout, degree=degree, activation=activation)
            else:
                raise ValueError(f"Unknown KAN variant: {variant}")

            self.layers.append(layer)
            w = torch.empty(dout, din, layer.coeff_dim)
            nn.init.trunc_normal_(w, mean=0.0, std=0.05)
            self.weights.append(nn.Parameter(w))

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer.forward_from_weight(x, self.weights[idx])
        return x

    def get_edge_weights(self):
        return [w.detach().cpu() for w in self.weights]

    def dense_storage_bits(self):
        return int(sum(w.numel() * 32 for w in self.weights))


class MLPBaseline(nn.Module):
    def __init__(self, input_dim=784, hidden_width=64, output_dim=10, activation="silu"):
        super().__init__()
        self.variant = "mlp"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            make_activation(activation),
            nn.Linear(hidden_width, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def dense_storage_bits(self):
        total = 0
        for p in self.parameters():
            total += p.numel() * 32
        return int(total)


class SharedCodebookKAN(nn.Module):
    compression_type = "shared"

    def __init__(self, dense_model: DirectKANVariant, codebooks: List[torch.Tensor], cluster_ids: List[torch.Tensor], train_codebooks=True):
        super().__init__()
        self.variant = dense_model.variant
        self.layers = dense_model.layers
        self.codebooks = nn.ParameterList([nn.Parameter(cb.clone().float(), requires_grad=train_codebooks) for cb in codebooks])
        self.id_names = []
        for idx, ids in enumerate(cluster_ids):
            self.register_buffer(f"cluster_ids_{idx}", ids.clone().long())
            self.id_names.append(f"cluster_ids_{idx}")

    def get_cluster_ids(self, idx):
        return getattr(self, self.id_names[idx])

    def reconstruct_weight(self, idx):
        ids = self.get_cluster_ids(idx).to(self.codebooks[idx].device)
        return self.codebooks[idx][ids]

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer.forward_from_weight(x, self.reconstruct_weight(idx))
        return x


class BranchCodebookKAN(nn.Module):
    compression_type = "branch"

    def __init__(
        self,
        dense_model: DirectKANVariant,
        spline_codebooks: List[torch.Tensor],
        spline_ids: List[torch.Tensor],
        base_codebooks: List[torch.Tensor],
        base_ids: List[torch.Tensor],
        train_codebooks=True,
    ):
        super().__init__()
        self.variant = dense_model.variant
        self.layers = dense_model.layers
        self.spline_codebooks = nn.ParameterList([nn.Parameter(cb.clone().float(), requires_grad=train_codebooks) for cb in spline_codebooks])
        self.base_codebooks = nn.ParameterList([nn.Parameter((cb[:, None] if cb.dim() == 1 else cb).clone().float(), requires_grad=train_codebooks) for cb in base_codebooks])
        self.spline_id_names = []
        self.base_id_names = []
        for idx, ids in enumerate(spline_ids):
            self.register_buffer(f"spline_cluster_ids_{idx}", ids.clone().long())
            self.spline_id_names.append(f"spline_cluster_ids_{idx}")
        for idx, ids in enumerate(base_ids):
            self.register_buffer(f"base_cluster_ids_{idx}", ids.clone().long())
            self.base_id_names.append(f"base_cluster_ids_{idx}")

    def get_spline_cluster_ids(self, idx):
        return getattr(self, self.spline_id_names[idx])

    def get_base_cluster_ids(self, idx):
        return getattr(self, self.base_id_names[idx])

    def reconstruct_weight(self, idx):
        device = self.spline_codebooks[idx].device
        sids = self.get_spline_cluster_ids(idx).to(device)
        bids = self.get_base_cluster_ids(idx).to(device)
        spline_weight = self.spline_codebooks[idx][sids]
        base_weight = self.base_codebooks[idx][bids].squeeze(-1)
        return torch.cat([spline_weight, base_weight.unsqueeze(-1)], dim=-1)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer.forward_from_weight(x, self.reconstruct_weight(idx))
        return x
