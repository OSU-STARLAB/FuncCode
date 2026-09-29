import torch
import torch.nn as nn
import torch.nn.functional as F

class SplineLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0), base_activation=nn.SiLU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.coeff_dim = grid_size + spline_order + 1
        self.spline_coeff_dim = self.coeff_dim - 1
        self.base_activation = base_activation()
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

    def b_splines(self, x):
        # Device/dtype safety: keep grid aligned with input x.
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
        # Device/dtype safety:
        # Hardened or exported compressed models can reconstruct weights on CPU,
        # while train/eval batches are on CUDA. Align before slicing.
        if weight.device != x.device or weight.dtype != x.dtype:
            weight = weight.to(device=x.device, dtype=x.dtype)

        spline_weight = weight[..., :-1]
        base_weight = weight[..., -1]

        base_output = F.linear(self.base_activation(x), base_weight)
        spline_basis = self.b_splines(x)
        spline_output = torch.einsum("nik,oik->no", spline_basis, spline_weight)

        return base_output + spline_output

class DenseSplineKAN(nn.Module):
    def __init__(self, input_dim=784, hidden_width=64, output_dim=10, grid_size=5, spline_order=3):
        super().__init__()
        widths = [input_dim, hidden_width, output_dim]
        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()
        for din, dout in zip(widths[:-1], widths[1:]):
            layer = SplineLayer(din, dout, grid_size=grid_size, spline_order=spline_order)
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

class ClusteredSplineKAN(nn.Module):
    compression_type = "shared"
    def __init__(self, input_dim, hidden_width, output_dim, grid_size, spline_order, codebooks, cluster_ids, train_codebooks=True):
        super().__init__()
        widths = [input_dim, hidden_width, output_dim]
        self.layers = nn.ModuleList([SplineLayer(din, dout, grid_size=grid_size, spline_order=spline_order) for din, dout in zip(widths[:-1], widths[1:])])
        self.codebooks = nn.ParameterList([nn.Parameter(cb.clone().float(), requires_grad=train_codebooks) for cb in codebooks])
        self.cluster_id_names = []
        for idx, ids in enumerate(cluster_ids):
            self.register_buffer(f"cluster_ids_{idx}", ids.clone().long())
            self.cluster_id_names.append(f"cluster_ids_{idx}")

    def get_cluster_ids(self, idx): return getattr(self, self.cluster_id_names[idx])
    def reconstruct_weight(self, idx): return self.codebooks[idx][self.get_cluster_ids(idx)]
    def forward(self, x):
        for idx, layer in enumerate(self.layers): x = layer.forward_from_weight(x, self.reconstruct_weight(idx))
        return x
    def export_clustered_state(self):
        state = {}
        for i, cb in enumerate(self.codebooks):
            state[f"clustered_weights.{i}"] = cb.detach().cpu(); state[f"cluster_ids.{i}"] = self.get_cluster_ids(i).detach().cpu()
        return state

class BranchAwareClusteredSplineKAN(nn.Module):
    compression_type = "branch"
    def __init__(self, input_dim, hidden_width, output_dim, grid_size, spline_order, spline_codebooks, spline_cluster_ids, base_codebooks, base_cluster_ids, train_codebooks=True):
        super().__init__()
        widths = [input_dim, hidden_width, output_dim]
        self.layers = nn.ModuleList([SplineLayer(din, dout, grid_size=grid_size, spline_order=spline_order) for din, dout in zip(widths[:-1], widths[1:])])
        self.spline_codebooks = nn.ParameterList([nn.Parameter(cb.clone().float(), requires_grad=train_codebooks) for cb in spline_codebooks])
        self.base_codebooks = nn.ParameterList([nn.Parameter((cb[:, None] if cb.dim() == 1 else cb).clone().float(), requires_grad=train_codebooks) for cb in base_codebooks])
        self.spline_id_names, self.base_id_names = [], []
        for idx, ids in enumerate(spline_cluster_ids):
            self.register_buffer(f"spline_cluster_ids_{idx}", ids.clone().long()); self.spline_id_names.append(f"spline_cluster_ids_{idx}")
        for idx, ids in enumerate(base_cluster_ids):
            self.register_buffer(f"base_cluster_ids_{idx}", ids.clone().long()); self.base_id_names.append(f"base_cluster_ids_{idx}")
    def get_spline_cluster_ids(self, idx): return getattr(self, self.spline_id_names[idx])
    def get_base_cluster_ids(self, idx): return getattr(self, self.base_id_names[idx])
    def reconstruct_weight(self, idx):
        sids = self.get_spline_cluster_ids(idx); bids = self.get_base_cluster_ids(idx)
        spline_weight = self.spline_codebooks[idx][sids]
        base_weight = self.base_codebooks[idx][bids].squeeze(-1)
        return torch.cat([spline_weight, base_weight.unsqueeze(-1)], dim=-1)
    def forward(self, x):
        for idx, layer in enumerate(self.layers): x = layer.forward_from_weight(x, self.reconstruct_weight(idx))
        return x
    def export_clustered_state(self):
        state = {}
        for i in range(len(self.spline_codebooks)):
            state[f"spline_clustered_weights.{i}"] = self.spline_codebooks[i].detach().cpu(); state[f"spline_cluster_ids.{i}"] = self.get_spline_cluster_ids(i).detach().cpu()
            state[f"base_clustered_weights.{i}"] = self.base_codebooks[i].detach().cpu(); state[f"base_cluster_ids.{i}"] = self.get_base_cluster_ids(i).detach().cpu()
        return state

class IndexEfficientBranchSplineKAN(nn.Module):
    compression_type = "branch_index"
    def __init__(self, input_dim, hidden_width, output_dim, grid_size, spline_order, spline_codebooks, spline_cluster_ids, conditional_base_codebooks, train_codebooks=True):
        super().__init__()
        widths = [input_dim, hidden_width, output_dim]
        self.layers = nn.ModuleList([SplineLayer(din, dout, grid_size=grid_size, spline_order=spline_order) for din, dout in zip(widths[:-1], widths[1:])])
        self.spline_codebooks = nn.ParameterList([nn.Parameter(cb.clone().float(), requires_grad=train_codebooks) for cb in spline_codebooks])
        self.conditional_base_codebooks = nn.ParameterList([nn.Parameter((cb[:, None] if cb.dim() == 1 else cb).clone().float(), requires_grad=train_codebooks) for cb in conditional_base_codebooks])
        self.spline_id_names = []
        for idx, ids in enumerate(spline_cluster_ids):
            self.register_buffer(f"spline_cluster_ids_{idx}", ids.clone().long()); self.spline_id_names.append(f"spline_cluster_ids_{idx}")
    def get_spline_cluster_ids(self, idx): return getattr(self, self.spline_id_names[idx])
    def reconstruct_weight(self, idx):
        ids = self.get_spline_cluster_ids(idx)
        spline_weight = self.spline_codebooks[idx][ids]
        base_weight = self.conditional_base_codebooks[idx][ids].squeeze(-1)
        return torch.cat([spline_weight, base_weight.unsqueeze(-1)], dim=-1)
    def forward(self, x):
        for idx, layer in enumerate(self.layers): x = layer.forward_from_weight(x, self.reconstruct_weight(idx))
        return x
    def export_clustered_state(self):
        state = {}
        for i in range(len(self.spline_codebooks)):
            state[f"spline_clustered_weights.{i}"] = self.spline_codebooks[i].detach().cpu(); state[f"spline_cluster_ids.{i}"] = self.get_spline_cluster_ids(i).detach().cpu(); state[f"conditional_base_weights.{i}"] = self.conditional_base_codebooks[i].detach().cpu()
        return state

class SparseResidualBranchSplineKAN(nn.Module):
    compression_type = "branch_residual"
    def __init__(self, input_dim, hidden_width, output_dim, grid_size, spline_order, spline_codebooks, spline_cluster_ids, conditional_base_codebooks, residual_base_codebooks, residual_positions, residual_cluster_ids, train_codebooks=True):
        super().__init__()
        widths = [input_dim, hidden_width, output_dim]
        self.layers = nn.ModuleList([SplineLayer(din, dout, grid_size=grid_size, spline_order=spline_order) for din, dout in zip(widths[:-1], widths[1:])])
        self.spline_codebooks = nn.ParameterList([nn.Parameter(cb.clone().float(), requires_grad=train_codebooks) for cb in spline_codebooks])
        self.conditional_base_codebooks = nn.ParameterList([nn.Parameter((cb[:, None] if cb.dim() == 1 else cb).clone().float(), requires_grad=train_codebooks) for cb in conditional_base_codebooks])
        self.residual_base_codebooks = nn.ParameterList([nn.Parameter((cb[:, None] if cb.dim() == 1 else cb).clone().float(), requires_grad=train_codebooks) for cb in residual_base_codebooks])
        self.spline_id_names, self.residual_position_names, self.residual_id_names = [], [], []
        for idx, ids in enumerate(spline_cluster_ids):
            self.register_buffer(f"spline_cluster_ids_{idx}", ids.clone().long()); self.spline_id_names.append(f"spline_cluster_ids_{idx}")
        for idx, pos in enumerate(residual_positions):
            self.register_buffer(f"residual_positions_{idx}", pos.clone().long()); self.residual_position_names.append(f"residual_positions_{idx}")
        for idx, ids in enumerate(residual_cluster_ids):
            self.register_buffer(f"residual_cluster_ids_{idx}", ids.clone().long()); self.residual_id_names.append(f"residual_cluster_ids_{idx}")
    def get_spline_cluster_ids(self, idx): return getattr(self, self.spline_id_names[idx])
    def get_residual_positions(self, idx): return getattr(self, self.residual_position_names[idx])
    def get_residual_cluster_ids(self, idx): return getattr(self, self.residual_id_names[idx])
    def reconstruct_weight(self, idx):
        sids = self.get_spline_cluster_ids(idx)
        out_features, in_features = sids.shape
        spline_weight = self.spline_codebooks[idx][sids]
        base_weight = self.conditional_base_codebooks[idx][sids].squeeze(-1)
        pos = self.get_residual_positions(idx); rids = self.get_residual_cluster_ids(idx)
        if pos.numel() > 0:
            flat_base = base_weight.reshape(-1).clone()
            residual_values = self.residual_base_codebooks[idx][rids].squeeze(-1)
            flat_base[pos] = flat_base[pos] + residual_values
            base_weight = flat_base.reshape(out_features, in_features)
        return torch.cat([spline_weight, base_weight.unsqueeze(-1)], dim=-1)
    def forward(self, x):
        for idx, layer in enumerate(self.layers): x = layer.forward_from_weight(x, self.reconstruct_weight(idx))
        return x
    def export_clustered_state(self):
        state = {}
        for i in range(len(self.spline_codebooks)):
            state[f"spline_clustered_weights.{i}"] = self.spline_codebooks[i].detach().cpu(); state[f"spline_cluster_ids.{i}"] = self.get_spline_cluster_ids(i).detach().cpu()
            state[f"conditional_base_weights.{i}"] = self.conditional_base_codebooks[i].detach().cpu(); state[f"residual_base_weights.{i}"] = self.residual_base_codebooks[i].detach().cpu()
            state[f"residual_positions.{i}"] = self.get_residual_positions(i).detach().cpu(); state[f"residual_cluster_ids.{i}"] = self.get_residual_cluster_ids(i).detach().cpu()
        return state
