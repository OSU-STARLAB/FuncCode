import torch
from ..models.spline import ClusteredSplineKAN, BranchAwareClusteredSplineKAN, IndexEfficientBranchSplineKAN, SparseResidualBranchSplineKAN
from ..analysis.storage import required_index_bits

def symmetric_quantize(x, bits, per_vector=True):
    assert 2 <= bits <= 8
    qmax = 2 ** (bits - 1) - 1; qmin = -2 ** (bits - 1)
    max_abs = x.abs().amax(dim=1, keepdim=True) if per_vector else x.abs().amax().view(1)
    scale = torch.clamp(max_abs / qmax, min=1e-8)
    q = torch.round(x / scale).clamp(qmin, qmax).to(torch.int8)
    return q, scale, q.float() * scale

def pack_unsigned(values, bits):
    values = values.reshape(-1).detach().cpu().to(torch.long)
    assert 1 <= bits <= 8
    total_bits = values.numel() * bits; num_bytes = (total_bits + 7) // 8
    out = torch.zeros(num_bytes, dtype=torch.uint8); bit_pos = 0
    for value in values.tolist():
        for b in range(bits - 1, -1, -1):
            if (value >> b) & 1:
                out[bit_pos // 8] |= (1 << (7 - (bit_pos % 8)))
            bit_pos += 1
    return out

def signed_to_unsigned(q, bits): return (q.to(torch.long) + 2 ** (bits - 1)).clamp(0, 2 ** bits - 1)

def _quantize_list(cbs, bits, per_vector):
    out = []
    for cb in cbs:
        _, _, dq = symmetric_quantize(cb.detach().cpu(), bits=bits, per_vector=per_vector)
        out.append(dq)
    return out

def quantize_clustered_model(model, bits, input_dim, hidden_width, output_dim, grid_size, spline_order, per_vector=True):
    ctype = getattr(model, "compression_type", "shared")
    if ctype == "branch":
        return BranchAwareClusteredSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order,
            _quantize_list(model.spline_codebooks, bits, per_vector), [model.get_spline_cluster_ids(i).detach().cpu() for i in range(len(model.spline_codebooks))],
            _quantize_list(model.base_codebooks, bits, per_vector), [model.get_base_cluster_ids(i).detach().cpu() for i in range(len(model.base_codebooks))], False)
    if ctype == "branch_index":
        return IndexEfficientBranchSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order,
            _quantize_list(model.spline_codebooks, bits, per_vector), [model.get_spline_cluster_ids(i).detach().cpu() for i in range(len(model.spline_codebooks))],
            _quantize_list(model.conditional_base_codebooks, bits, per_vector), False)
    if ctype == "branch_residual":
        return SparseResidualBranchSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order,
            _quantize_list(model.spline_codebooks, bits, per_vector), [model.get_spline_cluster_ids(i).detach().cpu() for i in range(len(model.spline_codebooks))],
            _quantize_list(model.conditional_base_codebooks, bits, per_vector), _quantize_list(model.residual_base_codebooks, bits, per_vector),
            [model.get_residual_positions(i).detach().cpu() for i in range(len(model.residual_base_codebooks))],
            [model.get_residual_cluster_ids(i).detach().cpu() for i in range(len(model.residual_base_codebooks))], False)
    return ClusteredSplineKAN(input_dim, hidden_width, output_dim, grid_size, spline_order,
        _quantize_list(model.codebooks, bits, per_vector), [model.get_cluster_ids(i).detach().cpu() for i in range(len(model.codebooks))], False)

def _export_one_codebook(state, metadata, prefix, idx, cb, ids, bits, per_vector=True):
    q, scale, _ = symmetric_quantize(cb.detach().cpu(), bits=bits, per_vector=per_vector)
    state[f"{prefix}_q_codebook_packed.{idx}"] = pack_unsigned(signed_to_unsigned(q.reshape(-1), bits), bits)
    state[f"{prefix}_scale.{idx}"] = scale
    item = {"codebook_shape": list(cb.shape), "num_clusters": int(cb.shape[0]), "coeff_dim": int(cb.shape[1]), "scale_values": int(scale.numel())}
    if ids is not None:
        idx_bits = required_index_bits(cb.shape[0])
        state[f"{prefix}_cluster_ids_packed.{idx}"] = pack_unsigned(ids.detach().cpu().long().reshape(-1), idx_bits)
        item.update({"num_ids": int(ids.numel()), "index_bits": int(idx_bits)})
    metadata[f"{prefix}_layer_{idx}"] = item

def export_hwq_state(model, bits, save_path, per_vector=True):
    state = {}; ctype = getattr(model, "compression_type", "shared"); metadata = {"bits": bits, "compression_type": ctype, "layers": {}}
    if ctype == "branch":
        for i, cb in enumerate(model.spline_codebooks): _export_one_codebook(state, metadata["layers"], "spline", i, cb, model.get_spline_cluster_ids(i), bits, per_vector)
        for i, cb in enumerate(model.base_codebooks): _export_one_codebook(state, metadata["layers"], "base", i, cb, model.get_base_cluster_ids(i), bits, per_vector)
    elif ctype == "branch_index":
        for i, cb in enumerate(model.spline_codebooks): _export_one_codebook(state, metadata["layers"], "spline", i, cb, model.get_spline_cluster_ids(i), bits, per_vector)
        for i, cb in enumerate(model.conditional_base_codebooks): _export_one_codebook(state, metadata["layers"], "conditional_base", i, cb, None, bits, per_vector)
    elif ctype == "branch_residual":
        for i, cb in enumerate(model.spline_codebooks): _export_one_codebook(state, metadata["layers"], "spline", i, cb, model.get_spline_cluster_ids(i), bits, per_vector)
        for i, cb in enumerate(model.conditional_base_codebooks): _export_one_codebook(state, metadata["layers"], "conditional_base", i, cb, None, bits, per_vector)
        for i, cb in enumerate(model.residual_base_codebooks): _export_one_codebook(state, metadata["layers"], "residual_base", i, cb, model.get_residual_cluster_ids(i), bits, per_vector); state[f"residual_positions.{i}"] = model.get_residual_positions(i).detach().cpu()
    else:
        for i, cb in enumerate(model.codebooks): _export_one_codebook(state, metadata["layers"], "shared", i, cb, model.get_cluster_ids(i), bits, per_vector)
    torch.save({"state": state, "metadata": metadata}, save_path)
    return metadata
