import math

def required_index_bits(num_clusters):
    return 1 if num_clusters <= 1 else int(math.ceil(math.log2(num_clusters)))

def bits_to_kib(bits): return bits / 8.0 / 1024.0

def compression_ratio(reference_bits, compressed_bits): return float(reference_bits / max(compressed_bits, 1))

def dense_fp32_bits_from_model(model): return int(sum(w.numel() * 32 for w in model.weights))

def zero_breakdown():
    return {"shared_codebook_bits":0,"shared_index_bits":0,"spline_codebook_bits":0,"spline_index_bits":0,"base_codebook_bits":0,"base_index_bits":0,"conditional_base_codebook_bits":0,"residual_base_codebook_bits":0,"residual_index_bits":0,"residual_position_bits":0,"scale_bits":0,"metadata_bits":0}

def dense_storage_breakdown(model):
    total = dense_fp32_bits_from_model(model); out = zero_breakdown(); out.update({"storage_total_bits":total,"storage_total_kib":bits_to_kib(total)}); return out

def shared_storage_breakdown(model, codebook_bits=32, scale_bits_per_value=0):
    out = zero_breakdown()
    for idx, cb in enumerate(model.codebooks):
        ids = model.get_cluster_ids(idx); k = cb.shape[0]
        out["shared_codebook_bits"] += int(cb.numel() * codebook_bits)
        out["shared_index_bits"] += int(ids.numel() * required_index_bits(k))
        if scale_bits_per_value > 0: out["scale_bits"] += int(k * scale_bits_per_value)
    total = sum(out.values()); out["storage_total_bits"] = int(total); out["storage_total_kib"] = bits_to_kib(total); return out

def branch_storage_breakdown(model, codebook_bits=32, scale_bits_per_value=0):
    out = zero_breakdown()
    for idx, cb in enumerate(model.spline_codebooks):
        ids = model.get_spline_cluster_ids(idx); k = cb.shape[0]
        out["spline_codebook_bits"] += int(cb.numel() * codebook_bits)
        out["spline_index_bits"] += int(ids.numel() * required_index_bits(k))
        if scale_bits_per_value > 0: out["scale_bits"] += int(k * scale_bits_per_value)
    for idx, cb in enumerate(model.base_codebooks):
        ids = model.get_base_cluster_ids(idx); k = cb.shape[0]
        out["base_codebook_bits"] += int(cb.numel() * codebook_bits)
        out["base_index_bits"] += int(ids.numel() * required_index_bits(k))
        if scale_bits_per_value > 0: out["scale_bits"] += int(k * scale_bits_per_value)
    total = sum(out.values()); out["storage_total_bits"] = int(total); out["storage_total_kib"] = bits_to_kib(total); return out

def branch_index_storage_breakdown(model, codebook_bits=32, scale_bits_per_value=0):
    out = zero_breakdown()
    for idx, cb in enumerate(model.spline_codebooks):
        ids = model.get_spline_cluster_ids(idx); k = cb.shape[0]
        out["spline_codebook_bits"] += int(cb.numel() * codebook_bits)
        out["spline_index_bits"] += int(ids.numel() * required_index_bits(k))
        if scale_bits_per_value > 0: out["scale_bits"] += int(k * scale_bits_per_value)
    for idx, cb in enumerate(model.conditional_base_codebooks):
        k = cb.shape[0]
        out["conditional_base_codebook_bits"] += int(cb.numel() * codebook_bits)
        if scale_bits_per_value > 0: out["scale_bits"] += int(k * scale_bits_per_value)
    total = sum(out.values()); out["storage_total_bits"] = int(total); out["storage_total_kib"] = bits_to_kib(total); return out

def branch_residual_storage_breakdown(model, codebook_bits=32, scale_bits_per_value=0):
    out = zero_breakdown()
    for idx, cb in enumerate(model.spline_codebooks):
        ids = model.get_spline_cluster_ids(idx); k = cb.shape[0]
        out["spline_codebook_bits"] += int(cb.numel() * codebook_bits)
        out["spline_index_bits"] += int(ids.numel() * required_index_bits(k))
        if scale_bits_per_value > 0: out["scale_bits"] += int(k * scale_bits_per_value)
    for idx, cb in enumerate(model.conditional_base_codebooks):
        k = cb.shape[0]
        out["conditional_base_codebook_bits"] += int(cb.numel() * codebook_bits)
        if scale_bits_per_value > 0: out["scale_bits"] += int(k * scale_bits_per_value)
    for idx, cb in enumerate(model.residual_base_codebooks):
        rids = model.get_residual_cluster_ids(idx); pos = model.get_residual_positions(idx); k = cb.shape[0]
        n_edges = model.get_spline_cluster_ids(idx).numel()
        out["residual_base_codebook_bits"] += int(cb.numel() * codebook_bits)
        out["residual_index_bits"] += int(rids.numel() * required_index_bits(k))
        out["residual_position_bits"] += int(pos.numel() * required_index_bits(n_edges))
        if scale_bits_per_value > 0: out["scale_bits"] += int(k * scale_bits_per_value)
    total = sum(out.values()); out["storage_total_bits"] = int(total); out["storage_total_kib"] = bits_to_kib(total); return out

def compressed_storage_breakdown(model, codebook_bits=32, scale_bits_per_value=0):
    ctype = getattr(model, "compression_type", "shared")
    if ctype == "branch": return branch_storage_breakdown(model, codebook_bits, scale_bits_per_value)
    if ctype == "branch_index": return branch_index_storage_breakdown(model, codebook_bits, scale_bits_per_value)
    if ctype == "branch_residual": return branch_residual_storage_breakdown(model, codebook_bits, scale_bits_per_value)
    return shared_storage_breakdown(model, codebook_bits, scale_bits_per_value)

def clustered_fp32_bits_from_model(model): return int(compressed_storage_breakdown(model, 32, 0)["storage_total_bits"])
def hwq_bits_from_model(model, codebook_bits, scale_bits=32): return int(compressed_storage_breakdown(model, codebook_bits, scale_bits)["storage_total_bits"])

def add_kib_columns(breakdown):
    out = dict(breakdown)
    for key, value in list(breakdown.items()):
        if key.endswith("_bits"): out[key.replace("_bits", "_kib")] = bits_to_kib(int(value))
    return out
