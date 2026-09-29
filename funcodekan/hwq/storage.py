import torch
from dataclasses import dataclass
from typing import Optional
from .bitpacking import required_index_bits

@dataclass
class LayerStorageReport:
    layer: int
    num_edges: int
    num_clusters: int
    coeffs_per_edge: int
    codebook_bits: int
    index_bits: int
    codebook_storage_bits: int
    index_storage_bits: int
    scale_storage_bits: int
    zero_point_storage_bits: int
    total_storage_bits: int
    fp32_uncompressed_bits: int
    compression_ratio_vs_fp32_edges: float

def count_scale_values(num_clusters: int, coeffs_per_edge: int, granularity: str) -> int:
    if granularity == "per_tensor":
        return 1
    if granularity == "per_vector":
        return num_clusters
    if granularity == "per_channel":
        return coeffs_per_edge
    raise ValueError(f"Unknown granularity: {granularity}")

def estimate_layer_storage(
    layer_idx: int,
    codebook: torch.Tensor,
    cluster_ids: torch.Tensor,
    codebook_bits: int,
    index_bits: Optional[int],
    granularity: str,
    mode: str,
    scale_bits: int = 32,
    zero_point_bits: int = 8,
) -> LayerStorageReport:
    codebook_2d = codebook.reshape(codebook.shape[0], -1)
    num_clusters, coeffs_per_edge = codebook_2d.shape
    num_edges = cluster_ids.numel()

    if index_bits is None:
        index_bits = required_index_bits(num_clusters)

    codebook_storage_bits = num_clusters * coeffs_per_edge * codebook_bits
    index_storage_bits = num_edges * index_bits

    n_scales = count_scale_values(num_clusters, coeffs_per_edge, granularity)
    scale_storage_bits = n_scales * scale_bits
    zero_point_storage_bits = n_scales * zero_point_bits if mode == "asymmetric" else 0

    total = codebook_storage_bits + index_storage_bits + scale_storage_bits + zero_point_storage_bits
    fp32 = num_edges * coeffs_per_edge * 32

    return LayerStorageReport(
        layer=layer_idx,
        num_edges=num_edges,
        num_clusters=num_clusters,
        coeffs_per_edge=coeffs_per_edge,
        codebook_bits=codebook_bits,
        index_bits=index_bits,
        codebook_storage_bits=codebook_storage_bits,
        index_storage_bits=index_storage_bits,
        scale_storage_bits=scale_storage_bits,
        zero_point_storage_bits=zero_point_storage_bits,
        total_storage_bits=total,
        fp32_uncompressed_bits=fp32,
        compression_ratio_vs_fp32_edges=float(fp32 / max(total, 1)),
    )

def summarize_reports(reports):
    total_bits = sum(r.total_storage_bits for r in reports)
    total_fp32 = sum(r.fp32_uncompressed_bits for r in reports)
    return {
        "total_storage_bits": total_bits,
        "total_storage_bytes": total_bits / 8.0,
        "total_storage_kib": total_bits / 8.0 / 1024.0,
        "fp32_uncompressed_bits": total_fp32,
        "fp32_uncompressed_kib": total_fp32 / 8.0 / 1024.0,
        "compression_ratio_vs_fp32_edges": float(total_fp32 / max(total_bits, 1)),
        "layers": [r.__dict__ for r in reports],
    }
