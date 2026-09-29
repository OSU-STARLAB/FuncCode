"""End-to-end smoke test on synthetic data (no dataset downloads).

Exercises every compression path of the framework on tiny models:
SplineKAN cluster methods (coefficient, function, branch, branch_index,
branch_residual), HWQ quantization + storage accounting, cross-variant
compression (spline/fast/gram + MLP), variant ablations (SRB, index-
efficient, soft-to-hard), and the soft-codebook builder.

These tests verify the package wiring, not paper numbers.
"""

import torch

from funcodekan.models.spline import DenseSplineKAN
from funcodekan.models.variants import DirectKANVariant, MLPBaseline
from funcodekan.models.soft_codebook import build_soft_index_branch_from_dense
from funcodekan.models.ablations import (
    build_index_efficient_from_dense,
    build_srb_from_dense,
    build_soft_to_hard_from_dense,
    quantize_variant_ablation_model,
    variant_ablation_storage_breakdown,
)
from funcodekan.compression.clustering import build_clustered_from_dense
from funcodekan.compression import cross_variant as comp
from funcodekan.quantization.hwq_pipeline import quantize_clustered_model
from funcodekan.analysis.storage import (
    dense_fp32_bits_from_model,
    clustered_fp32_bits_from_model,
    compression_ratio,
)

IN_DIM, WIDTH, OUT_DIM = 16, 8, 4
GRID, ORDER = 5, 3


def _dense_spline():
    torch.manual_seed(0)
    return DenseSplineKAN(IN_DIM, WIDTH, OUT_DIM, GRID, ORDER)


def _forward_ok(model):
    x = torch.randn(6, IN_DIM)
    y = model(x)
    assert y.shape == (6, OUT_DIM)
    assert torch.isfinite(y).all()


def test_dense_spline_forward():
    _forward_ok(_dense_spline())


def test_spline_cluster_methods_and_hwq():
    dense = _dense_spline()
    for method in ["coefficient", "function", "branch", "branch_index", "branch_residual"]:
        clustered = build_clustered_from_dense(
            dense, IN_DIM, WIDTH, OUT_DIM, GRID, ORDER,
            num_clusters=4, seed=42, cluster_method=method,
            function_samples=16, function_domain="grid",
            branch_spline_clusters=4, branch_base_clusters=2,
            branch_function_samples=16,
            residual_fraction=0.25, residual_base_clusters=2,
        )
        _forward_ok(clustered)
        d_bits = dense_fp32_bits_from_model(dense)
        c_bits = clustered_fp32_bits_from_model(clustered)
        assert compression_ratio(d_bits, c_bits) > 1.0
        q = quantize_clustered_model(clustered, 4, IN_DIM, WIDTH, OUT_DIM, GRID, ORDER)
        _forward_ok(q["model"] if isinstance(q, dict) and "model" in q else clustered)


def test_cross_variant_compression():
    for variant in ["spline", "fast", "gram"]:
        torch.manual_seed(0)
        dense = DirectKANVariant(variant, IN_DIM, WIDTH, OUT_DIM, grid_size=GRID, spline_order=ORDER)
        _forward_ok(dense)
        # coefficient, function-space, branch-aware
        m_coeff = comp.build_compressed_kan(dense, "coefficient", clusters=4, seed=42, function_samples=16)
        _forward_ok(m_coeff)
        m_func = comp.build_compressed_kan(dense, "function", clusters=4, seed=42, function_samples=16)
        _forward_ok(m_func)
        m_branch = comp.build_compressed_kan(dense, "branch", clusters=4, seed=42, function_samples=16, base_clusters=2)
        _forward_ok(m_branch)
        # W4 quantization of compressed models
        comp.quantize_compressed_kan(m_func, 4)
        comp.quantize_compressed_kan(m_branch, 4)
        _forward_ok(m_func)
        _forward_ok(m_branch)
        assert comp.compressed_storage_breakdown(m_branch, codebook_bits=4)["storage_total_bits"] > 0
    mlp = MLPBaseline(IN_DIM, WIDTH, OUT_DIM)
    _forward_ok(mlp)
    comp.quantize_mlp_copy(mlp, 4)


def test_variant_ablations():
    for variant in ["spline", "fast", "gram"]:
        torch.manual_seed(0)
        dense = DirectKANVariant(variant, IN_DIM, WIDTH, OUT_DIM, grid_size=GRID, spline_order=ORDER)
        idx = build_index_efficient_from_dense(dense, clusters=4, seed=42, function_samples=16)
        _forward_ok(idx)
        srb = build_srb_from_dense(
            dense, clusters=4, residual_fraction=0.25,
            residual_clusters=2, seed=42, function_samples=16,
        )
        _forward_ok(srb)
        soft = build_soft_to_hard_from_dense(dense, clusters=4, seed=42, function_samples=16)
        _forward_ok(soft)
        hard = soft.harden()
        _forward_ok(hard)
        quantize_variant_ablation_model(idx, 4)
        _forward_ok(idx)
        assert variant_ablation_storage_breakdown(srb, codebook_bits=4)["storage_total_bits"] > 0


def test_soft_codebook_builder():
    dense = _dense_spline()
    soft = build_soft_index_branch_from_dense(
        dense, IN_DIM, WIDTH, OUT_DIM, GRID, ORDER,
        spline_clusters=4, seed=42,
        spline_method="function", function_samples=16, function_domain="grid",
    )
    _forward_ok(soft)


if __name__ == "__main__":
    test_dense_spline_forward()
    test_spline_cluster_methods_and_hwq()
    test_cross_variant_compression()
    test_variant_ablations()
    test_soft_codebook_builder()
    print("ALL SMOKE TESTS PASSED")
