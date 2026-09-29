"""Regression tests for the convolutional KAGN / FuncCode path.

These lock down the three things that silently break and are hard to notice from
accuracy numbers alone: the edge view must be exactly invertible, the whitened
metric must equal the explicit function-space metric, and storage accounting
must never quietly drop a parameter.
"""

import math

import pytest
import torch

from funcodekan.compression.conv_baselines import (apply_magnitude_prune_,
                                                   apply_uniform_ptq_, attach_lsq_,
                                                   baseline_storage_bits)
from funcodekan.compression.conv_compression import (basis_matrix, cluster_layer,
                                                     compress_model, dense_storage_bits,
                                                     index_bits_for, quantize_codebooks_,
                                                     storage_breakdown, whiten_factor)
from funcodekan.models.conv_kagn import (BranchCodebookEdgeWeights, KAGNConv2d,
                                         KAGNLinear, SharedCodebookEdgeWeights,
                                         build_conv_kagn, compressible_layers,
                                         model_edge_bits, solve_iso_width)


# ---------------------------------------------------------------- edge view

@pytest.mark.parametrize("groups,degree,stride", [(1, 3, 1), (2, 3, 2), (4, 2, 1)])
def test_edge_roundtrip_is_exact(groups, degree, stride):
    torch.manual_seed(0)
    layer = KAGNConv2d(8, 16, 3, degree=degree, groups=groups, stride=stride, padding=1)
    x = torch.randn(3, 8, 12, 12)
    y0 = layer(x)

    em = layer.wprov.edge_matrix().detach()
    assert em.shape == (layer.wprov.n_edges, degree + 2)

    layer.wprov = SharedCodebookEdgeWeights(layer.wprov, em, torch.arange(em.shape[0]))
    assert torch.equal(y0, layer(x)), "identity codebook must reproduce the dense forward"


def test_branch_roundtrip_is_exact():
    torch.manual_seed(0)
    layer = KAGNLinear(10, 6, degree=3, final=True)
    x = torch.randn(4, 10)
    y0 = layer(x)
    em = layer.wprov.edge_matrix().detach()
    ids = torch.arange(em.shape[0])
    layer.wprov = BranchCodebookEdgeWeights(layer.wprov, em[:, :-1], ids, em[:, -1:], ids)
    assert torch.equal(y0, layer(x))


def test_edge_count_matches_geometry():
    layer = KAGNConv2d(12, 20, 3, groups=4)
    assert layer.wprov.n_edges == 20 * (12 // 4) * 3 * 3


# ------------------------------------------------------- whitening identity

def test_whitened_metric_equals_function_space_metric():
    """||L^T(w_a - w_b)||^2 == mean_s (phi_a(x_s) - phi_b(x_s))^2."""
    torch.manual_seed(0)
    B = basis_matrix(3, samples=128).float()
    W, _ = whiten_factor(B)
    w = torch.randn(64, 5)

    d_fn = torch.cdist(w @ B.T, w @ B.T) ** 2 / B.shape[0]
    d_wh = torch.cdist(w @ W, w @ W) ** 2
    assert torch.allclose(d_fn, d_wh, atol=1e-4, rtol=1e-3)


def test_whiten_and_signature_paths_agree():
    torch.manual_seed(0)
    layer = KAGNConv2d(8, 16, 3, degree=3)
    a = cluster_layer(layer, "l", "function", 8, seed=1, metric="whiten")
    b = cluster_layer(layer, "l", "function", 8, seed=1, metric="signature", normalize=False)
    assert (a.ids == b.ids).float().mean() > 0.99
    assert a.stats["fn_rel"] == pytest.approx(b.stats["fn_rel"], rel=1e-3)


def test_reconstruction_error_decreases_with_k():
    torch.manual_seed(0)
    layer = KAGNConv2d(8, 16, 3, degree=3)
    errs = [cluster_layer(layer, "l", "function", k, seed=0).stats["fn_rel"]
            for k in (4, 16, 64)]
    assert errs[0] > errs[1] > errs[2]


# ------------------------------------------------------------ storage rules

def test_storage_is_dominated_by_index_bits_at_scale():
    """The claim the paper leans on: codebooks amortise to nothing."""
    m = build_conv_kagn("kagn_simple_cifar10")
    m, _ = compress_model(m, "function", 32, seed=0, verbose=False)
    sb = storage_breakdown(m, codebook_bits=4)
    assert sb["index_bits_total"] > 20 * sb["codebook_bits_total"]


def test_bits_per_edge_matches_index_width():
    """Shared codebook at K must cost ceil(log2 K) bits per edge, plus overhead."""
    m = build_conv_kagn("kagn_simple_cifar10")
    n_edges = sum(l.wprov.n_edges for _, l in compressible_layers(m))
    for K in (16, 32, 256):
        mm = build_conv_kagn("kagn_simple_cifar10")
        mm, _ = compress_model(mm, "function", K, seed=0, verbose=False)
        sb = storage_breakdown(mm, codebook_bits=4)
        assert sb["index_bits_total"] == n_edges * index_bits_for(K)


def test_storage_accounting_loses_nothing():
    """Every parameter and every persistent buffer must be charged somewhere."""
    m = build_conv_kagn("kagn_simple_cifar10")
    m, _ = compress_model(m, "branch", 32, base_clusters=16, seed=0, verbose=False)
    sb = storage_breakdown(m, codebook_bits=4)

    counted_norm = sb["norm_bits"] / 16
    expected = sum(p.numel() for n, p in m.named_parameters()
                   if "codebook" not in n)
    expected += sum(b.numel() for n, b in m.named_buffers()
                    if n.endswith("running_mean") or n.endswith("running_var"))
    assert counted_norm == expected


def test_dense_storage_exceeds_every_compressed_variant():
    m = build_conv_kagn("kagn_simple_cifar10")
    dense = dense_storage_bits(m)
    for method, K in (("function", 32), ("branch", 32), ("coefficient", 16)):
        mm = build_conv_kagn("kagn_simple_cifar10")
        mm, _ = compress_model(mm, method, K, base_clusters=8, seed=0, verbose=False)
        assert storage_breakdown(mm, 4)["storage_total_bits"] < dense / 5


def test_baseline_storage_uses_same_conventions():
    """Uniform W4 must cost C*4 bits/edge -- 4x more than FuncCode K=32."""
    m = build_conv_kagn("kagn_simple_cifar10")
    n_edges = sum(l.wprov.n_edges for _, l in compressible_layers(m))
    sb = baseline_storage_bits(m, "uniform", bits=4)
    assert sb["weight_bits"] == n_edges * 5 * 4

    mm = build_conv_kagn("kagn_simple_cifar10")
    mm, _ = compress_model(mm, "function", 32, seed=0, verbose=False)
    fc = storage_breakdown(mm, codebook_bits=4)
    assert fc["storage_total_bits"] < sb["storage_total_bits"] / 3


# ------------------------------------------------------------------ smoke

def test_forward_shapes():
    for preset, ncls in (("kagn_simple_cifar10", 10),
                         ("kagn_simple_cifar100_8_layer_v2", 100)):
        m = build_conv_kagn(preset)
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, ncls)


def test_quantize_codebooks_changes_nothing_structural():
    m = build_conv_kagn("kagn_simple_cifar10")
    m, _ = compress_model(m, "function", 32, seed=0, verbose=False)
    before = storage_breakdown(m, 4)["storage_total_bits"]
    quantize_codebooks_(m, 4)
    assert storage_breakdown(m, 4)["storage_total_bits"] == before
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_lsq_is_differentiable():
    m = build_conv_kagn("kagn_tiny_smoke")
    attach_lsq_(m, 4)
    loss = m(torch.randn(2, 3, 32, 32)).square().mean()
    loss.backward()
    steps = [p for n, p in m.named_parameters() if "step_" in n]
    assert steps and all(p.grad is not None and torch.isfinite(p.grad).all() for p in steps)


def test_uniform_ptq_and_prune_run():
    m = build_conv_kagn("kagn_tiny_smoke")
    apply_uniform_ptq_(m, 4)
    assert torch.isfinite(m(torch.randn(2, 3, 32, 32))).all()
    m2, stats = apply_magnitude_prune_(build_conv_kagn("kagn_tiny_smoke"), 0.9, bits=4)
    assert 0.85 < stats["actual_sparsity"] < 0.95


def test_iso_width_solver_hits_target():
    target = 5 * 100_000
    scale = solve_iso_width("kagn_simple_cifar10", target, 10)
    got = model_edge_bits(build_conv_kagn("kagn_simple_cifar10", 10, width_scale=scale))
    assert abs(got - target) / target < 0.15


def test_compression_package_imports_standalone():
    """Regression: importing compression before models used to raise ImportError."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", "import funcodekan.compression"],
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
