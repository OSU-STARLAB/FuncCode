"""M1 acceptance tests: LSQ quantizer numerics + fake-quant value counts.

CPU-only, synthetic tensors, no downloads (CI-fast).
"""

import math

import pytest
import torch

from funcodekan.hw.hw_spec import (BASIS_SCALE, SILU_SCALE,
                                   requant_multiplier, round_half_up)
from funcodekan.hw.lsq import LsqQuantizer
from funcodekan.hw.qat import QuantSplineKAN
from funcodekan.hw.act_quant import QuantBranchKAN
from funcodekan.models.spline import (BranchAwareClusteredSplineKAN,
                                      DenseSplineKAN)

IN_DIM, WIDTH, OUT_DIM, GRID, ORDER = 16, 8, 4, 5, 3


def _distinct(t: torch.Tensor) -> int:
    return int(torch.unique(t).numel())


def test_round_half_up_is_hw_rounding():
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49, -0.49])
    expected = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0, -2.0, 0.0, 0.0])
    assert torch.equal(round_half_up(x), expected)


def test_lsq_step_init_formula():
    torch.manual_seed(0)
    x = torch.randn(1000)
    q = LsqQuantizer(bits=4, signed=True)
    q.init_from(x)
    expected = 2.0 * x.abs().mean() / math.sqrt(7)
    assert torch.allclose(q.s.detach(), expected)


def test_lsq_forward_on_grid_and_16_levels():
    torch.manual_seed(0)
    x = torch.randn(4096) * 2
    q = LsqQuantizer(bits=4, signed=True)
    q.train()
    y = q(x)
    s = q.step_size()
    codes = y / s
    assert torch.allclose(codes, codes.round(), atol=1e-5)
    assert codes.min() >= -8 and codes.max() <= 7
    assert _distinct(y) <= 16
    # eval path identical values
    q.eval()
    assert torch.equal(q(x), y)


def test_lsq_int_codes_match_fake_quant():
    torch.manual_seed(1)
    x = torch.randn(512)
    q = LsqQuantizer(bits=4, signed=True)
    q.init_from(x)
    y = q(x)
    codes = q.quantize_int(x)
    assert codes.dtype == torch.int32
    assert torch.allclose(codes.float() * q.step_size(), y, atol=1e-6)


def test_lsq_backward_ste_wrt_input():
    q = LsqQuantizer(bits=4, signed=True)
    q.init_from(torch.tensor([1.0]))
    with torch.no_grad():
        q.s.fill_(1.0)
    x = torch.tensor([0.3, 3.2, 100.0, -100.0], requires_grad=True)
    y = q(x)
    y.sum().backward()
    # STE: unit gradient inside the clamp range, zero outside
    assert torch.allclose(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_lsq_backward_wrt_step_matches_paper_formula():
    q = LsqQuantizer(bits=4, signed=True)
    q.init_from(torch.tensor([1.0]))
    with torch.no_grad():
        q.s.fill_(1.0)
    x = torch.tensor([0.3, 3.2, 100.0])
    xd = x.clone().requires_grad_(False)
    y = q(xd)
    y.sum().backward()
    g = 1.0 / math.sqrt(x.numel() * 7)
    # in-range: g*(round(x/s) - x/s); clamped: g*Qp
    expected = g * ((round_half_up(torch.tensor([0.3])) - 0.3)
                    + (round_half_up(torch.tensor([3.2])) - 3.2)
                    + 7.0)
    assert torch.allclose(q.s.grad, expected.squeeze(), atol=1e-5)


def test_lsq_eval_before_init_raises():
    q = LsqQuantizer(bits=4, signed=True)
    q.eval()
    with pytest.raises(RuntimeError):
        q(torch.randn(4))


def _dense():
    torch.manual_seed(0)
    return DenseSplineKAN(IN_DIM, WIDTH, OUT_DIM, GRID, ORDER)


def test_quant_spline_kan_fake_quant_tensors_have_16_levels():
    model = QuantSplineKAN(_dense())
    model.train()
    x = torch.randn(32, IN_DIM)
    y = model(x)
    assert y.shape == (32, OUT_DIM) and torch.isfinite(y).all()
    for l in range(2):
        sw, bw = model.quant_weights(l)
        assert _distinct(sw) <= 16
        assert _distinct(bw) <= 16
    for aq in model.act_quants:
        assert _distinct(aq(torch.randn(64, 8))) <= 16


def test_quant_branch_kan_fake_quant_tensors_have_16_levels():
    torch.manual_seed(0)
    ks, kb = 6, 4
    widths = [(WIDTH, IN_DIM), (OUT_DIM, WIDTH)]
    branch = BranchAwareClusteredSplineKAN(
        IN_DIM, WIDTH, OUT_DIM, GRID, ORDER,
        spline_codebooks=[torch.randn(ks, GRID + ORDER) for _ in widths],
        spline_cluster_ids=[torch.randint(0, ks, w) for w in widths],
        base_codebooks=[torch.randn(kb) for _ in widths],
        base_cluster_ids=[torch.randint(0, kb, w) for w in widths],
    )
    model = QuantBranchKAN(branch)
    model.train()
    y = model(torch.randn(32, IN_DIM))
    assert y.shape == (32, OUT_DIM) and torch.isfinite(y).all()
    for l in range(2):
        sw, bw = model.quant_weights(l)
        # expanded weights come from a quantized codebook: <= K distinct rows,
        # every value on the INT4 grid
        step = model.cb_spline_quants[l].step_size()
        codes = sw / step
        assert torch.allclose(codes, codes.round(), atol=1e-5)
        assert _distinct(sw) <= 16
        assert _distinct(bw) <= 16
    # indices are frozen buffers (never trained)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any("cluster_ids" in n for n in trainable)


def test_quant_wrappers_train_only_intended_parameters():
    model = QuantSplineKAN(_dense())
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("dense.weights") for n in names)
    assert any(".s" in n for n in names)


def test_requant_multiplier_fits_int32_and_reconstructs():
    ratios = [3.1e-6, 7.7e-7]
    mults, shift = requant_multiplier(ratios)
    for m, r in zip(mults, ratios):
        assert 0 < m <= 2**31 - 1
        assert abs(m / (1 << shift) - r) / r < 1e-6
    # the larger ratio uses most of the int32 range (precision)
    assert max(mults) > 2**30


def test_lut_scale_constants():
    assert BASIS_SCALE == 1 << 14
    assert SILU_SCALE == 1 << 10
