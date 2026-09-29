"""M3 acceptance tests (CI-fast subset): fixed-point emulator numerics.

Synthetic tiny models only — the full 10k L1 runs live in
funcodekan.hw.export/verify and are gated by scripts/hw/11_export_golden.sh.
"""

import numpy as np
import torch

from funcodekan.hw import fixed_point as fp
from funcodekan.hw.act_quant import QuantBranchKAN
from funcodekan.hw.hw_spec import (BASIS_SCALE, SILU_SCALE,
                                   lut_quantize_basis, lut_quantize_silu)
from funcodekan.hw.qat import QuantSplineKAN
from funcodekan.models.spline import (BranchAwareClusteredSplineKAN,
                                      DenseSplineKAN)

IN_DIM, WIDTH, OUT_DIM, GRID, ORDER = 16, 8, 4, 5, 3


def _trained_wrapper(seed=0):
    torch.manual_seed(seed)
    dense = DenseSplineKAN(IN_DIM, WIDTH, OUT_DIM, GRID, ORDER)
    w = QuantSplineKAN(dense)
    w.train()
    w(torch.randn(64, IN_DIM))  # init act quantizers
    return w.eval()


def test_emulator_luts_match_qat_lut_rounding():
    w = _trained_wrapper()
    model = fp.build_from_d2(w)
    for l, layer in enumerate(w.dense.layers):
        step = w.act_quants[l].step_size()
        q = torch.arange(-8, 8, dtype=torch.float32)
        x = q * torch.tensor(step, dtype=torch.float32)
        xx = x.unsqueeze(1).expand(16, layer.in_features).contiguous()
        basis_ref = lut_quantize_basis(layer.b_splines(xx)[:, 0, :])
        silu_ref = lut_quantize_silu(torch.nn.functional.silu(x))
        assert np.array_equal(model.layers[l].lut_b,
                              (basis_ref * BASIS_SCALE).numpy().astype(np.int64))
        assert np.array_equal(model.layers[l].lut_s,
                              (silu_ref * SILU_SCALE).numpy().astype(np.int64))


def test_emulator_input_codes_match_lsq_quantize_int():
    w = _trained_wrapper()
    model = fp.build_from_d2(w)
    x = torch.randn(256, IN_DIM)
    torch_codes = w.act_quants[0].quantize_int(x).numpy()
    emu_codes = model.quantize_input(x.numpy())
    assert np.array_equal(torch_codes, emu_codes)


def test_emulator_agrees_with_fake_quant_model():
    w = _trained_wrapper()
    model = fp.build_from_d2(w)
    x = torch.randn(512, IN_DIM)
    with torch.no_grad():
        t_logits = w(x).numpy()
    e_logits = model.forward_float(x.numpy())
    agree = (t_logits.argmax(1) == e_logits.argmax(1)).mean()
    assert agree >= 0.995
    # int32 logits at scale 2^-16 track the float logits closely
    assert np.allclose(e_logits * 2.0 ** -16, t_logits, atol=1e-3)


def test_quant_layer_integer_arithmetic_by_hand():
    # 1 input, 1 output, spline codes all zero except k=0; base weight 3.
    lut_b = np.zeros((16, 8), dtype=np.int64)
    lut_b[:, 0] = np.arange(16) * 100
    lut_s = np.arange(16, dtype=np.int64) * 10 - 40
    spline_q = np.zeros((1, 1, 8), dtype=np.int64)
    spline_q[0, 0, 0] = 2
    base_q = np.full((1, 1), 3, dtype=np.int64)
    ql = fp.QuantLayer(lut_b=lut_b, lut_s=lut_s, spline_q=spline_q,
                       base_q=base_q, mult_spline=1 << 20, mult_base=1 << 21,
                       shift=24, is_output=True)
    q = np.array([[5]], dtype=np.int64)          # u = 13
    # acc_s = 2*1300 = 2600 ; acc_b = 3*(130-40) = 270
    # t = 2600*2^20 + 270*2^21 = (2600 + 540) * 2^20 = 3140*2^20
    # logits = (3140*2^20 + 2^23) >> 24 = floor(3140/16 + 0.5) = 196.75+0.5 -> 196... check
    t = 3140 * (1 << 20) + (1 << 23)
    expected = t >> 24
    out = ql.forward_codes(q)
    assert out.shape == (1, 1)
    assert out[0, 0] == expected


def test_quant_layer_hidden_clamps_to_int4():
    lut_b = np.zeros((16, 8), dtype=np.int64)
    lut_s = np.full(16, SILU_SCALE, dtype=np.int64)
    ql = fp.QuantLayer(lut_b=lut_b, lut_s=lut_s,
                       spline_q=np.zeros((1, 1, 8), dtype=np.int64),
                       base_q=np.full((1, 1), 7, dtype=np.int64),
                       mult_spline=1, mult_base=1 << 30, shift=30,
                       is_output=False)
    out = ql.forward_codes(np.array([[0]], dtype=np.int64))
    assert out[0, 0] == 7  # 7*SILU_SCALE >> 0 clamped to QP


def test_d3_emulator_matches_codebook_lookup_and_d2_path():
    torch.manual_seed(3)
    ks, kb = 6, 4
    widths = [(WIDTH, IN_DIM), (OUT_DIM, WIDTH)]
    # 0.1x codebooks keep synthetic hidden activations inside the Q4.12 LUT
    # range (real trained act steps are far smaller than the 8/7 bound)
    branch = BranchAwareClusteredSplineKAN(
        IN_DIM, WIDTH, OUT_DIM, GRID, ORDER,
        spline_codebooks=[torch.randn(ks, GRID + ORDER) * 0.1 for _ in widths],
        spline_cluster_ids=[torch.randint(0, ks, s) for s in widths],
        base_codebooks=[torch.randn(kb) * 0.1 for _ in widths],
        base_cluster_ids=[torch.randint(0, kb, s) for s in widths],
    )
    w = QuantBranchKAN(branch)
    w.train()
    w(torch.randn(64, IN_DIM))
    w.eval()
    model = fp.build_from_d3(w)
    for ql in model.layers:
        assert np.array_equal(ql.spline_q,
                              ql.spline_codebook_q[ql.spline_ids])
        assert np.array_equal(ql.base_q, ql.base_codebook_q[ql.base_ids])
    x = torch.randn(256, IN_DIM)
    with torch.no_grad():
        t_logits = w(x).numpy()
    e_logits = model.forward_float(x.numpy())
    assert (t_logits.argmax(1) == e_logits.argmax(1)).mean() >= 0.99


def test_fp32_numpy_path_matches_torch_dense():
    torch.manual_seed(0)
    dense = DenseSplineKAN(IN_DIM, WIDTH, OUT_DIM, GRID, ORDER).eval()
    model = fp.build_from_dense(dense)
    x = torch.randn(256, IN_DIM)
    with torch.no_grad():
        t_logits = dense(x).numpy()
    n_logits = model.forward(x.numpy())
    assert np.max(np.abs(t_logits - n_logits)) <= 1e-4
    assert (t_logits.argmax(1) == n_logits.argmax(1)).all()
