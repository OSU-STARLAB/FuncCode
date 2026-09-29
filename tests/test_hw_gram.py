"""CI-fast tests for the GRAM HW extension (synthetic tiny models)."""

import math

import numpy as np
import torch

from funcodekan.hw import gram_fixed_point as gfp
from funcodekan.hw import gram_spec as gs
from funcodekan.hw.gram_qat import (QuantGramBranchKAN, QuantGramKAN,
                                    rebuild_gram_branch)
from funcodekan.compression import cross_variant as comp
from funcodekan.models.variants import DirectKANVariant

IN_DIM, WIDTH, OUT_DIM = 16, 8, 4


def test_isqrt64_matches_math_isqrt():
    vals = [0, 1, 2, 3, 4, 15, 16, 17, 10 ** 12, (1 << 62) - 1,
            687195, 123456789012345]
    x = np.array(vals, dtype=np.int64)
    got = gfp.isqrt64_np(x)
    exp = np.array([math.isqrt(v) for v in vals], dtype=np.int64)
    assert np.array_equal(got, exp)
    got_t = gs.isqrt64_torch(torch.tensor(vals, dtype=torch.int64))
    assert np.array_equal(got_t.numpy(), exp)


def test_int_layernorm_torch_numpy_identical():
    torch.manual_seed(0)
    v = torch.randint(-(1 << 20), 1 << 20, (32, 8), dtype=torch.int64)
    g = torch.randint(10000, 20000, (8,), dtype=torch.int64)
    b = torch.randint(-(1 << 20), 1 << 20, (8,), dtype=torch.int64)
    y_t = gs.int_layernorm_torch(v, g, b).numpy()
    y_n = gfp.int_layernorm_np(v.numpy(), g.numpy(), b.numpy())
    assert np.array_equal(y_t, y_n)


def test_int_layernorm_close_to_float_layernorm():
    torch.manual_seed(1)
    x = torch.randn(64, 8) * 3
    v = torch.floor(x * 4096 + 0.5).to(torch.int64)
    g = torch.full((8,), 1 << gs.GAMMA_FRAC, dtype=torch.int64)  # gamma=1
    b = torch.zeros(8, dtype=torch.int64)              # beta=0
    y_int = gs.int_layernorm_torch(v, g, b).float() * 2.0 ** -12
    y_ref = torch.nn.functional.layer_norm(x, (8,), eps=1e-5)
    assert torch.allclose(y_int, y_ref, atol=2e-3)


def test_silu_interp_int_torch_numpy_identical_and_accurate():
    y = torch.arange(-9 * 4096, 9 * 4096, 37, dtype=torch.int64)
    table = torch.from_numpy(gs.SILU_TABLE.copy())
    s_t = gs.silu_interp_int_torch(y, table).numpy()
    s_n = gfp.silu_interp_int_np(y.numpy())
    assert np.array_equal(s_t, s_n)
    # accuracy vs real silu inside the table domain
    yy = torch.arange(-8 * 4096, 8 * 4096 - 1, 13, dtype=torch.int64)
    s = gs.silu_interp_int_torch(yy, table).float() * 2.0 ** -10
    x = yy.float() * 2.0 ** -12
    ref = torch.nn.functional.silu(x)
    assert torch.allclose(s, ref, atol=3e-3)


def _tiny_dense(seed=0):
    torch.manual_seed(seed)
    return DirectKANVariant("gram", IN_DIM, WIDTH, OUT_DIM, degree=3)


def _init_wrapper(w):
    w.train()
    w(torch.randn(64, IN_DIM))   # init act quantizers
    return w.eval()


def test_g2_wrapper_eval_matches_emulator():
    w = _init_wrapper(QuantGramKAN(_tiny_dense()))
    model = gfp.build_from_g2(w)
    x = torch.randn(256, IN_DIM)
    with torch.no_grad():
        t_logits = w(x).numpy()          # float, real scale (exact ints/2^16)
    e_logits = model.forward_float(x.numpy()) * 2.0 ** -16
    agree = (t_logits.argmax(1) == e_logits.argmax(1)).mean()
    assert agree >= 0.98
    assert np.abs(t_logits - e_logits).max() < 2e-3


def test_g3_wrapper_and_rebuild_roundtrip():
    dense = _tiny_dense(3)
    branch = comp.build_compressed_kan(dense, "branch", clusters=4, seed=42,
                                       function_samples=16, base_clusters=2)
    sd = branch.state_dict()
    rebuilt = rebuild_gram_branch(sd, IN_DIM, WIDTH, OUT_DIM, 3)
    x = torch.randn(16, IN_DIM)
    with torch.no_grad():
        assert torch.allclose(branch(x), rebuilt(x), atol=1e-6)
    w = _init_wrapper(QuantGramBranchKAN(rebuilt))
    model = gfp.build_from_g3(w)
    for ql in model.layers:
        assert np.array_equal(ql.basis_q, ql.basis_codebook_q[ql.basis_ids])
    with torch.no_grad():
        t_logits = w(x).numpy()
    e_logits = model.forward_codes(model.quantize_input(x.numpy()))
    assert (t_logits.argmax(1) == e_logits.argmax(1)).mean() >= 0.9
    assert np.abs(t_logits - e_logits * 2.0 ** -16).max() < 2e-3


def test_g1_numpy_matches_torch_dense_gram():
    dense = _tiny_dense().eval()
    model = gfp.build_from_gram_dense(dense)
    x = torch.randn(256, IN_DIM)
    with torch.no_grad():
        t = dense(x).numpy()
    n = model.forward(x.numpy())
    assert np.max(np.abs(t - n)) <= 1e-4
    assert (t.argmax(1) == n.argmax(1)).all()


def test_gram_fake_quant_value_counts():
    w = _init_wrapper(QuantGramKAN(_tiny_dense()))
    for l in range(2):
        bw, base = w.quant_weights(l)
        assert torch.unique(bw).numel() <= 16
        assert torch.unique(base).numel() <= 16
