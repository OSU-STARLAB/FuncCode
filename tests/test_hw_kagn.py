"""CI-fast tests for the KAGN-Conv HW extension (synthetic tiny models)."""

import numpy as np
import torch

from funcodekan.hw import kagn_fixed_point as kfp
from funcodekan.hw.kagn_conv import (CompressedKagnConvNet, KagnConvNet,
                                     gram_basis)
from funcodekan.hw.kagn_qat import QuantKagnBranchNet, QuantKagnConvNet
from funcodekan.analysis.storage import compressed_storage_breakdown


def _init(w, steps=3):
    w.train()
    for _ in range(steps):
        w(torch.randn(16, 1, 28, 28) * 0.5)
    return w.eval()


def test_gram_basis_matches_fc_layer_basis():
    from funcodekan.models.variants import GramPolynomialLayer
    layer = GramPolynomialLayer(4, 4, degree=3)
    x = torch.randn(32, 4)
    assert torch.allclose(gram_basis(x, 3), layer.basis(x), atol=1e-7)


def test_dense_forward_and_storage():
    torch.manual_seed(0)
    net = KagnConvNet(channels=(4, 8))
    y = net(torch.randn(2, 1, 28, 28))
    assert y.shape == (2, 10) and torch.isfinite(y).all()
    assert net.dense_storage_bits() > 0


def test_compressed_roundtrip_and_storage_dispatch():
    torch.manual_seed(0)
    net = KagnConvNet(channels=(4, 8))
    comp = CompressedKagnConvNet(net, ks=6, kb=4, seed=1)
    y = comp(torch.randn(2, 1, 28, 28))
    assert y.shape == (2, 10) and torch.isfinite(y).all()
    bd = compressed_storage_breakdown(comp, 4, 32)
    assert bd["storage_total_bits"] > 0
    # state_dict roundtrip through a freshly-clustered instance
    sd = comp.state_dict()
    net2 = KagnConvNet(channels=(4, 8))
    comp2 = CompressedKagnConvNet(net2, ks=6, kb=4, seed=2)
    comp2.load_state_dict(sd)
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        assert torch.allclose(comp(x), comp2(x), atol=1e-6)
    # indices are frozen buffers
    names = {n for n, p in comp.named_parameters() if p.requires_grad}
    assert not any("cluster_ids" in n for n in names)


def test_k2_wrapper_eval_exactly_matches_emulator():
    torch.manual_seed(0)
    w = _init(QuantKagnConvNet(KagnConvNet(channels=(4, 8))))
    model = kfp.build_from_kagn(w, is_k3=False)
    x = torch.randn(64, 1, 28, 28) * 0.5
    with torch.no_grad():
        t = w(x).numpy()
    e = model.forward_float(x.numpy()) * 2.0 ** -16
    assert np.abs(t - e).max() == 0.0


def test_k3_wrapper_eval_exactly_matches_emulator():
    torch.manual_seed(3)
    net = KagnConvNet(channels=(4, 8))
    comp = CompressedKagnConvNet(net, ks=6, kb=4, seed=1)
    w = _init(QuantKagnBranchNet(comp))
    model = kfp.build_from_kagn(w, is_k3=True)
    for ql in model.conv_layers:
        assert np.array_equal(ql.poly_q, ql.poly_codebook_q[ql.poly_ids])
    x = torch.randn(64, 1, 28, 28) * 0.5
    with torch.no_grad():
        t = w(x).numpy()
    e = model.forward_float(x.numpy()) * 2.0 ** -16
    assert np.abs(t - e).max() == 0.0


def test_k1_numpy_close_to_torch_dense():
    torch.manual_seed(0)
    net = KagnConvNet(channels=(4, 8))
    net.train()
    for _ in range(3):
        net(torch.randn(16, 1, 28, 28) * 0.5)   # populate BN stats
    net.eval()
    model = kfp.build_from_kagn_dense(net)
    x = torch.randn(64, 1, 28, 28) * 0.5
    with torch.no_grad():
        t = net(x).numpy()
    n = model.forward(x.numpy())
    assert np.max(np.abs(t - n)) <= 1e-3
    assert (t.argmax(1) == n.argmax(1)).all()


def test_int_instance_norm_close_to_float():
    import funcodekan.hw.gram_spec as gs
    torch.manual_seed(0)
    x = torch.randn(8, 6, 7, 7) * 2
    v = torch.floor(x * 4096 + 0.5).to(torch.int64).reshape(8, 6, 49)
    g = torch.full((6, 1), 1 << gs.GAMMA_FRAC, dtype=torch.int64)
    b = torch.zeros((6, 1), dtype=torch.int64)
    y_int = (gs.int_layernorm_torch(v, g, b).float() * 2.0 ** -12
             ).reshape(8, 6, 7, 7)
    y_ref = torch.nn.functional.instance_norm(x, eps=1e-5)
    assert torch.allclose(y_int, y_ref, atol=3e-3)
