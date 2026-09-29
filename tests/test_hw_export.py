"""M3 acceptance tests (CI-fast subset): golden export integrity.

Uses a tiny synthetic 10-class dataset (100 images per class) so the
stratified N=1000 export path runs end-to-end without downloads.
"""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from funcodekan.hw import export as hw_export
from funcodekan.hw import fixed_point as fp
from funcodekan.hw.qat import QuantSplineKAN
from funcodekan.models.spline import DenseSplineKAN

IN_DIM, WIDTH, OUT_DIM, GRID, ORDER = 16, 8, 4, 5, 3


def _fake_test_loader(n_per_class=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(10 * n_per_class, IN_DIM, generator=g)
    y = torch.arange(10).repeat_interleave(n_per_class)
    perm = torch.randperm(len(y), generator=g)
    return DataLoader(TensorDataset(x[perm], y[perm]), batch_size=256)


def _wrapper(seed=0):
    torch.manual_seed(seed)
    w = QuantSplineKAN(DenseSplineKAN(IN_DIM, WIDTH, OUT_DIM, GRID, ORDER))
    w.train()
    w(torch.randn(64, IN_DIM))
    return w.eval()


def test_stratified_indices_deterministic_and_balanced():
    labels = np.tile(np.arange(10), 150)
    idx = hw_export.stratified_indices(labels, per_class=100)
    assert len(idx) == 1000
    counts = np.bincount(labels[idx], minlength=10)
    assert (counts == 100).all()
    assert np.array_equal(idx, hw_export.stratified_indices(labels, 100))


def test_c_array_formatting():
    s = hw_export.c_array("foo", "w4_t", np.array([[1, -2], [3, 4]]))
    assert "static const w4_t foo[4]" in s
    assert "1, -2, 3, 4" in s
    f = hw_export.c_array("bar", "float", np.array([0.1], dtype=np.float32),
                          fmt=hw_export._fmt_f32)
    assert f.endswith("};\n") and "f" in f
    # float32 exact round-trip
    assert float(np.float32("0.100000001")) == float(np.float32(0.1))


def test_export_d2_artifacts_and_manifest(tmp_path):
    w = _wrapper()
    loader = _fake_test_loader()
    manifest = hw_export.export_d2(
        w, loader, out_dir=tmp_path / "lsq_w4a4",
        log_path=tmp_path / "verification_log.json")
    out = tmp_path / "lsq_w4a4"
    for name in ["params.h", "golden.h", "golden_inputs.npy",
                 "golden_outputs.npy", "golden_labels.npy", "manifest.json"]:
        assert (out / name).exists()
    # SHA256 in the manifest matches the artifacts on disk
    for name, digest in manifest["sha256"].items():
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert actual == digest, name
    # golden outputs reproduce from golden inputs through the emulator
    model = fp.build_from_d2(w)
    gin = np.load(out / "golden_inputs.npy")
    gout = np.load(out / "golden_outputs.npy")
    assert gin.dtype == np.int8 and gout.dtype == np.int32
    assert gin.shape == (1000, IN_DIM) and gout.shape == (1000, OUT_DIM)
    assert gin.min() >= -8 and gin.max() <= 7
    recomputed = model.forward_codes(gin.astype(np.int64))
    assert np.array_equal(recomputed, gout)
    # params.h sanity
    params = (out / "params.h").read_text()
    assert f"#define L0_IN {IN_DIM}" in params
    assert "l0_lut_b" in params and "l1_mult_spline" in params
    assert "static_assert" in params
    # verification log went to the test-local path, entries recorded
    log = json.loads((tmp_path / "verification_log.json").read_text())
    assert any(e["rung"] == "L1" for e in log)


def test_export_d1_artifacts(tmp_path):
    torch.manual_seed(0)
    dense = DenseSplineKAN(IN_DIM, WIDTH, OUT_DIM, GRID, ORDER).eval()
    loader = _fake_test_loader()
    manifest = hw_export.export_d1(
        dense, loader, out_dir=tmp_path / "fp32",
        log_path=tmp_path / "verification_log.json")
    assert manifest["l1"]["passed"]
    gin = np.load(tmp_path / "fp32" / "golden_inputs.npy")
    gout = np.load(tmp_path / "fp32" / "golden_outputs.npy")
    assert gin.dtype == np.float32 and gout.dtype == np.float32
    model = fp.build_from_dense(dense)
    assert np.max(np.abs(model.forward(gin) - gout)) == 0.0
    params = (tmp_path / "fp32" / "params.h").read_text()
    assert "l0_weight" in params and "l0_grid" in params
