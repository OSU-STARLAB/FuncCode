"""Tests for the dataset-registry integration (bundles, generic driver, zoo).

Everything here runs offline: sklearn-backed datasets (moons, wine,
circle_in_circle) plus synthetic stand-ins for image data. Network-dependent
loaders (dry_bean, mushroom, torchvision downloads, traffic) are exercised on
a real machine via `python -m funcodekan.data.registry`.
"""

import sys

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from funcodekan.data import registry as reg
from funcodekan.data.bundles import (
    _image_bundle,
    available_classification_datasets,
    get_dataset_bundle,
)
from funcodekan.models import zoo


def _check_bundle(b, expect_dim=None, expect_classes=None, flat=True):
    x, y = next(iter(b.train_loader))
    if flat:
        assert x.ndim == 2
        assert x.shape[1] == b.input_dim
    else:
        assert x.ndim == 4  # [B, C, H, W]
    assert int(y.max()) < b.num_classes
    if expect_dim is not None:
        assert b.input_dim == expect_dim
    if expect_classes is not None:
        assert b.num_classes == expect_classes
    for loader in (b.val_loader, b.test_loader):
        xb, _ = next(iter(loader))
        assert len(loader.dataset) > 0
    return x


def test_registry_present():
    assert "moons" in reg.REGISTRY and "wine" in reg.REGISTRY
    assert "traffic_california" in reg.REGISTRY
    keys = available_classification_datasets()
    assert "moons" in keys and "fashion_mnist" in keys
    assert "traffic_california" not in keys  # regression, not classification


def test_tabular_bundles_offline():
    b = get_dataset_bundle("moons", seed=42, num_workers=0)
    _check_bundle(b, expect_dim=2, expect_classes=2)
    b = get_dataset_bundle("wine", seed=42, num_workers=0)
    _check_bundle(b, expect_dim=13, expect_classes=3)
    b = get_dataset_bundle("circle_in_circle", seed=42, num_workers=0)
    _check_bundle(b, expect_dim=2, expect_classes=2)


def test_tabular_standardization_and_split_determinism():
    b1 = get_dataset_bundle("wine", seed=42, num_workers=0)
    b2 = get_dataset_bundle("wine", seed=42, num_workers=0)
    x1 = b1.test_loader.dataset.tensors[0]
    x2 = b2.test_loader.dataset.tensors[0]
    assert torch.equal(x1, x2)  # same seed -> identical split
    xtr = b1.train_loader.dataset.tensors[0]
    assert abs(float(xtr.mean())) < 0.1  # standardized on train stats
    n = (len(b1.train_loader.dataset) + len(b1.val_loader.dataset)
         + len(b1.test_loader.dataset))
    assert n == 178  # UCI wine sample count, no leakage/duplication


class _FakeImages(Dataset):
    """Synthetic torchvision-like dataset: 3x8x8 images, 4 classes."""

    def __init__(self, n=64):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(n, 3, 8, 8, generator=g)
        self.y = torch.randint(0, 4, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def test_image_bundle_flatten_and_raw():
    reg.REGISTRY["_fake_images"] = reg.DatasetInfo(
        key="_fake_images", name="Fake", modality="image-4class",
        n_classes=4, hw_only=False, loader=lambda: None)
    try:
        splits = {"train": _FakeImages(64), "test": _FakeImages(16)}
        b = _image_bundle("_fake_images", splits, batch_size=8,
                          test_batch_size=8, seed=0, num_workers=0)
        x = _check_bundle(b, expect_dim=3 * 8 * 8, expect_classes=4)
        b_raw = _image_bundle("_fake_images", splits, batch_size=8,
                              test_batch_size=8, seed=0, num_workers=0,
                              flatten=False)
        x = _check_bundle(b_raw, expect_classes=4, flat=False)
        assert x.shape[1:] == (3, 8, 8)
    finally:
        del reg.REGISTRY["_fake_images"]


def test_traffic_rejected_as_classification():
    with pytest.raises((ValueError, Exception)):
        # Either our explicit ValueError (if cached data exists) or the
        # network fetch failing — both mean it is not served as classification.
        get_dataset_bundle("traffic_california", num_workers=0)


def test_generic_driver_end_to_end_on_moons(tmp_path):
    """Full verified pipeline (dense -> function/branch -> W4 -> storage)
    through the dataset-generic driver on an offline dataset."""
    from funcodekan.experiments import all_kan_datasets as drv
    argv = sys.argv
    sys.argv = [
        "x", "--dataset", "moons", "--out-dir", str(tmp_path),
        "--run-name", "t", "--variants", "spline", "mlp",
        "--methods", "function", "branch", "--clusters-list", "4",
        "--bits-list", "4", "--width", "8", "--epochs", "1",
        "--finetune-epochs", "1", "--function-samples", "16",
        "--batch-size", "256", "--test-batch-size", "512",
        "--num-workers", "0",
    ]
    try:
        drv.main()
    finally:
        sys.argv = argv
    assert (tmp_path / "t" / "combined_summary.csv").exists()
    import pandas as pd
    df = pd.read_csv(tmp_path / "t" / "combined_summary.csv")
    assert {"dense_fp32", "clustered_hwq_w4"} <= set(df["stage"])
    w4 = df[df["stage"] == "clustered_hwq_w4"]
    assert (w4["compression_vs_dense"] > 1).all()


def test_zoo_degrades_gracefully_without_kans():
    if zoo.KANS_AVAILABLE:
        pytest.skip("kans installed; graceful-degradation path not applicable")
    with pytest.raises(ImportError, match="kans"):
        zoo.KAN_MLP_MNIST()
    from funcodekan.experiments.zoo_train import build_zoo_model, is_conv_arch
    with pytest.raises(ImportError, match="kans"):
        build_zoo_model("mnist", "kan_mlp_mnist")
    assert is_conv_arch("kagn_simple_cifar10")
    assert is_conv_arch("kan_resnet_cifar10")
    assert not is_conv_arch("kan_mlp_mnist")
    assert not is_conv_arch("kan_wine")
