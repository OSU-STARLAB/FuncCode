"""Self-contained KAGN (Gram-polynomial KAN) convolutional models, wired for
FuncCode edge-function compression.

Why this file exists
--------------------
``funcodekan.models.zoo`` already declares ``SimpleConvKAGN`` and
``EightSimpleConvKAGN``, but they import ``KAGNConv2DLayer`` from the external
``kans`` package, which is not vendored here, and -- more importantly -- those
layers store their parameters in a layout that FuncCode's edge clustering
cannot see. This module reimplements the same architectures with no external
dependency and with an explicit *edge* view, so the existing function-space /
branch-aware machinery applies unchanged.

Edge convention
---------------
Every compressible layer exposes an edge matrix

    edge_matrix() -> [E, C]     with C = degree + 2

where each row is one Kolmogorov-Arnold edge function

    phi_e(x) = sum_{d=0..degree} w[e, d] * P_d(tanh(x))  +  w[e, -1] * silu(x)

For a conv layer an edge is one (out_channel, in_channel, kernel_row,
kernel_col) tuple, so E = out * (in/groups) * kh * kw. Slot ``[-1]`` is the
base branch, matching the ``[..., :-1] = basis, [..., -1] = base`` convention in
``funcodekan.models.variants``. Branch-aware clustering therefore splits the
same way it does for MLP-KANs.

Storage note: parameters are held in *conv-ready* layout (poly_weight,
base_weight) so the dense forward pass costs no reshape. The [E, C] edge view is
materialised only for clustering, and the codebook providers emit conv-ready
tensors directly.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "gram_poly",
    "DenseEdgeWeights",
    "SharedCodebookEdgeWeights",
    "BranchCodebookEdgeWeights",
    "KAGNConv2d",
    "KAGNLinear",
    "SimpleConvKAGN",
    "EightSimpleConvKAGN",
    "compressible_layers",
    "build_conv_kagn",
    "CONV_KAGN_PRESETS",
]


# --------------------------------------------------------------------------
# Gram / Legendre-style polynomial basis
# --------------------------------------------------------------------------

def gram_poly(x: torch.Tensor, degree: int, dim: int) -> torch.Tensor:
    """Stack P_0..P_degree of the Gram (shifted Legendre) recurrence.

    ``x`` is expected to already lie in [-1, 1] (we tanh upstream). Returns a
    tensor with a new axis of size ``degree + 1`` inserted at ``dim``.
    """
    p0 = torch.ones_like(x)
    if degree == 0:
        return p0.unsqueeze(dim)

    p1 = x
    terms = [p0, p1]
    for i in range(2, degree + 1):
        p2 = (x * p1 * (2.0 * i - 1.0) - p0 * (i - 1.0)) / i
        terms.append(p2)
        p0, p1 = p1, p2
    return torch.stack(terms, dim=dim)


# --------------------------------------------------------------------------
# Weight providers: dense, shared codebook, branch-aware codebook
# --------------------------------------------------------------------------

class _EdgeWeightProvider(nn.Module):
    """Supplies (poly_weight, base_weight) in conv/linear-ready layout.

    Subclasses differ only in where the numbers come from. ``kind`` is used by
    the storage accountant.
    """

    kind = "dense"

    def __init__(self, out_features: int, in_features: int, spatial: Tuple[int, ...], degree: int):
        super().__init__()
        self.out_features = int(out_features)
        self.in_features = int(in_features)      # already divided by groups
        self.spatial = tuple(int(s) for s in spatial)
        self.degree = int(degree)
        self.coeff_dim = int(degree) + 2
        self.n_edges = int(out_features * in_features * math.prod(self.spatial)) if self.spatial else int(out_features * in_features)

    # -- shapes ----------------------------------------------------------
    @property
    def _poly_shape(self) -> Tuple[int, ...]:
        # conv:   [out, in*(D+1), kh, kw]     linear: [out, in*(D+1)]
        return (self.out_features, self.in_features * (self.degree + 1)) + self.spatial

    @property
    def _base_shape(self) -> Tuple[int, ...]:
        return (self.out_features, self.in_features) + self.spatial

    def _edges_to_conv(self, vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[E, C] edge rows -> (poly_weight, base_weight) in conv-ready layout."""
        d1 = self.degree + 1
        edge_shape = (self.out_features, self.in_features) + self.spatial
        poly = vals[:, :d1].view(*edge_shape, d1)
        # move the degree axis next to the input-channel axis: (o, i, D+1, *spatial)
        perm = (0, 1, len(edge_shape)) + tuple(range(2, 2 + len(self.spatial)))
        poly = poly.permute(*perm).reshape(*self._poly_shape)
        base = vals[:, -1].view(*self._base_shape)
        return poly, base

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def edge_matrix(self) -> torch.Tensor:
        """[E, C] view of the *current* weights, edge-ordered (o, i, *spatial)."""
        poly, base = self()
        d1 = self.degree + 1
        p = poly.view(self.out_features, self.in_features, d1, *self.spatial)
        perm = (0, 1) + tuple(range(3, 3 + len(self.spatial))) + (2,)
        p = p.permute(*perm).reshape(-1, d1)
        b = base.reshape(-1, 1)
        return torch.cat([p, b], dim=1)


class DenseEdgeWeights(_EdgeWeightProvider):
    kind = "dense"

    def __init__(self, out_features, in_features, spatial, degree):
        super().__init__(out_features, in_features, spatial, degree)
        fan_in = self.in_features * (math.prod(self.spatial) if self.spatial else 1)

        poly = torch.empty(*self._poly_shape)
        nn.init.normal_(poly, mean=0.0, std=1.0 / math.sqrt(max(fan_in * (degree + 1), 1)))
        self.poly_weight = nn.Parameter(poly)

        base = torch.empty(*self._base_shape)
        nn.init.kaiming_uniform_(base.view(self.out_features, -1), a=math.sqrt(5))
        self.base_weight = nn.Parameter(base)

    def forward(self):
        return self.poly_weight, self.base_weight


class SharedCodebookEdgeWeights(_EdgeWeightProvider):
    """One codebook for the whole edge vector (FuncCode single-codebook arm)."""

    kind = "shared"

    def __init__(self, template: _EdgeWeightProvider, codebook: torch.Tensor,
                 ids: torch.Tensor, train_codebook: bool = True):
        super().__init__(template.out_features, template.in_features, template.spatial, template.degree)
        assert codebook.shape[1] == self.coeff_dim, (codebook.shape, self.coeff_dim)
        self.codebook = nn.Parameter(codebook.clone().float(), requires_grad=train_codebook)
        self.register_buffer("ids", ids.clone().to(torch.int32))
        self.num_codes = int(codebook.shape[0])

    def forward(self):
        vals = self.codebook[self.ids.long()]
        return self._edges_to_conv(vals)


class BranchCodebookEdgeWeights(_EdgeWeightProvider):
    """Separate codebooks for the polynomial branch and the base branch."""

    kind = "branch"

    def __init__(self, template: _EdgeWeightProvider,
                 poly_codebook: torch.Tensor, poly_ids: torch.Tensor,
                 base_codebook: torch.Tensor, base_ids: torch.Tensor,
                 train_codebook: bool = True):
        super().__init__(template.out_features, template.in_features, template.spatial, template.degree)
        if base_codebook.dim() == 1:
            base_codebook = base_codebook[:, None]
        self.poly_codebook = nn.Parameter(poly_codebook.clone().float(), requires_grad=train_codebook)
        self.base_codebook = nn.Parameter(base_codebook.clone().float(), requires_grad=train_codebook)
        self.register_buffer("poly_ids", poly_ids.clone().to(torch.int32))
        self.register_buffer("base_ids", base_ids.clone().to(torch.int32))
        self.num_poly_codes = int(poly_codebook.shape[0])
        self.num_base_codes = int(base_codebook.shape[0])

    def forward(self):
        p = self.poly_codebook[self.poly_ids.long()]          # [E, D+1]
        b = self.base_codebook[self.base_ids.long()]          # [E, 1]
        return self._edges_to_conv(torch.cat([p, b], dim=1))


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

class KAGNConv2d(nn.Module):
    """Gram-polynomial KAN convolution.

        y = act( norm( conv(P(tanh x), W_poly) + conv(silu x, W_base) ) )

    Functionally equivalent to ``kans.kagn_kagn_conv.KAGNConv2DLayer`` but with
    the FuncCode edge view exposed and no external dependency.
    """

    is_compressible = True

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 degree: int = 3, groups: int = 1, padding: int = 1, stride: int = 1,
                 dilation: int = 1, dropout: float = 0.0, affine: bool = True,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in/out channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.degree = degree
        self.stride, self.padding, self.dilation = stride, padding, dilation
        k = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.kernel_size = tuple(k)

        self.wprov = DenseEdgeWeights(out_channels, in_channels // groups, self.kernel_size, degree)

        self.norm = norm_layer(out_channels, affine=affine) if norm_layer is not None else nn.Identity()
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else None

    def forward(self, x):
        if self.drop is not None:
            x = self.drop(x)
        poly_w, base_w = self.wprov()
        poly_w = poly_w.to(x.dtype)
        base_w = base_w.to(x.dtype)

        xn = torch.tanh(x)
        basis = gram_poly(xn, self.degree, dim=2)                       # [B, C, D+1, H, W]
        basis = basis.flatten(1, 2)                                     # [B, C*(D+1), H, W]

        y = F.conv2d(basis, poly_w, None, self.stride, self.padding, self.dilation, self.groups)
        y = y + F.conv2d(self.act(x), base_w, None, self.stride, self.padding, self.dilation, self.groups)
        return self.act(self.norm(y))


class KAGNLinear(nn.Module):
    """Gram-polynomial KAN linear layer (used for the classifier head)."""

    is_compressible = True

    def __init__(self, in_features: int, out_features: int, degree: int = 3,
                 dropout: float = 0.0, use_norm: bool = True, final: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.final = final

        self.wprov = DenseEdgeWeights(out_features, in_features, (), degree)
        self.norm = nn.LayerNorm(in_features) if use_norm else nn.Identity()
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.norm(x)
        if self.drop is not None:
            x = self.drop(x)
        poly_w, base_w = self.wprov()
        poly_w = poly_w.to(x.dtype)
        base_w = base_w.to(x.dtype)

        xn = torch.tanh(x)
        basis = gram_poly(xn, self.degree, dim=2).flatten(1, 2)         # [B, in*(D+1)]
        y = F.linear(basis, poly_w) + F.linear(self.act(x), base_w)
        return y if self.final else self.act(y)


def compressible_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Named ``KAGNConv2d`` / ``KAGNLinear`` layers, in forward order."""
    return [(n, m) for n, m in model.named_modules() if getattr(m, "is_compressible", False)]


# --------------------------------------------------------------------------
# Architectures (standard convolutional KAGN backbones)
# --------------------------------------------------------------------------

def _stage(in_c, out_c, degree, groups, stride, dropout, affine, norm_layer):
    return KAGNConv2d(in_c, out_c, kernel_size=3, degree=degree, groups=groups,
                      padding=1, stride=stride, dilation=1, dropout=dropout,
                      affine=affine, norm_layer=norm_layer)


class _ConvKAGNBase(nn.Module):
    def __init__(self, layer_sizes, strides, num_classes, input_channels, degree,
                 degree_out, groups, dropout, dropout_linear, affine, norm_layer):
        super().__init__()
        blocks = []
        c_in = input_channels
        for i, (c_out, s) in enumerate(zip(layer_sizes, strides)):
            blocks.append(_stage(c_in, c_out, degree,
                                 1 if i == 0 else groups,          # first layer never grouped
                                 s, 0.0 if i == 0 else dropout,
                                 affine, norm_layer))
            c_in = c_out
        blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers = nn.Sequential(*blocks)

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear),
                                        nn.Linear(layer_sizes[-1], num_classes))
        else:
            self.output = KAGNLinear(layer_sizes[-1], num_classes, degree=degree_out,
                                     dropout=dropout_linear, use_norm=True, final=True)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.output(x)


class SimpleConvKAGN(_ConvKAGNBase):
    """4-layer KAGN CNN. Strides (1, 2, 2, 1) -> 32x32 features at 8x8."""

    def __init__(self, layer_sizes, num_classes: int = 10, input_channels: int = 1,
                 degree: int = 3, degree_out: int = 3, groups: int = 1,
                 dropout: float = 0.0, dropout_linear: float = 0.0,
                 l1_penalty: float = 0.0, affine: bool = True,
                 norm_layer=nn.BatchNorm2d):
        assert len(layer_sizes) == 4
        super().__init__(layer_sizes, (1, 2, 2, 1), num_classes, input_channels,
                         degree, degree_out, groups, dropout, dropout_linear,
                         affine, norm_layer)
        self.l1_penalty = l1_penalty


class EightSimpleConvKAGN(_ConvKAGNBase):
    """8-layer KAGN CNN. Strides (1, 2, 2, 1, 1, 2, 1, 1)."""

    def __init__(self, layer_sizes, num_classes: int = 10, input_channels: int = 1,
                 degree: int = 3, degree_out: int = 3, groups: int = 1,
                 dropout: float = 0.0, dropout_linear: float = 0.0,
                 l1_penalty: float = 0.0, affine: bool = True,
                 norm_layer=nn.BatchNorm2d):
        assert len(layer_sizes) == 8
        super().__init__(layer_sizes, (1, 2, 2, 1, 1, 2, 1, 1), num_classes,
                         input_channels, degree, degree_out, groups, dropout,
                         dropout_linear, affine, norm_layer)
        self.l1_penalty = l1_penalty


# --------------------------------------------------------------------------
# Presets -- the reported backbones, plus iso-storage variants
# --------------------------------------------------------------------------

CONV_KAGN_PRESETS = {
    # ---- the two backbones used for the reported CIFAR results -----------
    "kagn_simple_cifar10": dict(
        cls="SimpleConvKAGN", layer_sizes=[8 * 4, 16 * 4, 32 * 4, 64 * 4],
        num_classes=10, input_channels=3, degree=3, groups=4,
        dropout=0.25, dropout_linear=0.5, l1_penalty=0.0, degree_out=3,
    ),
    "kagn_simple_cifar100_8_layer_v2": dict(
        cls="EightSimpleConvKAGN", layer_sizes=[64, 128, 256, 256, 384, 384, 512, 256],
        num_classes=100, input_channels=3, degree=3, groups=1,
        dropout=0.25, dropout_linear=0.5, l1_penalty=0.0, degree_out=3,
    ),
    # ---- cross-applied so both datasets get both depths -------------------
    "kagn_simple_cifar100": dict(
        cls="SimpleConvKAGN", layer_sizes=[8 * 4, 16 * 4, 32 * 4, 64 * 4],
        num_classes=100, input_channels=3, degree=3, groups=4,
        dropout=0.25, dropout_linear=0.5, l1_penalty=0.0, degree_out=3,
    ),
    "kagn_simple_cifar10_8_layer_v2": dict(
        cls="EightSimpleConvKAGN", layer_sizes=[64, 128, 256, 256, 384, 384, 512, 256],
        num_classes=10, input_channels=3, degree=3, groups=1,
        dropout=0.25, dropout_linear=0.5, l1_penalty=0.0, degree_out=3,
    ),
    # ---- smoke test ------------------------------------------------------
    "kagn_tiny_smoke": dict(
        cls="SimpleConvKAGN", layer_sizes=[8, 16, 16, 32],
        num_classes=10, input_channels=3, degree=3, groups=1,
        dropout=0.0, dropout_linear=0.0, l1_penalty=0.0, degree_out=3,
    ),
}

_CLASSES = {"SimpleConvKAGN": SimpleConvKAGN, "EightSimpleConvKAGN": EightSimpleConvKAGN}


def build_conv_kagn(preset: str, num_classes: Optional[int] = None,
                    width_scale: float = 1.0, **overrides) -> nn.Module:
    if preset not in CONV_KAGN_PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Available: {sorted(CONV_KAGN_PRESETS)}")
    cfg = dict(CONV_KAGN_PRESETS[preset])
    cls = _CLASSES[cfg.pop("cls")]
    if num_classes is not None:
        cfg["num_classes"] = num_classes
    cfg.update(overrides)
    layer_sizes = cfg.pop("layer_sizes")

    if width_scale != 1.0:
        g = int(cfg.get("groups", 1))
        # keep every width divisible by groups so the grouped convs stay legal
        layer_sizes = [max(g, int(round(c * width_scale / g)) * g) for c in layer_sizes]
    return cls(layer_sizes, **cfg)


def model_edge_bits(model: nn.Module, bits_per_coeff: int = 32) -> int:
    """FP32-equivalent storage of all edge coefficients."""
    return int(sum(l.wprov.n_edges * l.wprov.coeff_dim * bits_per_coeff
                   for _, l in compressible_layers(model)))


def solve_iso_width(preset: str, target_bits: int, num_classes: Optional[int] = None,
                    lo: float = 0.02, hi: float = 1.0, tol: float = 0.01) -> float:
    """Width multiplier whose dense FP32 model matches ``target_bits`` of storage.

    Used to build the "same storage, fewer channels" baseline, which answers
    the natural control question: if the budget is N bits, is it better spent
    on a codebook or on a smaller dense network?
    """
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        bits = model_edge_bits(build_conv_kagn(preset, num_classes=num_classes, width_scale=mid))
        if abs(bits - target_bits) / target_bits < tol:
            return mid
        if bits > target_bits:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
