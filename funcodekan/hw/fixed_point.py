"""Bit-exact integer emulator of the HLS datapath (D2/D3) + FP32 path (D1).

This module IS the specification of the hardware arithmetic: the HLS C++ in
hw/hls/ must produce identical integers (L2 of the verification ladder), and
the QAT wrappers already share its LUT rounding (hw_spec.lut_quantize).

Datapath per layer (frozen; see docs/HW_DESIGN_CONTRACT.md):
  u        = q + 8                                   (offset code, 0..15)
  acc_s[o] = sum_{i,k} spline_q[o,i,k] * LUT_B[u_i][k]   int32
  acc_b[o] = sum_i     base_q[o,i]     * LUT_S[u_i]      int32
  t        = acc_s*M_s + acc_b*M_b                       int64
  hidden:  q' = clamp((t + 2^(shift-1)) >> shift, -8, 7)
  output:  logit_int32 = (t + 2^(shift-1)) >> shift      (scale 2^-8)

D3 differs ONLY in where spline_q/base_q come from: codebook[index] lookups.
All emulator arrays are int64 for headroom, with hard assertions that every
intermediate fits the declared HW width.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .hw_spec import (BASIS_FRAC_BITS, BASIS_SCALE, LOGIT_FRAC_BITS,
                      LUT_INT_MAX, LUT_INT_MIN, QN, QP, SILU_FRAC_BITS,
                      SILU_SCALE, requant_multiplier)

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1


def _round_half_up_np(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


def build_layer_luts(layer, act_step: float) -> tuple[np.ndarray, np.ndarray]:
    """16-entry basis and SiLU LUTs (int Q4.12) for one layer.

    Built with the layer's OWN torch float32 b_splines/silu on the 16
    dequantized values q*act_step — the same float32 values the QAT forward
    saw — then rounded with the HW rule. This function is the single source
    of LUT truth for QAT (via lut_quantize), the emulator and the export.
    """
    q = torch.arange(QN, QP + 1, dtype=torch.float32)
    step = torch.tensor(act_step, dtype=torch.float32)
    x = q * step                                     # [16] float32
    xx = x.unsqueeze(1).expand(len(q), layer.in_features).contiguous()
    basis = layer.b_splines(xx)[:, 0, :]             # [16, 8] (grid rows equal)
    silu = F.silu(x)                                 # [16]
    lut_b = _round_half_up_np(basis.numpy().astype(np.float64) * BASIS_SCALE)
    lut_s = _round_half_up_np(silu.numpy().astype(np.float64) * SILU_SCALE)
    for name, lut in (("basis Q2.14", lut_b), ("silu Q5.11", lut_s)):
        if lut.min() < LUT_INT_MIN or lut.max() > LUT_INT_MAX:
            raise OverflowError(
                f"{name} LUT exceeds int16 (act_step={act_step})")
    return lut_b.astype(np.int64), lut_s.astype(np.int64)


@dataclass
class QuantLayer:
    """One integer layer. For D3, spline_q/base_q are materialized from the
    codebooks through the frozen indices — integer-identical to the HW's
    per-edge BRAM lookup (documented equivalence)."""
    lut_b: np.ndarray            # [16, 8] int
    lut_s: np.ndarray            # [16] int
    spline_q: np.ndarray         # [out, in, 8] int, -8..7
    base_q: np.ndarray           # [out, in] int, -8..7
    mult_spline: int
    mult_base: int
    shift: int
    is_output: bool
    # D3 provenance (None for D2); kept so export can emit codebooks+indices
    spline_codebook_q: np.ndarray | None = None
    base_codebook_q: np.ndarray | None = None
    spline_ids: np.ndarray | None = None
    base_ids: np.ndarray | None = None

    def forward_codes(self, q: np.ndarray) -> np.ndarray:
        assert q.min() >= QN and q.max() <= QP
        u = (q.astype(np.int64) - QN)                      # 0..15
        b = self.lut_b[u]                                  # [N, in, 8]
        s = self.lut_s[u]                                  # [N, in]
        acc_s = np.einsum("nik,oik->no", b, self.spline_q, dtype=np.int64)
        acc_b = s @ self.base_q.T.astype(np.int64)
        for acc in (acc_s, acc_b):
            if acc.min() < INT32_MIN or acc.max() > INT32_MAX:
                raise OverflowError("int32 accumulator overflow")
        t = acc_s * self.mult_spline + acc_b * self.mult_base   # int64
        r = (t + (1 << (self.shift - 1))) >> self.shift          # arithmetic
        if self.is_output:
            if r.min() < INT32_MIN or r.max() > INT32_MAX:
                raise OverflowError("int32 logit overflow")
            return r
        return np.clip(r, QN, QP)


@dataclass
class FixedPointModel:
    """Integer-only inference for D2/D3."""
    input_step: float
    layers: list[QuantLayer] = field(default_factory=list)

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        """float image -> INT4 codes. All ops in float32 so the arithmetic is
        bit-identical to the torch LSQ quantizer (x/s then floor(v+0.5))."""
        xs = x.astype(np.float32) / np.float32(self.input_step)
        q = np.floor(xs + np.float32(0.5))
        return np.clip(q, QN, QP).astype(np.int64)

    def forward_codes(self, q: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            q = layer.forward_codes(q)
        return q  # int32 logits at scale 2^-LOGIT_FRAC_BITS

    def forward_float(self, x: np.ndarray, batch: int = 1024) -> np.ndarray:
        """Batched full-set inference (int64 basis gathers are memory-heavy)."""
        outs = []
        for i in range(0, len(x), batch):
            outs.append(self.forward_codes(self.quantize_input(x[i:i + batch])))
        return np.concatenate(outs)


def _requant_for_layer(spline_step: float, base_step: float,
                       out_step: float) -> tuple[int, int, int]:
    ratio_s = spline_step * (2.0 ** -BASIS_FRAC_BITS) / out_step
    ratio_b = base_step * (2.0 ** -SILU_FRAC_BITS) / out_step
    (m_s, m_b), shift = requant_multiplier([ratio_s, ratio_b])
    return m_s, m_b, shift


def _act_out_steps(act_steps: list[float]) -> list[float]:
    """Requant target step per layer: next act step, then the logit scale."""
    return act_steps[1:] + [2.0 ** -LOGIT_FRAC_BITS]


def build_from_d2(wrapper) -> FixedPointModel:
    """QuantSplineKAN -> integer model."""
    wrapper = wrapper.cpu().eval()
    act_steps = wrapper.act_steps()
    out_steps = _act_out_steps(act_steps)
    model = FixedPointModel(input_step=act_steps[0])
    n = len(wrapper.dense.layers)
    for l in range(n):
        iw = wrapper.int_weights(l)
        lut_b, lut_s = build_layer_luts(wrapper.dense.layers[l], act_steps[l])
        m_s, m_b, shift = _requant_for_layer(iw["spline_step"],
                                             iw["base_step"], out_steps[l])
        model.layers.append(QuantLayer(
            lut_b=lut_b, lut_s=lut_s,
            spline_q=iw["spline_q"].numpy().astype(np.int64),
            base_q=iw["base_q"].numpy().astype(np.int64),
            mult_spline=m_s, mult_base=m_b, shift=shift,
            is_output=(l == n - 1)))
    return model


def build_from_d3(wrapper) -> FixedPointModel:
    """QuantBranchKAN -> integer model (codebooks + frozen indices)."""
    wrapper = wrapper.cpu().eval()
    act_steps = wrapper.act_steps()
    out_steps = _act_out_steps(act_steps)
    model = FixedPointModel(input_step=act_steps[0])
    n = len(wrapper.branch.layers)
    for l in range(n):
        cb = wrapper.int_codebooks(l)
        lut_b, lut_s = build_layer_luts(wrapper.branch.layers[l], act_steps[l])
        m_s, m_b, shift = _requant_for_layer(cb["spline_step"],
                                             cb["base_step"], out_steps[l])
        scb = cb["spline_codebook_q"].numpy().astype(np.int64)
        bcb = cb["base_codebook_q"].numpy().astype(np.int64)
        sids = cb["spline_ids"].numpy()
        bids = cb["base_ids"].numpy()
        model.layers.append(QuantLayer(
            lut_b=lut_b, lut_s=lut_s,
            spline_q=scb[sids],          # == HW per-edge codebook lookup
            base_q=bcb[bids],
            mult_spline=m_s, mult_base=m_b, shift=shift,
            is_output=(l == n - 1),
            spline_codebook_q=scb, base_codebook_q=bcb,
            spline_ids=sids, base_ids=bids))
    return model


# ---------------------------------------------------------------------------
# D1: FP32 numpy reference (mirrors SplineLayer exactly, float32 throughout)
# ---------------------------------------------------------------------------

def _b_splines_np(x: np.ndarray, grid: np.ndarray, order: int) -> np.ndarray:
    """numpy float32 replica of SplineLayer.b_splines (same eps, same
    half-open indicators)."""
    x = x[..., None].astype(np.float32)                    # [N, in, 1]
    g = grid.astype(np.float32)                            # [in, 12]
    bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).astype(np.float32)
    eps = np.float32(1e-8)
    for k in range(1, order + 1):
        delta_prev = g[:, k:-1] - g[:, :-(k + 1)]
        delta_next = g[:, k + 1:] - g[:, 1:(-k)]
        term1 = (x - g[:, :-(k + 1)]) / (delta_prev + eps) * bases[:, :, :-1]
        term2 = (g[:, k + 1:] - x) / (delta_next + eps) * bases[:, :, 1:]
        bases = term1 + term2
    return bases


def _silu_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


@dataclass
class Fp32Layer:
    weight: np.ndarray           # [out, in, 9] float32
    grid: np.ndarray             # [in, 12] float32
    order: int

    def forward(self, x: np.ndarray) -> np.ndarray:
        spline_w = self.weight[..., :-1]
        base_w = self.weight[..., -1]
        basis = _b_splines_np(x, self.grid, self.order)
        y = np.einsum("nik,oik->no", basis, spline_w, dtype=np.float32)
        return y + _silu_np(x) @ base_w.T


@dataclass
class Fp32Model:
    layers: list[Fp32Layer] = field(default_factory=list)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        for layer in self.layers:
            x = layer.forward(x)
        return x


def build_from_dense(dense) -> Fp32Model:
    dense = dense.cpu().eval()
    model = Fp32Model()
    for l, layer in enumerate(dense.layers):
        model.layers.append(Fp32Layer(
            weight=dense.weights[l].detach().numpy().astype(np.float32),
            grid=layer.grid.numpy().astype(np.float32),
            order=layer.spline_order))
    return model
