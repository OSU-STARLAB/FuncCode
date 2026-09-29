"""Bit-exact integer emulator for the GRAM designs (G2/G3) + FP32 (G1).

numpy twins of the frozen integer kernels in gram_spec.py (torch) and
hw/hls/common/int_layernorm.h (HLS). Datapath per layer:

  u        = q + 8
  acc_s[o] = sum_{i,k} basis_q[o,i,k] * LUT_B[u_i][k]      int32
  acc_b[o] = sum_i     base_q[o,i]    * LUT_S[u_i]         int32
  v[o]     = (acc_s*M_s + acc_b*M_b + 2^(s-1)) >> s        int32 Q.12
  y[o]     = int_layernorm(v; gamma_q, beta_q, E)          int Q.12
  s[o]     = silu_table_interp(y)                          int Q6.10
  hidden:  q' = clamp((s*M_a + 2^(sh-1)) >> sh, -8, 7)
  output:  logit_int32 = s << 6                            (scale 2^-16)

G3 differs ONLY in basis_q/base_q provenance: codebook[index] lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .hw_spec import (BASIS_SCALE, LUT_INT_MAX, LUT_INT_MIN, QN, QP,
                      SILU_SCALE, requant_multiplier)
from .gram_spec import (GAMMA_FRAC, LN_EPS, LOGIT_SHIFT, NORM_FRAC,
                        PRE_LN_FRAC, SILU_TAB_FRAC, SILU_TAB_LO, SILU_TABLE,
                        ln_eps_int)

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1


def _rhu(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


# ---------------------------------------------------------------------------
# numpy twins of the frozen integer kernels
# ---------------------------------------------------------------------------

def isqrt64_np(x: np.ndarray) -> np.ndarray:
    num = x.astype(np.int64).copy()
    res = np.zeros_like(num)
    bit = np.int64(1) << np.int64(62)
    for _ in range(32):
        t = res + bit
        ge = num >= t
        num = np.where(ge, num - t, num)
        res = np.where(ge, (res >> 1) + bit, res >> 1)
        bit >>= np.int64(2)
    return res


def int_layernorm_np(v: np.ndarray, gamma_q: np.ndarray,
                     beta_q: np.ndarray) -> np.ndarray:
    n = v.shape[-1]
    s = v.sum(axis=-1, keepdims=True)
    c = n * v - s
    q = (c * c).sum(axis=-1, keepdims=True)
    if (q >= (1 << 62)).any():
        raise OverflowError("int LayerNorm variance exceeds 2^62")
    r = isqrt64_np(q // n + ln_eps_int(n))
    n_fx = np.floor_divide(2 * c * (1 << NORM_FRAC) + r, 2 * r)
    t = n_fx * gamma_q + beta_q
    return np.floor_divide(t + (1 << (GAMMA_FRAC - 1)), 1 << GAMMA_FRAC)


def silu_interp_int_np(y: np.ndarray) -> np.ndarray:
    lo = int(SILU_TAB_LO * (1 << NORM_FRAC))
    yc = np.clip(y, lo, -lo - 1)
    u = yc - lo
    i = u >> 8
    frac = u & 255
    t0 = SILU_TABLE[i]
    t1 = SILU_TABLE[i + 1]
    return t0 + np.floor_divide((t1 - t0) * frac + 128, 256)


def build_gram_luts(layer, act_step: float):
    """16-entry basis (Q2.14) and input-SiLU (Q6.10) LUTs, built with the
    verified layer's own torch float32 basis()."""
    q = torch.arange(QN, QP + 1, dtype=torch.float32)
    x = q * torch.tensor(act_step, dtype=torch.float32)
    basis = layer.basis(x)                       # [16, basis_dim]
    silu = F.silu(x)
    lut_b = _rhu(basis.numpy().astype(np.float64) * BASIS_SCALE)
    lut_s = _rhu(silu.numpy().astype(np.float64) * SILU_SCALE)
    for name, lut in (("gram basis Q2.14", lut_b), ("silu Q6.10", lut_s)):
        if lut.min() < LUT_INT_MIN or lut.max() > LUT_INT_MAX:
            raise OverflowError(f"{name} LUT exceeds int16 "
                                f"(act_step={act_step})")
    return lut_b.astype(np.int64), lut_s.astype(np.int64)


# ---------------------------------------------------------------------------
# layers / model
# ---------------------------------------------------------------------------

@dataclass
class GramQuantLayer:
    lut_b: np.ndarray            # [16, basis_dim] int
    lut_s: np.ndarray            # [16] int
    basis_q: np.ndarray          # [out, in, basis_dim] int -8..7
    base_q: np.ndarray           # [out, in] int
    mult_basis: int
    mult_base: int
    shift: int
    gamma_q: np.ndarray          # [out] int64 Q2.14
    beta_q: np.ndarray           # [out] int64 Q.26
    act_mult: int                # hidden requant (0 for output layer)
    act_shift: int
    is_output: bool
    basis_codebook_q: np.ndarray | None = None
    base_codebook_q: np.ndarray | None = None
    basis_ids: np.ndarray | None = None
    base_ids: np.ndarray | None = None

    def forward_codes(self, q: np.ndarray) -> np.ndarray:
        assert q.min() >= QN and q.max() <= QP
        u = q.astype(np.int64) - QN
        b = self.lut_b[u]                                    # [N, in, K]
        s = self.lut_s[u]                                    # [N, in]
        acc_s = np.einsum("nik,oik->no", b, self.basis_q, dtype=np.int64)
        acc_b = s @ self.base_q.T.astype(np.int64)
        for acc in (acc_s, acc_b):
            if acc.min() < INT32_MIN or acc.max() > INT32_MAX:
                raise OverflowError("int32 accumulator overflow")
        t = acc_s * self.mult_basis + acc_b * self.mult_base
        v = (t + (1 << (self.shift - 1))) >> self.shift       # Q.12
        if v.min() < INT32_MIN or v.max() > INT32_MAX:
            raise OverflowError("int32 pre-LN overflow")
        y = int_layernorm_np(v, self.gamma_q, self.beta_q)
        so = silu_interp_int_np(y)                            # Q6.10
        if self.is_output:
            return so << LOGIT_SHIFT                          # Q.16 logits
        ta = so * self.act_mult
        qn = (ta + (1 << (self.act_shift - 1))) >> self.act_shift
        return np.clip(qn, QN, QP)


@dataclass
class GramFixedPointModel:
    input_step: float
    layers: list[GramQuantLayer] = field(default_factory=list)

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        xs = x.astype(np.float32) / np.float32(self.input_step)
        return np.clip(np.floor(xs + np.float32(0.5)), QN, QP
                       ).astype(np.int64)

    def forward_codes(self, q: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            q = layer.forward_codes(q)
        return q

    def forward_float(self, x: np.ndarray, batch: int = 1024) -> np.ndarray:
        outs = []
        for i in range(0, len(x), batch):
            outs.append(self.forward_codes(
                self.quantize_input(x[i:i + batch])))
        return np.concatenate(outs)


def _layer_consts(wrapper, l: int, basis_step: float, base_step: float,
                  n_layers: int):
    """Requant + LN constants for one layer.

    pre-LN ratios: acc_s scale = basis_step*2^-14 (Q2.14 LUT), acc_b scale
    = base_step*2^-10 (Q6.10 LUT); target grid 2^-PRE_LN_FRAC."""
    r_s = basis_step * (2.0 ** (PRE_LN_FRAC - 14))
    r_b = base_step * (2.0 ** (PRE_LN_FRAC - 10))
    (m_s, m_b), shift = requant_multiplier([r_s, r_b])
    ln = wrapper.ln_consts(l)
    if l < n_layers - 1:
        s_a_next = wrapper.act_quants[l + 1].step_size()
        (m_a,), a_shift = requant_multiplier(
            [(2.0 ** -SILU_TAB_FRAC) / s_a_next])
    else:
        m_a, a_shift = 0, 1
    return (m_s, m_b, shift,
            ln["gamma_q"].numpy().astype(np.int64),
            ln["beta_q"].numpy().astype(np.int64),
            m_a, a_shift)


def build_from_g2(wrapper) -> GramFixedPointModel:
    wrapper = wrapper.cpu().eval()
    steps = wrapper.act_steps()
    model = GramFixedPointModel(input_step=steps[0])
    layers = wrapper.gram_layers()
    n = len(layers)
    for l in range(n):
        iw = wrapper.int_weights(l)
        lut_b, lut_s = build_gram_luts(layers[l], steps[l])
        m_s, m_b, sh, g_q, b_q, m_a, a_sh = _layer_consts(
            wrapper, l, iw["basis_step"], iw["base_step"], n)
        model.layers.append(GramQuantLayer(
            lut_b=lut_b, lut_s=lut_s,
            basis_q=iw["basis_q"].numpy().astype(np.int64),
            base_q=iw["base_q"].numpy().astype(np.int64),
            mult_basis=m_s, mult_base=m_b, shift=sh,
            gamma_q=g_q, beta_q=b_q, act_mult=m_a, act_shift=a_sh,
            is_output=(l == n - 1)))
    return model


def build_from_g3(wrapper) -> GramFixedPointModel:
    wrapper = wrapper.cpu().eval()
    steps = wrapper.act_steps()
    model = GramFixedPointModel(input_step=steps[0])
    layers = wrapper.gram_layers()
    n = len(layers)
    for l in range(n):
        cb = wrapper.int_codebooks(l)
        lut_b, lut_s = build_gram_luts(layers[l], steps[l])
        m_s, m_b, sh, g_q, b_q, m_a, a_sh = _layer_consts(
            wrapper, l, cb["basis_step"], cb["base_step"], n)
        scb = cb["basis_codebook_q"].numpy().astype(np.int64)
        bcb = cb["base_codebook_q"].numpy().astype(np.int64)
        sids = cb["basis_ids"].numpy()
        bids = cb["base_ids"].numpy()
        model.layers.append(GramQuantLayer(
            lut_b=lut_b, lut_s=lut_s,
            basis_q=scb[sids], base_q=bcb[bids],
            mult_basis=m_s, mult_base=m_b, shift=sh,
            gamma_q=g_q, beta_q=b_q, act_mult=m_a, act_shift=a_sh,
            is_output=(l == n - 1),
            basis_codebook_q=scb, base_codebook_q=bcb,
            basis_ids=sids, base_ids=bids))
    return model


# ---------------------------------------------------------------------------
# G1: FP32 numpy reference (mirrors GramPolynomialLayer, float32)
# ---------------------------------------------------------------------------

def _silu_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


def _gram_basis_np(x: np.ndarray, degree: int) -> np.ndarray:
    z = np.tanh(x.astype(np.float32))
    terms = [np.ones_like(z)]
    if degree >= 1:
        terms.append(z)
    for _ in range(2, degree + 1):
        terms.append(z * terms[-1] - np.float32(0.1) * terms[-2])
    return np.stack(terms, axis=-1)


@dataclass
class GramFp32Layer:
    weight: np.ndarray           # [out, in, coeff_dim] float32
    gamma: np.ndarray            # [out] float32
    beta: np.ndarray             # [out] float32
    degree: int

    def forward(self, x: np.ndarray) -> np.ndarray:
        basis_w = self.weight[..., :-1]
        base_w = self.weight[..., -1]
        base = _silu_np(x) @ base_w.T
        basis = _gram_basis_np(x, self.degree)
        pre = base + np.einsum("nik,oik->no", basis, basis_w,
                               dtype=np.float32)
        mu = pre.mean(axis=-1, keepdims=True, dtype=np.float32)
        var = np.mean((pre - mu) ** 2, axis=-1, keepdims=True,
                      dtype=np.float32)
        y = (pre - mu) / np.sqrt(var + np.float32(LN_EPS))
        y = y * self.gamma + self.beta
        return _silu_np(y)


@dataclass
class GramFp32Model:
    layers: list[GramFp32Layer] = field(default_factory=list)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        for layer in self.layers:
            x = layer.forward(x)
        return x


def build_from_gram_dense(dense) -> GramFp32Model:
    dense = dense.cpu().eval()
    model = GramFp32Model()
    for l, layer in enumerate(dense.layers):
        model.layers.append(GramFp32Layer(
            weight=dense.weights[l].detach().numpy().astype(np.float32),
            gamma=layer.norm.weight.detach().numpy().astype(np.float32),
            beta=layer.norm.bias.detach().numpy().astype(np.float32),
            degree=layer.degree))
    return model
