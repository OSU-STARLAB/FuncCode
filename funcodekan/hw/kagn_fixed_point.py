"""Bit-exact integer emulator for the KAGN-Conv designs (K2/K3) + FP32 (K1).

Conv layer integer datapath (frozen; docs/HW_DESIGN_CONTRACT.md K-series):
  u maps    = q + 8 ; B = LUT_B[u] (Q2.14), S = LUT_S[u] (Q6.10);
              spatial ZERO padding pads the B/S MAPS (matching the float
              model's conv2d zero padding — padded taps contribute nothing)
  acc_s[o]  = sum_{i,ky,kx,k} poly_q * B ; acc_b likewise with base_q*S
  v         = (acc_s*M_s[ch] + acc_b*M_b[ch] + B_fix[ch] + 2^(s-1)) >> s
              (BatchNorm running stats FOLDED into per-channel multipliers
              and offset; v on the Q4.12 grid)
  so        = shared interpolated integer SiLU (Q6.10)
  codes     = clamp((so*M_a + 2^(sh-1)) >> sh, -8, 7)
Pool: integer SUM of the 7x7 codes; the 1/49 folds into the head-input
requant. Head: the gram FC integer layer (int LayerNorm etc.) reused from
gram_fixed_point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .hw_spec import MULT_MAX, QN, QP, requant_multiplier
from .gram_spec import PRE_LN_FRAC, SILU_TAB_FRAC
from .gram_fixed_point import (GramQuantLayer, build_gram_luts,
                               int_layernorm_np, silu_interp_int_np)

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1


def signed_requant_multipliers(ratios: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-channel signed multipliers: largest common shift with
    max |round(ratio*2^shift)| <= 2^31-1."""
    max_ratio = float(np.abs(ratios).max())
    if max_ratio <= 0:
        raise ValueError("all ratios zero")
    shift = int(math.floor(math.log2(MULT_MAX / max_ratio)))
    while (np.abs(np.floor(ratios * (1 << shift) + 0.5)) > MULT_MAX).any():
        shift -= 1
    mults = np.floor(ratios * (1 << shift) + 0.5).astype(np.int64)
    return mults, shift


@dataclass
class KagnConvQuantLayer:
    lut_b: np.ndarray            # [16, 4]
    lut_s: np.ndarray            # [16]
    poly_q: np.ndarray           # [O, I, k, k, 4] int
    base_q: np.ndarray           # [O, I, k, k] int
    stride: int
    padding: int
    m_s: int                     # per-tensor requant to the Q.12 grid
    m_b: int
    shift: int
    gamma_q: np.ndarray          # [O] int64 (instance-norm affine, Q4.12)
    beta_q: np.ndarray           # [O] int64 (Q.24)
    act_mult: int
    act_shift: int
    poly_codebook_q: np.ndarray | None = None
    base_codebook_q: np.ndarray | None = None
    poly_ids: np.ndarray | None = None
    base_ids: np.ndarray | None = None

    def forward_codes(self, q: np.ndarray) -> np.ndarray:
        n, c, h, w = q.shape
        k = self.poly_q.shape[2]
        u = q.astype(np.int64) - QN
        b = self.lut_b[u]                            # [N,C,H,W,4]
        s = self.lut_s[u]                            # [N,C,H,W]
        p = self.padding
        bp = np.zeros((n, c, h + 2 * p, w + 2 * p, b.shape[-1]),
                      dtype=np.int64)
        sp = np.zeros((n, c, h + 2 * p, w + 2 * p), dtype=np.int64)
        bp[:, :, p:p + h, p:p + w] = b
        sp[:, :, p:p + h, p:p + w] = s
        oh = (h + 2 * p - k) // self.stride + 1
        ow = (w + 2 * p - k) // self.stride + 1
        o_ch = self.poly_q.shape[0]
        acc_s = np.zeros((n, o_ch, oh, ow), dtype=np.int64)
        acc_b = np.zeros((n, o_ch, oh, ow), dtype=np.int64)
        st = self.stride
        for ky in range(k):
            for kx in range(k):
                bs = bp[:, :, ky:ky + st * oh:st, kx:kx + st * ow:st, :]
                ss = sp[:, :, ky:ky + st * oh:st, kx:kx + st * ow:st]
                acc_s += np.einsum("ncyxk,ock->noyx", bs,
                                   self.poly_q[:, :, ky, kx, :],
                                   dtype=np.int64)
                acc_b += np.einsum("ncyx,oc->noyx", ss,
                                   self.base_q[:, :, ky, kx],
                                   dtype=np.int64)
        for acc in (acc_s, acc_b):
            if acc.min() < INT32_MIN or acc.max() > INT32_MAX:
                raise OverflowError("int32 conv accumulator overflow")
        t = acc_s * self.m_s + acc_b * self.m_b
        v = (t + (1 << (self.shift - 1))) >> self.shift       # Q.12
        if v.min() < INT32_MIN or v.max() > INT32_MAX:
            raise OverflowError("int32 pre-norm overflow")
        # integer INSTANCE norm: per-channel over spatial positions,
        # reusing the frozen integer LayerNorm kernel
        nb = v.shape[0]
        y = int_layernorm_np(v.reshape(nb, o_ch, oh * ow),
                             self.gamma_q[:, None], self.beta_q[:, None])
        so = silu_interp_int_np(y.reshape(nb, o_ch, oh, ow))
        ta = so * self.act_mult
        qn = (ta + (1 << (self.act_shift - 1))) >> self.act_shift
        return np.clip(qn, QN, QP)


@dataclass
class KagnFixedPointModel:
    input_step: float
    conv_layers: list[KagnConvQuantLayer] = field(default_factory=list)
    pool_mult: int = 0
    pool_shift: int = 1
    head: GramQuantLayer | None = None

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        xs = x.astype(np.float32) / np.float32(self.input_step)
        return np.clip(np.floor(xs + np.float32(0.5)), QN, QP
                       ).astype(np.int64)

    def forward_codes(self, q: np.ndarray) -> np.ndarray:
        for layer in self.conv_layers:
            q = layer.forward_codes(q)
        psum = q.sum(axis=(2, 3))                    # [N, C] int
        t = psum * self.pool_mult
        qh = (t + (1 << (self.pool_shift - 1))) >> self.pool_shift
        qh = np.clip(qh, QN, QP)
        return self.head.forward_codes(qh)           # int32 logits Q.16

    def forward_float(self, x: np.ndarray, batch: int = 256) -> np.ndarray:
        outs = []
        for i in range(0, len(x), batch):
            outs.append(self.forward_codes(
                self.quantize_input(x[i:i + batch])))
        return np.concatenate(outs)


def _conv_consts(conv, poly_step, base_step, act_step_next):
    from .gram_spec import quantize_ln_affine
    r_s = poly_step * (2.0 ** (PRE_LN_FRAC - 14))
    r_b = base_step * (2.0 ** (PRE_LN_FRAC - 10))
    (m_s, m_b), shift = requant_multiplier([r_s, r_b])
    g_q, b_q = quantize_ln_affine(conv.norm.weight, conv.norm.bias)
    (m_a,), a_shift = requant_multiplier(
        [(2.0 ** -SILU_TAB_FRAC) / act_step_next])
    return (m_s, m_b, shift, g_q.numpy().astype(np.int64),
            b_q.numpy().astype(np.int64), m_a, a_shift)


def build_from_kagn(wrapper, is_k3: bool) -> KagnFixedPointModel:
    wrapper = wrapper.cpu().eval()
    steps = wrapper.act_steps()                      # [in, a1, a2, head]
    model = KagnFixedPointModel(input_step=steps[0])
    convs = wrapper.convs()
    for i, conv in enumerate(convs):
        if is_k3:
            cb = wrapper.int_codebooks(i)
            poly_q = cb["poly_codebook_q"].numpy().astype(np.int64)[
                cb["poly_ids"].numpy()]
            base_q = cb["base_codebook_q"].numpy().astype(np.int64)[
                cb["base_ids"].numpy()]
            p_step, b_step = cb["poly_step"], cb["base_step"]
            prov = (cb["poly_codebook_q"].numpy().astype(np.int64),
                    cb["base_codebook_q"].numpy().astype(np.int64),
                    cb["poly_ids"].numpy(), cb["base_ids"].numpy())
        else:
            iw = wrapper.int_weights(i)
            poly_q = iw["poly_q"].numpy().astype(np.int64)
            base_q = iw["base_q"].numpy().astype(np.int64)
            p_step, b_step = iw["poly_step"], iw["base_step"]
            prov = (None, None, None, None)
        lut_b, lut_s = build_gram_luts(wrapper.head(), steps[i])
        m_s, m_b, shift, g_q, b_q, m_a, a_sh = _conv_consts(
            conv, p_step, b_step, steps[i + 1])
        model.conv_layers.append(KagnConvQuantLayer(
            lut_b=lut_b, lut_s=lut_s, poly_q=poly_q, base_q=base_q,
            stride=conv.stride, padding=conv.padding,
            m_s=m_s, m_b=m_b, shift=shift, gamma_q=g_q, beta_q=b_q,
            act_mult=m_a, act_shift=a_sh,
            poly_codebook_q=prov[0], base_codebook_q=prov[1],
            poly_ids=prov[2], base_ids=prov[3]))
    # pool: ratio = s_a2 / (49 * s_head)
    hw_positions = 49
    (m_p,), p_shift = requant_multiplier(
        [steps[2] / (hw_positions * steps[3])])
    model.pool_mult, model.pool_shift = m_p, p_shift
    # head (gram FC integer layer, output layer)
    if is_k3:
        cb = wrapper.int_codebooks(2)
        basis_q = cb["poly_codebook_q"].numpy().astype(np.int64)[
            cb["poly_ids"].numpy()]
        base_q = cb["base_codebook_q"].numpy().astype(np.int64)[
            cb["base_ids"].numpy()]
        p_step, b_step = cb["poly_step"], cb["base_step"]
        prov = (cb["poly_codebook_q"].numpy().astype(np.int64),
                cb["base_codebook_q"].numpy().astype(np.int64),
                cb["poly_ids"].numpy(), cb["base_ids"].numpy())
    else:
        iw = wrapper.int_weights(2)
        basis_q = iw["poly_q"].numpy().astype(np.int64)
        base_q = iw["base_q"].numpy().astype(np.int64)
        p_step, b_step = iw["poly_step"], iw["base_step"]
        prov = (None, None, None, None)
    lut_b, lut_s = build_gram_luts(wrapper.head(), steps[3])
    from .gram_spec import quantize_ln_affine
    g_q, b_q = quantize_ln_affine(wrapper.head().norm.weight,
                                  wrapper.head().norm.bias)
    from .hw_spec import requant_multiplier as _rm
    r_s = p_step * (2.0 ** (PRE_LN_FRAC - 14))
    r_b = b_step * (2.0 ** (PRE_LN_FRAC - 10))
    (m_s, m_b), shift = _rm([r_s, r_b])
    model.head = GramQuantLayer(
        lut_b=lut_b, lut_s=lut_s, basis_q=basis_q, base_q=base_q,
        mult_basis=m_s, mult_base=m_b, shift=shift,
        gamma_q=g_q.numpy().astype(np.int64),
        beta_q=b_q.numpy().astype(np.int64),
        act_mult=0, act_shift=1, is_output=True,
        basis_codebook_q=prov[0], base_codebook_q=prov[1],
        basis_ids=prov[2], base_ids=prov[3])
    return model


# ---------------------------------------------------------------------------
# K1: FP32 numpy reference
# ---------------------------------------------------------------------------

from .gram_fixed_point import (GramFp32Layer, _gram_basis_np,  # noqa: E402
                               _silu_np)


@dataclass
class KagnFp32Model:
    convs: list = field(default_factory=list)   # (weight, gamma, beta, s, p)
    head: GramFp32Layer | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        for weight, gamma, beta, stride, pad in self.convs:
            x = self._conv(x, weight, gamma, beta, stride, pad)
        x = x.mean(axis=(2, 3), dtype=np.float32)
        return self.head.forward(x)

    @staticmethod
    def _conv(x, weight, gamma, beta, stride, pad):
        n, c, h, w = x.shape
        k = weight.shape[2]
        basis = _gram_basis_np(x, 3)                 # [N,C,H,W,4]
        silu = _silu_np(x)
        bp = np.zeros((n, c, h + 2 * pad, w + 2 * pad, 4), dtype=np.float32)
        sp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=np.float32)
        bp[:, :, pad:pad + h, pad:pad + w] = basis
        sp[:, :, pad:pad + h, pad:pad + w] = silu
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        o_ch = weight.shape[0]
        acc = np.zeros((n, o_ch, oh, ow), dtype=np.float32)
        for ky in range(k):
            for kx in range(k):
                bs = bp[:, :, ky:ky + stride * oh:stride,
                        kx:kx + stride * ow:stride, :]
                ss = sp[:, :, ky:ky + stride * oh:stride,
                        kx:kx + stride * ow:stride]
                acc += np.einsum("ncyxk,ock->noyx", bs,
                                 weight[:, :, ky, kx, :-1],
                                 dtype=np.float32)
                acc += np.einsum("ncyx,oc->noyx", ss,
                                 weight[:, :, ky, kx, -1],
                                 dtype=np.float32)
        # instance norm: per (sample, channel) over spatial positions
        mu = acc.mean(axis=(2, 3), keepdims=True, dtype=np.float32)
        var = np.mean((acc - mu) ** 2, axis=(2, 3), keepdims=True,
                      dtype=np.float32)
        y = (acc - mu) / np.sqrt(var + np.float32(1e-5))
        y = y * gamma[None, :, None, None] + beta[None, :, None, None]
        return _silu_np(y)


def build_from_kagn_dense(net) -> KagnFp32Model:
    net = net.cpu().eval()
    model = KagnFp32Model()
    for conv in (net.conv1, net.conv2):
        model.convs.append((
            conv.weight.detach().numpy().astype(np.float32),
            conv.norm.weight.detach().numpy().astype(np.float32),
            conv.norm.bias.detach().numpy().astype(np.float32),
            conv.stride, conv.padding))
    model.head = GramFp32Layer(
        weight=net.head_weight.detach().numpy().astype(np.float32),
        gamma=net.head_layer.norm.weight.detach().numpy().astype(np.float32),
        beta=net.head_layer.norm.bias.detach().numpy().astype(np.float32),
        degree=3)
    return model
