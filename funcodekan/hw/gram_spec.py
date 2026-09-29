"""Frozen numeric contract additions for the GRAM (KAGN-basis FC) designs.

Everything in hw_spec.py carries over (INT4 ranges, per-tensor LSQ,
round-half-up, requant machinery). This module adds the GRAM-specific
pieces — most importantly the DETERMINISTIC integer LayerNorm and the
interpolated integer SiLU table used after it. The same algorithms are
implemented in three places that must stay bit-identical:
  - torch (eval-reference path of the QAT wrappers, this module),
  - numpy (funcodekan/hw/gram_fixed_point.py emulator),
  - HLS C++ (hw/hls/common/int_layernorm.h).
See docs/HW_DESIGN_CONTRACT.md "Frozen design decisions — GRAM integer
datapath". Do not change one implementation without the others.
"""

from __future__ import annotations

import numpy as np
import torch

from .hw_spec import round_half_up

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
GRAM_BASIS_DIM = 4                 # degree 3 -> [1, z, z^2-0.1, z^3-0.2z]
PRE_LN_FRAC = 12                   # pre-LayerNorm values v: int32 Q.12
NORM_FRAC = 12                     # normalized values n: Q4.12
# LN gamma: int16 Q4.12 (trained gammas reach ~2.0, exceeding Q2.14's
# bound; Q4.12 covers +-8 with 2^-12 granularity — export asserts)
GAMMA_FRAC = 12
BETA_FRAC = PRE_LN_FRAC + GAMMA_FRAC   # LN beta: int32 Q.24
LN_EPS = 1e-5                      # nn.LayerNorm default, frozen
SILU_TAB_FRAC = 10                 # SiLU table entries: int16 Q6.10
SILU_TAB_LO = -8.0                 # table domain [-8, 8), step 1/16
SILU_TAB_STEP_BITS = 4             # 16 steps per unit
SILU_TAB_N = 256                   # +1 guard entry for interpolation
LOGIT_SHIFT = 6                    # Q6.10 silu -> INT32 logits at 2^-16


def ln_eps_int(n_features: int) -> int:
    """E = round(N^2 * 2^(2*PRE_LN_FRAC) * eps) — integer eps term."""
    return int(round(n_features ** 2 * (1 << (2 * PRE_LN_FRAC)) * LN_EPS))


def quantize_ln_affine(gamma: torch.Tensor, beta: torch.Tensor):
    """LN affine constants in the frozen fixed-point formats."""
    g = round_half_up(gamma.detach().float() * (1 << GAMMA_FRAC)).to(torch.int64)
    b = round_half_up(beta.detach().float() * (1 << BETA_FRAC)).to(torch.int64)
    return g, b


def build_silu_table() -> np.ndarray:
    """The 257-entry SiLU table (model-independent, generated once)."""
    k = np.arange(SILU_TAB_N + 1, dtype=np.float64)
    x = SILU_TAB_LO + k / (1 << SILU_TAB_STEP_BITS)
    silu = x / (1.0 + np.exp(-x))
    t = np.floor(silu * (1 << SILU_TAB_FRAC) + 0.5).astype(np.int64)
    assert t.min() >= -(2 ** 15) and t.max() <= 2 ** 15 - 1
    return t


SILU_TABLE = build_silu_table()    # int64 values fitting int16


# ---------------------------------------------------------------------------
# Shared integer kernels — torch flavor (eval-reference).
# The numpy twins live in gram_fixed_point.py with IDENTICAL arithmetic.
# ---------------------------------------------------------------------------

def isqrt64_torch(x: torch.Tensor) -> torch.Tensor:
    """floor(sqrt(x)) for int64 x in [0, 2^62), fixed 32 iterations
    (bit-by-bit restoring — the exact algorithm of the HLS isqrt64)."""
    num = x.clone()
    res = torch.zeros_like(x)
    bit = torch.full_like(x, 1 << 62)
    for _ in range(32):
        t = res + bit
        ge = num >= t
        num = torch.where(ge, num - t, num)
        res = torch.where(ge, (res >> 1) + bit, res >> 1)
        bit = bit >> 2
    return res


def int_layernorm_torch(v: torch.Tensor, gamma_q: torch.Tensor,
                        beta_q: torch.Tensor) -> torch.Tensor:
    """Deterministic integer LayerNorm. v: int64 [..., N] at Q.12.
    Returns int64 [..., N] at Q.12 (post-affine)."""
    n = v.shape[-1]
    s = v.sum(dim=-1, keepdim=True)
    c = n * v - s                                   # exact centered, x N*2^12
    q = (c * c).sum(dim=-1, keepdim=True)
    if bool((q >= (1 << 62)).any()):
        raise OverflowError("int LayerNorm variance exceeds 2^62")
    r = isqrt64_torch(torch.div(q, n, rounding_mode="floor")
                      + ln_eps_int(n))
    # n_fx = round_half_up(c * 2^12 / r): (2*c*2^12 + r) floor-div (2r)
    n_fx = torch.div(2 * c * (1 << NORM_FRAC) + r, 2 * r,
                     rounding_mode="floor")
    t = n_fx * gamma_q + beta_q                     # Q.12*Q.14 + Q.26
    return torch.div(t + (1 << (GAMMA_FRAC - 1)), 1 << GAMMA_FRAC,
                     rounding_mode="floor")         # -> Q.12


def silu_interp_int_torch(y: torch.Tensor,
                          table: torch.Tensor) -> torch.Tensor:
    """Integer interpolated SiLU. y: int64 Q4.12; returns int64 Q6.10."""
    lo = int(SILU_TAB_LO * (1 << NORM_FRAC))
    hi = -lo - 1
    yc = torch.clamp(y, lo, hi)
    u = yc - lo                                     # 0 .. 2^16-1
    i = u >> 8
    frac = u & 255
    t0 = table[i]
    t1 = table[i + 1]
    return t0 + torch.div((t1 - t0) * frac + 128, 256,
                          rounding_mode="floor")


def silu_interp_float_torch(y: torch.Tensor,
                            table: torch.Tensor) -> torch.Tensor:
    """Differentiable float twin of the integer table-SiLU (training
    forward): same table values, continuous interpolation fraction.
    Piecewise-linear, matches the integer version to <2^-10."""
    step = 1.0 / (1 << SILU_TAB_STEP_BITS)
    yc = torch.clamp(y, SILU_TAB_LO, -SILU_TAB_LO - step / 256)
    u = (yc - SILU_TAB_LO) / step
    i = torch.clamp(u.floor().long(), 0, SILU_TAB_N - 1)
    frac = u - i.float()
    tf = table.float() / (1 << SILU_TAB_FRAC)
    return tf[i] + (tf[i + 1] - tf[i]) * frac
