"""Frozen numeric contract shared by QAT, the fixed-point emulator and the HLS.

Every constant and rounding rule here is mirrored in hw/hls/common/hw_config.h.
Changing anything here invalidates trained checkpoints, golden vectors and the
HLS designs simultaneously — see docs/HW_DESIGN_CONTRACT.md ("Frozen design
decisions") and docs/HW_FAIRNESS.md before touching it.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Bit widths and ranges (signed symmetric INT4 everywhere, per-tensor scales)
# ---------------------------------------------------------------------------
WEIGHT_BITS = 4
ACT_BITS = 4
QN = -(2 ** (ACT_BITS - 1))          # -8
QP = 2 ** (ACT_BITS - 1) - 1         # +7
NUM_ACT_LEVELS = 2 ** ACT_BITS      # 16 -> 16-entry basis/SiLU LUTs

# ---------------------------------------------------------------------------
# LUT fixed-point formats, both int16, per-branch (the requant multipliers
# are per-branch anyway, so mixed formats cost nothing in HW):
#   basis LUT: Q2.14 — B-spline bases are a partition of unity in [0,1], so
#              2^14 scaling always fits int16 (max 16384) with max precision.
#   SiLU LUT:  Q6.10 — covers (-32, +32); needs act_step < 32/7 ~ 4.57
#              (export asserts). SiLU(x) ~ x at the +7*step extreme, and
#              trained hidden steps run ~3, so Q5.11's +-16 was too tight.
# ---------------------------------------------------------------------------
BASIS_FRAC_BITS = 14
SILU_FRAC_BITS = 10
BASIS_SCALE = 1 << BASIS_FRAC_BITS   # 16384
SILU_SCALE = 1 << SILU_FRAC_BITS     # 2048
LUT_INT_MIN = -(2 ** 15)
LUT_INT_MAX = 2 ** 15 - 1

# ---------------------------------------------------------------------------
# Requantization / logits
# ---------------------------------------------------------------------------
MULT_MAX = 2 ** 31 - 1               # requant multipliers are int32
# Logits are int32 ~ logit * 2^16. 16 fractional bits so that argmax ties at
# the logit grid are far below any real decision margin (resolution 1.5e-5;
# MNIST logits span ~+-50, 50*2^16 << 2^31).
LOGIT_FRAC_BITS = 16

# FPGA target (single source of truth; hw/tcl scripts carry the same values).
# NOT the ZU7EV of the original spec: D1's 1.744 MiB of FP32 constant ROMs
# needs ceil(14,635,008 b / 36 Kib) = 397 BRAM36, ZU7EV has 312, and US+ URAM
# cannot hold initialized ROMs (no bitstream init). Adjusted for ALL THREE
# designs identically; see docs/HW_DESIGN_CONTRACT.md.
FPGA_PART = "xczu9eg-ffvb1156-2-e"   # ZCU102-class, 912 BRAM36
CLOCK_PERIOD_NS = 6.67               # 150 MHz


def round_half_up(x: torch.Tensor) -> torch.Tensor:
    """The one rounding rule of the HW: floor(x + 0.5).

    Used for input quantization, LUT entry generation and (in integer form,
    via +2^(shift-1) >> shift) requantization. Deliberately NOT torch.round
    (half-to-even).
    """
    return torch.floor(x + 0.5)


def lut_quantize_basis(v: torch.Tensor) -> torch.Tensor:
    """Round basis values to the Q2.14 LUT grid (float values on the grid;
    integer LUT entries are value * BASIS_SCALE)."""
    return round_half_up(v * BASIS_SCALE) / BASIS_SCALE


def lut_quantize_silu(v: torch.Tensor) -> torch.Tensor:
    """Round SiLU values to the Q5.11 LUT grid."""
    return round_half_up(v * SILU_SCALE) / SILU_SCALE


def _ste(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (q - v).detach() + v


def lut_quantize_basis_ste(v: torch.Tensor) -> torch.Tensor:
    """Basis LUT rounding with a straight-through gradient (QAT forward ==
    HW forward)."""
    return _ste(lut_quantize_basis(v), v)


def lut_quantize_silu_ste(v: torch.Tensor) -> torch.Tensor:
    return _ste(lut_quantize_silu(v), v)


def requant_multiplier(ratios: list[float]) -> tuple[list[int], int]:
    """Fixed-point multipliers for one requant site.

    Given the real ratios (one per branch, e.g. [s_ws*2^-12/s_out,
    s_wb*2^-12/s_out]), pick the largest common SHIFT such that every
    multiplier round(ratio * 2^SHIFT) fits in int32, and return
    ([M_0, M_1, ...], SHIFT). Frozen scheme of HW_DESIGN_CONTRACT.md.
    """
    max_ratio = max(abs(r) for r in ratios)
    if max_ratio <= 0:
        raise ValueError("all requant ratios are zero")
    shift = int(math.floor(math.log2(MULT_MAX / max_ratio)))
    # guard against float edge cases
    while any(int(math.floor(r * (1 << shift) + 0.5)) > MULT_MAX for r in ratios):
        shift -= 1
    mults = [int(math.floor(r * (1 << shift) + 0.5)) for r in ratios]
    return mults, shift
