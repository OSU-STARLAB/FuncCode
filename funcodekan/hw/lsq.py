"""LSQ — Learned Step Size Quantization (Esser et al., ICLR 2020).

Used for both weights and activations of the D2/D3 W4A4 models. Per-tensor
step sizes only (HW contract; see docs/HW_FAIRNESS.md). Rounding is the HW
rule floor(x+0.5) — round-half-up — in BOTH training STE and eval, so the
fake-quant model never drifts from the emulator/HLS rounding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .hw_spec import round_half_up


def _grad_scale(x: torch.Tensor, g: float) -> torch.Tensor:
    """Scale the gradient of x by g without changing its value."""
    return (x - x * g).detach() + x * g


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round-half-up with a straight-through gradient."""
    return (round_half_up(x) - x).detach() + x


class LsqQuantizer(nn.Module):
    """Fake-quantizer with a learnable step size.

    forward(x) returns fake-quantized values q*s with q = clamp(round(x/s)).
    quantize_int(x) returns the integer codes (for export/emulator use).

    Step init: s = 2*mean(|x|)/sqrt(Qp), taken from the first forward in
    training mode (or immediately via init_from(tensor)).
    Gradient scaling: g = 1/sqrt(numel(x)*Qp) per the LSQ paper.
    """

    def __init__(self, bits: int = 4, signed: bool = True,
                 per_channel: bool = False):
        super().__init__()
        if per_channel:
            # Contract is per-tensor; kept in the signature for the fallback
            # fallback path (would need a HW contract update to enable).
            raise NotImplementedError(
                "per-channel LSQ is not part of the frozen HW contract")
        self.bits = bits
        self.signed = signed
        if signed:
            self.qn = -(2 ** (bits - 1))
            self.qp = 2 ** (bits - 1) - 1
        else:
            self.qn = 0
            self.qp = 2 ** bits - 1
        self.s = nn.Parameter(torch.ones(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def init_from(self, x: torch.Tensor) -> None:
        mean_abs = x.detach().abs().mean()
        init = 2.0 * mean_abs / math.sqrt(self.qp)
        self.s.copy_(torch.clamp(init, min=1e-8))
        self.initialized.fill_(True)

    def step_size(self) -> float:
        return float(self.s.detach().abs().clamp(min=1e-8))

    def _effective_step(self, x: torch.Tensor) -> torch.Tensor:
        g = 1.0 / math.sqrt(x.numel() * self.qp)
        return _grad_scale(self.s.abs().clamp(min=1e-8), g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized):
            if self.training:
                self.init_from(x)
            else:
                raise RuntimeError(
                    "LsqQuantizer used in eval mode before step init")
        s = self._effective_step(x)
        q = _round_ste(torch.clamp(x / s, self.qn, self.qp))
        return q * s

    @torch.no_grad()
    def quantize_int(self, x: torch.Tensor) -> torch.Tensor:
        """Integer codes exactly as the HW computes them."""
        s = self.step_size()
        return torch.clamp(round_half_up(x / s),
                           self.qn, self.qp).to(torch.int32)
