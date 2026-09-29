"""QAT wrappers for the KAGN-Conv designs K2/K3 (see kagn_conv.py).

Quantization contract (mirrors the other families, frozen):
  A4 (per-tensor LSQ): network input, after conv1, after conv2, and the
      pooled head input. W4 (per-tensor LSQ): poly + base weights of each
      conv layer and of the FC gram head (K3: on the codebooks, indices
      frozen).
  Conv integer datapath: input-value basis/SiLU LUTs (Q2.14/Q6.10), INT32
      accumulators, BatchNorm FOLDED (running stats) into per-channel
      requant multipliers + offset to the Q4.12 grid, shared interpolated
      integer SiLU table, per-tensor activation requant. Zero-padding
      contributes nothing (the float model pads the basis/SiLU MAPS with
      zeros). Pool = integer sum of 49 codes, folded 1/49 into the head
      input requant. Head reuses the gram integer-LN datapath.

forward() dispatches on self.training (train: float BN/LN, differentiable;
eval: deployed arithmetic — the L1 reference).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hw_spec import (lut_quantize_basis_ste, lut_quantize_silu_ste,
                      round_half_up)
from .gram_spec import (LOGIT_SHIFT, PRE_LN_FRAC, SILU_TAB_FRAC, SILU_TABLE,
                        int_layernorm_torch, quantize_ln_affine,
                        silu_interp_float_torch, silu_interp_int_torch)
from .kagn_conv import (CompressedKagnConvNet, KagnConvNet, gram_basis)
from .lsq import LsqQuantizer

_PRE = 1 << PRE_LN_FRAC


def _q12_ste(v):
    q = round_half_up(v * _PRE) / _PRE
    return (q - v).detach() + v


class _KagnQuantBase(nn.Module):
    """K2/K3 shared plumbing. Subclasses provide weight_of(i) (fake-quant
    [.., COEFF_DIM] edge tensor for stage i: 0=conv1, 1=conv2, 2=head)."""

    N_STAGES = 3

    def _init_quants(self):
        self.act_quants = nn.ModuleList(
            [LsqQuantizer(4, signed=True) for _ in range(4)])
        self.w_poly_quants = nn.ModuleList(
            [LsqQuantizer(4, signed=True) for _ in range(self.N_STAGES)])
        self.w_base_quants = nn.ModuleList(
            [LsqQuantizer(4, signed=True) for _ in range(self.N_STAGES)])
        self.register_buffer("silu_table",
                             torch.from_numpy(SILU_TABLE.copy()))

    def act_steps(self):
        return [q.step_size() for q in self.act_quants]

    def convs(self):
        return (self.net.conv1, self.net.conv2)

    def head(self):
        return self.net.head_layer

    # ---- forward ---------------------------------------------------------

    def _conv_stage(self, i, x):
        conv = self.convs()[i]
        w = self.weight_of(i)
        basis = lut_quantize_basis_ste(gram_basis(x, conv.degree))
        silu = lut_quantize_silu_ste(F.silu(x))
        pre = conv.pre_norm_from_weight(x, w, basis_x=basis, silu_x=silu)
        if self.training:
            y = conv.norm(pre)
            return silu_interp_float_torch(y, self.silu_table)
        # deployed arithmetic: Q.12 grid -> integer instance norm
        # (per-channel over spatial positions; reuses the frozen integer
        # LayerNorm kernel) -> integer table SiLU
        n, o, hh, ww = pre.shape
        v = round_half_up(pre * _PRE).to(torch.int64).reshape(n, o, hh * ww)
        g_q, b_q = quantize_ln_affine(conv.norm.weight, conv.norm.bias)
        y = int_layernorm_torch(v, g_q.view(-1, 1).to(v.device),
                                b_q.view(-1, 1).to(v.device))
        so = silu_interp_int_torch(y.reshape(n, o, hh, ww),
                                   self.silu_table)
        return so.float() * (2.0 ** -SILU_TAB_FRAC)

    def forward(self, x):
        x = self.act_quants[0](x)
        x = self._conv_stage(0, x)
        x = self.act_quants[1](x)
        x = self._conv_stage(1, x)
        x = self.act_quants[2](x)
        x = x.mean(dim=(2, 3))                     # pool on dequant values
        x = self.act_quants[3](x)
        # head: gram FC math
        head = self.head()
        w = self.weight_of(2)
        basis = lut_quantize_basis_ste(head.basis(x))
        silu = lut_quantize_silu_ste(F.silu(x))
        pre = (torch.einsum("nik,oik->no", basis, w[..., :-1])
               + F.linear(silu, w[..., -1]))
        if self.training:
            y = head.norm(_q12_ste(pre))
            return silu_interp_float_torch(y, self.silu_table)
        v = round_half_up(pre * _PRE).to(torch.int64)
        g_q, b_q = quantize_ln_affine(head.norm.weight, head.norm.bias)
        y = int_layernorm_torch(v, g_q.to(v.device), b_q.to(v.device))
        so = silu_interp_int_torch(y, self.silu_table)
        return (so << LOGIT_SHIFT).float() * 2.0 ** -16


class QuantKagnConvNet(_KagnQuantBase):
    """K2: LSQ W4A4 wrapper around the dense KagnConvNet."""

    def __init__(self, net: KagnConvNet):
        super().__init__()
        self.net = net
        self._init_quants()
        for i, w in enumerate([net.conv1.weight, net.conv2.weight,
                               net.head_weight]):
            self.w_poly_quants[i].init_from(w[..., :-1])
            self.w_base_quants[i].init_from(w[..., -1])

    def weight_of(self, i):
        w = [self.net.conv1.weight, self.net.conv2.weight,
             self.net.head_weight][i]
        poly = self.w_poly_quants[i](w[..., :-1])
        base = self.w_base_quants[i](w[..., -1])
        return torch.cat([poly, base.unsqueeze(-1)], dim=-1)

    @torch.no_grad()
    def int_weights(self, i):
        w = [self.net.conv1.weight, self.net.conv2.weight,
             self.net.head_weight][i].detach().cpu()
        pq, bq = self.w_poly_quants[i], self.w_base_quants[i]
        return {"poly_q": pq.quantize_int(w[..., :-1]),
                "base_q": bq.quantize_int(w[..., -1]),
                "poly_step": pq.step_size(), "base_step": bq.step_size()}


class QuantKagnBranchNet(_KagnQuantBase):
    """K3: FuncCode W4A4 — per-tensor LSQ on codebooks, indices frozen."""

    def __init__(self, comp: CompressedKagnConvNet):
        super().__init__()
        self.net = comp
        self._init_quants()
        for i in range(self.N_STAGES):
            self.w_poly_quants[i].init_from(comp.spline_codebooks[i])
            self.w_base_quants[i].init_from(comp.base_codebooks[i])

    def weight_of(self, i):
        poly_cb = self.w_poly_quants[i](self.net.spline_codebooks[i])
        base_cb = self.w_base_quants[i](self.net.base_codebooks[i])
        sids = self.net.get_spline_cluster_ids(i)
        bids = self.net.get_base_cluster_ids(i)
        poly = poly_cb[sids]
        base = base_cb[bids].squeeze(-1)
        return torch.cat([poly, base.unsqueeze(-1)], dim=-1)

    @torch.no_grad()
    def int_codebooks(self, i):
        pcb = self.net.spline_codebooks[i].detach().cpu()
        bcb = self.net.base_codebooks[i].detach().cpu()
        pq, bq = self.w_poly_quants[i], self.w_base_quants[i]
        return {"poly_codebook_q": pq.quantize_int(pcb),
                "base_codebook_q": bq.quantize_int(bcb).squeeze(-1),
                "poly_ids": self.net.get_spline_cluster_ids(i).cpu(),
                "base_ids": self.net.get_base_cluster_ids(i).cpu(),
                "poly_step": pq.step_size(), "base_step": bq.step_size()}
