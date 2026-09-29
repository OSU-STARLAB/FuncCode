"""QAT wrappers for the GRAM (KAGN-basis FC) designs G2/G3.

Same contract philosophy as the spline wrappers (qat.py/act_quant.py):
fake-quant at (a) layer inputs A4, (b) basis coefficients W4 per-tensor,
(c) base weights W4 per-tensor, (d) inter-layer activations A4 — plus the
GRAM-specific frozen pieces (gram_spec.py): Q.12 pre-LN grid, integer
LayerNorm, interpolated-table SiLU.

forward() dispatches on self.training:
  train: differentiable — float LayerNorm, float twin of the table SiLU,
         STE rounding onto every HW grid (train == deploy up to LN).
  eval:  the DEPLOYED integer arithmetic for LN/SiLU/logits — this is the
         L1 reference and the accuracy that gets reported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from funcodekan.models.variants import BranchCodebookKAN, DirectKANVariant

from .hw_spec import (lut_quantize_basis_ste, lut_quantize_silu_ste,
                      round_half_up)
from .gram_spec import (LOGIT_SHIFT, PRE_LN_FRAC, SILU_TAB_FRAC, SILU_TABLE,
                        int_layernorm_torch, quantize_ln_affine,
                        silu_interp_float_torch, silu_interp_int_torch)
from .lsq import LsqQuantizer

_PRE_SCALE = 1 << PRE_LN_FRAC


def _q12_ste(v: torch.Tensor) -> torch.Tensor:
    q = round_half_up(v * _PRE_SCALE) / _PRE_SCALE
    return (q - v).detach() + v


def gram_quant_forward(layers, x, weight_fn, act_quants, silu_table,
                       int_eval: bool):
    """Shared G2/G3 forward. weight_fn(l) -> fake-quant ([out,in,4],
    [out,in]). Returns float logits (train) or int64 Q.16 logits (eval)."""
    n_layers = len(layers)
    for l, layer in enumerate(layers):
        x = act_quants[l](x)
        basis_w, base_w = weight_fn(l)
        basis = lut_quantize_basis_ste(layer.basis(x))       # Q2.14 grid
        silu_in = lut_quantize_silu_ste(F.silu(x))           # Q6.10 grid
        pre = (torch.einsum("nik,oik->no", basis, basis_w)
               + F.linear(silu_in, base_w))
        if int_eval:
            v = round_half_up(pre * _PRE_SCALE).to(torch.int64)
            g_q, b_q = quantize_ln_affine(layer.norm.weight, layer.norm.bias)
            y_ln = int_layernorm_torch(v, g_q.to(v.device), b_q.to(v.device))
            s_out = silu_interp_int_torch(y_ln, silu_table.to(v.device))
            if l < n_layers - 1:
                x = s_out.float() * (2.0 ** -SILU_TAB_FRAC)
            else:
                # INT32 logits at 2^-16, returned as EXACT floats at real
                # scale (|int| < 2^24 so float32 is lossless) so the
                # verified evaluate()'s cross-entropy stays computable.
                return (s_out << LOGIT_SHIFT).float() * 2.0 ** -16
        else:
            pre = _q12_ste(pre)
            y = layer.norm(pre)
            s = silu_interp_float_torch(y, silu_table.to(pre.device))
            if l < n_layers - 1:
                x = s
            else:
                return s
    raise RuntimeError("unreachable")


class _GramQuantBase(nn.Module):
    """Common quantizer plumbing for the two GRAM wrappers."""

    def _init_quants(self, n_layers: int):
        self.act_quants = nn.ModuleList(
            [LsqQuantizer(4, signed=True) for _ in range(n_layers)])
        self.w_basis_quants = nn.ModuleList(
            [LsqQuantizer(4, signed=True) for _ in range(n_layers)])
        self.w_base_quants = nn.ModuleList(
            [LsqQuantizer(4, signed=True) for _ in range(n_layers)])
        self.register_buffer("silu_table",
                             torch.from_numpy(SILU_TABLE.copy()))

    def act_steps(self):
        return [q.step_size() for q in self.act_quants]

    def ln_consts(self, l: int):
        layer = self.gram_layers()[l]
        g_q, b_q = quantize_ln_affine(layer.norm.weight, layer.norm.bias)
        return {"gamma_q": g_q.cpu(), "beta_q": b_q.cpu(),
                "n_features": layer.out_features}

    def forward(self, x):
        return gram_quant_forward(self.gram_layers(), x, self.quant_weights,
                                  self.act_quants, self.silu_table,
                                  int_eval=not self.training)


class QuantGramKAN(_GramQuantBase):
    """G2: LSQ W4A4 wrapper around an untouched DirectKANVariant('gram')."""

    def __init__(self, dense: DirectKANVariant):
        super().__init__()
        assert dense.variant == "gram"
        self.dense = dense
        self._init_quants(len(dense.layers))
        for l, w in enumerate(dense.weights):
            self.w_basis_quants[l].init_from(w[..., :-1])
            self.w_base_quants[l].init_from(w[..., -1])

    def gram_layers(self):
        return self.dense.layers

    def quant_weights(self, l: int):
        w = self.dense.weights[l]
        return (self.w_basis_quants[l](w[..., :-1]),
                self.w_base_quants[l](w[..., -1]))

    @torch.no_grad()
    def int_weights(self, l: int):
        w = self.dense.weights[l].detach().cpu()
        bq, sq = self.w_base_quants[l], self.w_basis_quants[l]
        return {"basis_q": sq.quantize_int(w[..., :-1]),
                "base_q": bq.quantize_int(w[..., -1]),
                "basis_step": sq.step_size(),
                "base_step": bq.step_size()}


def rebuild_gram_branch(state_dict: dict, input_dim=784, hidden_width=64,
                        output_dim=10, degree=3) -> BranchCodebookKAN:
    """Reconstruct the verified BranchCodebookKAN('gram') from a
    clustered_finetuned.pt state_dict (codebooks are parameters
    'spline_codebooks.{i}'/'base_codebooks.{i}', ids are buffers
    'spline_cluster_ids_{i}'/'base_cluster_ids_{i}', LN params live under
    'layers.{i}.norm.*')."""
    n = len([k for k in state_dict if k.startswith("spline_codebooks.")])
    dense_like = DirectKANVariant("gram", input_dim, hidden_width,
                                  output_dim, degree=degree)
    model = BranchCodebookKAN(
        dense_like,
        [state_dict[f"spline_codebooks.{i}"] for i in range(n)],
        [state_dict[f"spline_cluster_ids_{i}"] for i in range(n)],
        [state_dict[f"base_codebooks.{i}"] for i in range(n)],
        [state_dict[f"base_cluster_ids_{i}"] for i in range(n)],
        train_codebooks=True)
    model.load_state_dict(state_dict)
    return model


class QuantGramBranchKAN(_GramQuantBase):
    """G3: FuncCode W4A4 — per-tensor LSQ on the codebooks (identical
    contract to spline D3), frozen indices, integer LN/SiLU datapath."""

    def __init__(self, branch: BranchCodebookKAN):
        super().__init__()
        self.branch = branch
        self._init_quants(len(branch.layers))
        for l in range(len(branch.layers)):
            self.w_basis_quants[l].init_from(branch.spline_codebooks[l])
            self.w_base_quants[l].init_from(branch.base_codebooks[l])

    def gram_layers(self):
        return self.branch.layers

    def quant_weights(self, l: int):
        basis_cb = self.w_basis_quants[l](self.branch.spline_codebooks[l])
        base_cb = self.w_base_quants[l](self.branch.base_codebooks[l])
        sids = self.branch.get_spline_cluster_ids(l)
        bids = self.branch.get_base_cluster_ids(l)
        return basis_cb[sids], base_cb[bids].squeeze(-1)

    @torch.no_grad()
    def int_codebooks(self, l: int):
        scb = self.branch.spline_codebooks[l].detach().cpu()
        bcb = self.branch.base_codebooks[l].detach().cpu()
        sq, bq = self.w_basis_quants[l], self.w_base_quants[l]
        return {"basis_codebook_q": sq.quantize_int(scb),
                "base_codebook_q": bq.quantize_int(bcb).squeeze(-1),
                "basis_ids": self.branch.get_spline_cluster_ids(l).cpu(),
                "base_ids": self.branch.get_base_cluster_ids(l).cpu(),
                "basis_step": sq.step_size(),
                "base_step": bq.step_size()}
