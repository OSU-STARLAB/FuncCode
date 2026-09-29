// G2 -- GRAM LSQ W4A4: dense INT4 weights (nibble-packed POD ROMs),
// 16-entry basis/SiLU LUTs, INT32 accumulation, fixed-point pre-LN
// requant, DETERMINISTIC integer LayerNorm, interpolated integer SiLU.
//
// FAIRNESS: loop structure, pragmas, II and unroll factors are IDENTICAL
// to hw/hls/gram_funccode_w4a4/top.cpp; ONLY the weight fetch differs
// (docs/HW_FAIRNESS.md). The LN/SiLU blocks are the shared
// hw/hls/common/int_layernorm.h used by BOTH quantized gram designs.

#include "hw_config.h"
#include "int_layernorm.h"
#include "params.h"

static void layer0(const act_t in[L0_IN], act_t out[L0_OUT]) {
#pragma HLS ARRAY_PARTITION variable=l0_lut_b complete dim=2
#pragma HLS RESOURCE variable=l0_basis_pk core=ROM_1P_BRAM
#pragma HLS RESOURCE variable=l0_base_pk core=ROM_1P_BRAM
  acc_t acc_s[L0_OUT], acc_b[L0_OUT];
#pragma HLS ARRAY_PARTITION variable=acc_s complete
#pragma HLS ARRAY_PARTITION variable=acc_b complete
INIT0:
  for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
    acc_s[o] = 0;
    acc_b[o] = 0;
  }
IN0:
  for (int i = 0; i < L0_IN; ++i) {
    uidx_t u = (uidx_t)(in[i] + 8);
    lut_t b[GRAM_BASIS_DIM];
#pragma HLS ARRAY_PARTITION variable=b complete
LUT0:
    for (int k = 0; k < GRAM_BASIS_DIM; ++k) {
#pragma HLS UNROLL
      b[k] = l0_lut_b[u][k];
    }
    lut_t s = l0_lut_s[u];
OUT0:
    for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
      unsigned int wrow = (unsigned int)l0_basis_pk[o * L0_IN + i]; // G2 fetch
      unsigned int brow = l0_base_pk[(o >> 3) * L0_IN + i];         // G2 fetch
      acc_t ss = 0;
MAC0:
      for (int k = 0; k < GRAM_BASIS_DIM; ++k) {
#pragma HLS UNROLL
        ss += (acc_t)w4_unpack(wrow, k) * (acc_t)b[k];              // G2 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += (acc_t)w4_unpack(brow, o & 7) * (acc_t)s;         // G2 fetch
    }
  }
  acc_t v[L0_OUT], y[L0_OUT];
PRE0:
  for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
    wide_t t = (wide_t)acc_s[o] * (wide_t)l0_mult_basis
             + (wide_t)acc_b[o] * (wide_t)l0_mult_base;
    v[o] = (acc_t)((t + ((wide_t)1 << (l0_shift - 1))) >> l0_shift);
  }
  int_layernorm<L0_OUT>(v, l0_gamma_q, l0_beta_q, l0_eps_int, y);
ACT0:
  for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
    lut_t so = silu_interp(y[o], silu_tab);
    out[o] = act_requant(so, l0_act_mult, l0_act_shift);
  }
}

static void layer1(const act_t in[L1_IN], acc_t out[L1_OUT]) {
#pragma HLS ARRAY_PARTITION variable=l1_lut_b complete dim=2
#pragma HLS RESOURCE variable=l1_basis_pk core=ROM_1P_BRAM
#pragma HLS RESOURCE variable=l1_base_pk core=ROM_1P_BRAM
  acc_t acc_s[L1_OUT], acc_b[L1_OUT];
#pragma HLS ARRAY_PARTITION variable=acc_s complete
#pragma HLS ARRAY_PARTITION variable=acc_b complete
INIT1:
  for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
    acc_s[o] = 0;
    acc_b[o] = 0;
  }
IN1:
  for (int i = 0; i < L1_IN; ++i) {
    uidx_t u = (uidx_t)(in[i] + 8);
    lut_t b[GRAM_BASIS_DIM];
#pragma HLS ARRAY_PARTITION variable=b complete
LUT1:
    for (int k = 0; k < GRAM_BASIS_DIM; ++k) {
#pragma HLS UNROLL
      b[k] = l1_lut_b[u][k];
    }
    lut_t s = l1_lut_s[u];
OUT1:
    for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
      unsigned int wrow = (unsigned int)l1_basis_pk[o * L1_IN + i]; // G2 fetch
      unsigned int brow = l1_base_pk[(o >> 3) * L1_IN + i];         // G2 fetch
      acc_t ss = 0;
MAC1:
      for (int k = 0; k < GRAM_BASIS_DIM; ++k) {
#pragma HLS UNROLL
        ss += (acc_t)w4_unpack(wrow, k) * (acc_t)b[k];              // G2 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += (acc_t)w4_unpack(brow, o & 7) * (acc_t)s;         // G2 fetch
    }
  }
  acc_t v[L1_OUT], y[L1_OUT];
PRE1:
  for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
    wide_t t = (wide_t)acc_s[o] * (wide_t)l1_mult_basis
             + (wide_t)acc_b[o] * (wide_t)l1_mult_base;
    v[o] = (acc_t)((t + ((wide_t)1 << (l1_shift - 1))) >> l1_shift);
  }
  int_layernorm<L1_OUT>(v, l1_gamma_q, l1_beta_q, l1_eps_int, y);
ACT1:
  for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
    lut_t so = silu_interp(y[o], silu_tab);
    out[o] = ((acc_t)so) << 6;   // Q6.10 -> INT32 logits at 2^-16
  }
}

void kan_top(const act_t in[L0_IN], acc_t out[L1_OUT]) {
#pragma HLS INTERFACE ap_memory port=in
#pragma HLS INTERFACE ap_memory port=out
#pragma HLS INTERFACE ap_ctrl_hs port=return
  act_t hidden[L0_OUT];
  layer0(in, hidden);
  layer1(hidden, out);
}
