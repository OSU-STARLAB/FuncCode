// G1 -- GRAM FP32 dense baseline: float weights in on-chip ROMs, runtime
// FP32 tanh-polynomial basis + SiLU, float accumulation, float LayerNorm
// + SiLU output (gram_basis_fp32.h replicates the verified layer math).
//
// FAIRNESS: same INIT/IN/OUT/MAC/PRE/LN/ACT skeleton and II/unroll
// directives as the two quantized gram tops; only the numeric format
// differs (docs/HW_FAIRNESS.md).

#include "hw_config.h"
#include "gram_basis_fp32.h"
#include "params.h"

static void layer0(const float in[L0_IN], float out[L0_OUT]) {
#pragma HLS ARRAY_PARTITION variable=l0_weight complete dim=2
#pragma HLS RESOURCE variable=l0_weight core=ROM_1P_BRAM
  float acc_s[L0_OUT], acc_b[L0_OUT];
#pragma HLS ARRAY_PARTITION variable=acc_s complete
#pragma HLS ARRAY_PARTITION variable=acc_b complete
INIT0:
  for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
    acc_s[o] = 0.0f;
    acc_b[o] = 0.0f;
  }
IN0:
  for (int i = 0; i < L0_IN; ++i) {
    float x = in[i];
    float b[GRAM_BDIM];
#pragma HLS ARRAY_PARTITION variable=b complete
    gram_basis_f32(x, b);
    float s = gram_silu_f32(x);
OUT0:
    for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
      float ss = 0.0f;
MAC0:
      for (int k = 0; k < GRAM_BDIM; ++k) {
#pragma HLS UNROLL
        ss += l0_weight[o * L0_IN + i][k] * b[k];                   // G1 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += l0_weight[o * L0_IN + i][GRAM_BDIM] * s;          // G1 fetch
    }
  }
  float pre[L0_OUT];
PRE0:
  for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
    pre[o] = acc_s[o] + acc_b[o];
  }
  gram_ln_silu_f32<L0_OUT>(pre, l0_gamma, l0_beta, out);
}

static void layer1(const float in[L1_IN], float out[L1_OUT]) {
#pragma HLS ARRAY_PARTITION variable=l1_weight complete dim=2
#pragma HLS RESOURCE variable=l1_weight core=ROM_1P_BRAM
  float acc_s[L1_OUT], acc_b[L1_OUT];
#pragma HLS ARRAY_PARTITION variable=acc_s complete
#pragma HLS ARRAY_PARTITION variable=acc_b complete
INIT1:
  for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
    acc_s[o] = 0.0f;
    acc_b[o] = 0.0f;
  }
IN1:
  for (int i = 0; i < L1_IN; ++i) {
    float x = in[i];
    float b[GRAM_BDIM];
#pragma HLS ARRAY_PARTITION variable=b complete
    gram_basis_f32(x, b);
    float s = gram_silu_f32(x);
OUT1:
    for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
      float ss = 0.0f;
MAC1:
      for (int k = 0; k < GRAM_BDIM; ++k) {
#pragma HLS UNROLL
        ss += l1_weight[o * L1_IN + i][k] * b[k];                   // G1 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += l1_weight[o * L1_IN + i][GRAM_BDIM] * s;          // G1 fetch
    }
  }
  float pre[L1_OUT];
PRE1:
  for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
    pre[o] = acc_s[o] + acc_b[o];
  }
  gram_ln_silu_f32<L1_OUT>(pre, l1_gamma, l1_beta, out);
}

void kan_top(const float in[L0_IN], float out[L1_OUT]) {
#pragma HLS INTERFACE ap_memory port=in
#pragma HLS INTERFACE ap_memory port=out
#pragma HLS INTERFACE ap_ctrl_hs port=return
  float hidden[L0_OUT];
  layer0(in, hidden);
  layer1(hidden, out);
}
