// D1 -- FP32 dense baseline: float weights in on-chip ROMs, runtime FP32
// Cox-de Boor basis + SiLU (per-precision-class algorithm; D2/D3 use the
// A4 16-entry LUTs instead), float accumulation, float logits.
//
// FAIRNESS: same layer/loop skeleton, same I/O style, same INIT/IN/OUT/MAC
// loop nest and II/unroll factors as the D2/D3 tops. Only the numeric
// format differs (docs/HW_FAIRNESS.md).

#include "hw_config.h"
#include "bspline_fp32.h"
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
    float b[8];
#pragma HLS ARRAY_PARTITION variable=b complete
    bspline_basis_f32(x, l0_grid, b);
    float s = silu_f32(x);
OUT0:
    for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
      float ss = 0.0f;
MAC0:
      for (int k = 0; k < 8; ++k) {
#pragma HLS UNROLL
        ss += l0_weight[o * L0_IN + i][k] * b[k];                   // D1 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += l0_weight[o * L0_IN + i][8] * s;                  // D1 fetch
    }
  }
REQ0:
  for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
    out[o] = acc_s[o] + acc_b[o];
  }
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
    float b[8];
#pragma HLS ARRAY_PARTITION variable=b complete
    bspline_basis_f32(x, l1_grid, b);
    float s = silu_f32(x);
OUT1:
    for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
      float ss = 0.0f;
MAC1:
      for (int k = 0; k < 8; ++k) {
#pragma HLS UNROLL
        ss += l1_weight[o * L1_IN + i][k] * b[k];                   // D1 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += l1_weight[o * L1_IN + i][8] * s;                  // D1 fetch
    }
  }
REQ1:
  for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
    out[o] = acc_s[o] + acc_b[o];
  }
}

void kan_top(const float in[L0_IN], float out[L1_OUT]) {
#pragma HLS INTERFACE ap_memory port=in
#pragma HLS INTERFACE ap_memory port=out
#pragma HLS INTERFACE ap_ctrl_hs port=return
  float hidden[L0_OUT];
  layer0(in, hidden);
  layer1(hidden, out);
}
