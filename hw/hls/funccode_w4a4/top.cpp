// D3 -- FuncCode W4A4: INT4 codebooks + per-edge index streams
// (spline idx 5 b, base idx 4 b), 16-entry basis/SiLU LUTs, INT32
// accumulation, fixed-point requantization.
//
// FAIRNESS: loop structure, pragmas, II and unroll factors are IDENTICAL to
// hw/hls/lsq_w4a4/top.cpp. The ONLY difference between the two files is the
// weight fetch: index-stream -> codebook BRAM lookup here vs dense ROM read
// there. Keep the files in lockstep (docs/HW_FAIRNESS.md).

#include "hw_config.h"
#include "requant.h"
#include "params.h"

static void layer0(const act_t in[L0_IN], act_t out[L0_OUT]) {
#pragma HLS ARRAY_PARTITION variable=l0_lut_b complete dim=2
#pragma HLS ARRAY_PARTITION variable=l0_spline_codebook_q complete dim=2
#pragma HLS RESOURCE variable=l0_spline_ids core=ROM_1P_BRAM
#pragma HLS RESOURCE variable=l0_base_ids core=ROM_1P_BRAM
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
    lut_t b[8];
#pragma HLS ARRAY_PARTITION variable=b complete
LUT0:
    for (int k = 0; k < 8; ++k) {
#pragma HLS UNROLL
      b[k] = l0_lut_b[u][k];
    }
    lut_t s = l0_lut_s[u];
OUT0:
    for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
      sidx_t sid = l0_spline_ids[o * L0_IN + i];                    // D3 fetch
      bidx_t bid = l0_base_ids[o * L0_IN + i];                      // D3 fetch
      acc_t ss = 0;
MAC0:
      for (int k = 0; k < 8; ++k) {
#pragma HLS UNROLL
        ss += (acc_t)l0_spline_codebook_q[sid][k] * (acc_t)b[k];    // D3 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += (acc_t)l0_base_codebook_q[bid] * (acc_t)s;        // D3 fetch
    }
  }
REQ0:
  for (int o = 0; o < L0_OUT; ++o) {
#pragma HLS PIPELINE II=1
    out[o] = requant_code(acc_s[o], acc_b[o],
                          l0_mult_spline, l0_mult_base, l0_shift);
  }
}

static void layer1(const act_t in[L1_IN], acc_t out[L1_OUT]) {
#pragma HLS ARRAY_PARTITION variable=l1_lut_b complete dim=2
#pragma HLS ARRAY_PARTITION variable=l1_spline_codebook_q complete dim=2
#pragma HLS RESOURCE variable=l1_spline_ids core=ROM_1P_BRAM
#pragma HLS RESOURCE variable=l1_base_ids core=ROM_1P_BRAM
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
    lut_t b[8];
#pragma HLS ARRAY_PARTITION variable=b complete
LUT1:
    for (int k = 0; k < 8; ++k) {
#pragma HLS UNROLL
      b[k] = l1_lut_b[u][k];
    }
    lut_t s = l1_lut_s[u];
OUT1:
    for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
      sidx_t sid = l1_spline_ids[o * L1_IN + i];                    // D3 fetch
      bidx_t bid = l1_base_ids[o * L1_IN + i];                      // D3 fetch
      acc_t ss = 0;
MAC1:
      for (int k = 0; k < 8; ++k) {
#pragma HLS UNROLL
        ss += (acc_t)l1_spline_codebook_q[sid][k] * (acc_t)b[k];    // D3 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += (acc_t)l1_base_codebook_q[bid] * (acc_t)s;        // D3 fetch
    }
  }
REQ1:
  for (int o = 0; o < L1_OUT; ++o) {
#pragma HLS PIPELINE II=1
    out[o] = requant_logit(acc_s[o], acc_b[o],
                           l1_mult_spline, l1_mult_base, l1_shift);
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
