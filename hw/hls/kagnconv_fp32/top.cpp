// K1 -- KAGN-Conv FP32 dense baseline: float weights in on-chip ROMs,
// runtime FP32 tanh-polynomial basis + SiLU per tap, float accumulation,
// BatchNorm (inference = folded per-channel affine, float), float average
// pool, gram-FC head with float LayerNorm+SiLU.
//
// FAIRNESS: same LOAD/CONV(Y,X,TAP,O)/REQ/POOL/HEAD skeleton and II/unroll
// directives as the K2/K3 tops; only the numeric format differs.

#include "hw_config.h"
#include "gram_basis_fp32.h"
#include "params.h"

template <int CIN, int COUT, int IH, int IW, int OH, int OW>
void conv_stage(const float fin[CIN][IH][IW], float fout[COUT][OH][OW],
                const float weight[][5], const float gamma[COUT],
                const float beta[COUT]) {
  float v_map[COUT][OH * OW];
  float acc[COUT];
#pragma HLS ARRAY_PARTITION variable=acc complete
CONV_Y:
  for (int y = 0; y < OH; ++y) {
CONV_X:
    for (int x = 0; x < OW; ++x) {
INIT:
      for (int o = 0; o < COUT; ++o) {
#pragma HLS PIPELINE II=1
        acc[o] = 0.0f;
      }
TAP_I:
      for (int i = 0; i < CIN; ++i) {
TAP_K:
        for (int t = 0; t < 9; ++t) {
          int ky = t / 3, kx = t % 3;
          int iy = y * CONV_STRIDE + ky - CONV_PAD;
          int ix = x * CONV_STRIDE + kx - CONV_PAD;
          bool inb = (iy >= 0) && (iy < IH) && (ix >= 0) && (ix < IW);
          float xv = inb ? fin[i][iy][ix] : 0.0f;
          float b[GRAM_BDIM];
#pragma HLS ARRAY_PARTITION variable=b complete
          gram_basis_f32(xv, b);
          float s = gram_silu_f32(xv);
          if (!inb) {
LUTZ:
            for (int k = 0; k < GRAM_BDIM; ++k) {
#pragma HLS UNROLL
              b[k] = 0.0f;
            }
            s = 0.0f;
          }
O_LOOP:
          for (int o = 0; o < COUT; ++o) {
#pragma HLS PIPELINE II=1
            int edge = ((o * CIN + i) * 3 + ky) * 3 + kx;
            float ss = 0.0f;
MACK:
            for (int k = 0; k < GRAM_BDIM; ++k) {
#pragma HLS UNROLL
              ss += weight[edge][k] * b[k];                     // K1 fetch
            }
            acc[o] += ss + weight[edge][GRAM_BDIM] * s;         // K1 fetch
          }
        }
      }
REQ:
      for (int o = 0; o < COUT; ++o) {
#pragma HLS PIPELINE II=1
        v_map[o][y * OW + x] = acc[o];
      }
    }
  }
NORM_O:
  for (int o = 0; o < COUT; ++o) {
    float yb[OH * OW];
    gram_in_silu_f32<OH * OW>(v_map[o], gamma[o], beta[o], yb);
ACT_P:
    for (int p = 0; p < OH * OW; ++p) {
#pragma HLS PIPELINE II=1
      fout[o][p / OW][p % OW] = yb[p];
    }
  }
}

static void head_stage(const float hx[HEAD_IN], float out[HEAD_OUT]) {
  float acc[HEAD_OUT];
#pragma HLS ARRAY_PARTITION variable=acc complete
HINIT:
  for (int o = 0; o < HEAD_OUT; ++o) {
#pragma HLS PIPELINE II=1
    acc[o] = 0.0f;
  }
HIN:
  for (int i = 0; i < HEAD_IN; ++i) {
    float xv = hx[i];
    float b[GRAM_BDIM];
#pragma HLS ARRAY_PARTITION variable=b complete
    gram_basis_f32(xv, b);
    float s = gram_silu_f32(xv);
HOUT:
    for (int o = 0; o < HEAD_OUT; ++o) {
#pragma HLS PIPELINE II=1
      float ss = 0.0f;
HMAC:
      for (int k = 0; k < GRAM_BDIM; ++k) {
#pragma HLS UNROLL
        ss += hd_weight[o * HEAD_IN + i][k] * b[k];             // K1 fetch
      }
      acc[o] += ss + hd_weight[o * HEAD_IN + i][GRAM_BDIM] * s; // K1 fetch
    }
  }
  gram_ln_silu_f32<HEAD_OUT>(acc, hd_gamma, hd_beta, out);
}

void kan_top(const float in[IMG_H * IMG_W], float out[HEAD_OUT]) {
#pragma HLS INTERFACE ap_memory port=in
#pragma HLS INTERFACE ap_memory port=out
#pragma HLS INTERFACE ap_ctrl_hs port=return
  static float f0[CONV1_IN_CH][IMG_H][IMG_W];
  static float f1[CONV1_OUT_CH][F1_H][F1_W];
  static float f2[CONV2_OUT_CH][F2_H][F2_W];
LOAD:
  for (int y = 0; y < IMG_H; ++y) {
    for (int x = 0; x < IMG_W; ++x) {
#pragma HLS PIPELINE II=1
      f0[0][y][x] = in[y * IMG_W + x];
    }
  }
  conv_stage<CONV1_IN_CH, CONV1_OUT_CH, IMG_H, IMG_W, F1_H, F1_W>(
      f0, f1, c1_weight, c1_gamma, c1_beta);
  conv_stage<CONV2_IN_CH, CONV2_OUT_CH, F1_H, F1_W, F2_H, F2_W>(
      f1, f2, c2_weight, c2_gamma, c2_beta);
  float hx[HEAD_IN];
POOL:
  for (int o = 0; o < HEAD_IN; ++o) {
#pragma HLS PIPELINE II=1
    float psum = 0.0f;
    for (int y = 0; y < F2_H; ++y) {
      for (int x = 0; x < F2_W; ++x) {
        psum += f2[o][y][x];
      }
    }
    hx[o] = psum / 49.0f;
  }
  head_stage(hx, out);
}
