// K3 -- KAGN-Conv FuncCode W4A4: INT4 codebooks + per-edge index streams,
// per-input-value basis/SiLU LUTs, INT32 conv accumulation, BatchNorm
// FOLDED into per-channel requant constants, shared interpolated integer
// SiLU, integer pooling, gram-FC head with integer LayerNorm.
//
// FAIRNESS: loop structure, pragmas, II and unroll factors are IDENTICAL
// to hw/hls/kagnconv_lsq_w4a4/top.cpp; ONLY the weight fetch differs
// (index-stream -> codebook lookup). Emulator kagn_fixed_point.py is the
// normative spec.

#include "hw_config.h"
#include "int_layernorm.h"
#include "params.h"

template <int CIN, int COUT, int IH, int IW, int OH, int OW>
void conv_stage(const act_t fin[CIN][IH][IW], act_t fout[COUT][OH][OW],
                const lut_t lut_b[16][4], const lut_t lut_s[16],
                const w4_t poly_cb[][4], const w4_t base_cb[],
                const kidxrom_t poly_ids[], const bidxrom_t base_ids[],
                mult_t mult_poly, mult_t mult_base, int shift,
                const lut_t gamma_q[COUT], const int beta_q[COUT],
                long long eps_int, mult_t act_mult, int act_shift) {
  acc_t v_map[COUT][OH * OW];
  acc_t acc_s[COUT], acc_b[COUT];
#pragma HLS ARRAY_PARTITION variable=acc_s complete
#pragma HLS ARRAY_PARTITION variable=acc_b complete
CONV_Y:
  for (int y = 0; y < OH; ++y) {
CONV_X:
    for (int x = 0; x < OW; ++x) {
INIT:
      for (int o = 0; o < COUT; ++o) {
#pragma HLS PIPELINE II=1
        acc_s[o] = 0;
        acc_b[o] = 0;
      }
TAP_I:
      for (int i = 0; i < CIN; ++i) {
TAP_K:
        for (int t = 0; t < 9; ++t) {
          int ky = t / 3, kx = t % 3;
          int iy = y * CONV_STRIDE + ky - CONV_PAD;
          int ix = x * CONV_STRIDE + kx - CONV_PAD;
          bool inb = (iy >= 0) && (iy < IH) && (ix >= 0) && (ix < IW);
          uidx_t u = inb ? (uidx_t)(fin[i][iy][ix] + 8) : (uidx_t)0;
          lut_t b[GRAM_BASIS_DIM];
#pragma HLS ARRAY_PARTITION variable=b complete
LUTK:
          for (int k = 0; k < GRAM_BASIS_DIM; ++k) {
#pragma HLS UNROLL
            b[k] = inb ? lut_b[u][k] : (lut_t)0;
          }
          lut_t s = inb ? lut_s[u] : (lut_t)0;
O_LOOP:
          for (int o = 0; o < COUT; ++o) {
#pragma HLS PIPELINE II=1
            int edge = ((o * CIN + i) * 3 + ky) * 3 + kx;
            kidx_t sid = poly_ids[edge];                        // K3 fetch
            bidx_t bid = base_ids[edge];                        // K3 fetch
            acc_t ss = 0;
MACK:
            for (int k = 0; k < GRAM_BASIS_DIM; ++k) {
#pragma HLS UNROLL
              ss += (acc_t)poly_cb[sid][k] * (acc_t)b[k];       // K3 fetch
            }
            acc_s[o] += ss;
            acc_b[o] += (acc_t)base_cb[bid] * (acc_t)s;         // K3 fetch
          }
        }
      }
REQ:
      for (int o = 0; o < COUT; ++o) {
#pragma HLS PIPELINE II=1
        wide_t t2 = (wide_t)acc_s[o] * (wide_t)mult_poly
                  + (wide_t)acc_b[o] * (wide_t)mult_base;
        v_map[o][y * OW + x] =
            (acc_t)((t2 + ((wide_t)1 << (shift - 1))) >> shift);
      }
    }
  }
NORM_O:
  for (int o = 0; o < COUT; ++o) {
    acc_t yb[OH * OW];
    int_instnorm<OH * OW>(v_map[o], gamma_q[o], beta_q[o], eps_int, yb);
ACT_P:
    for (int p = 0; p < OH * OW; ++p) {
#pragma HLS PIPELINE II=1
      lut_t so = silu_interp(yb[p], silu_tab);
      fout[o][p / OW][p % OW] = act_requant(so, act_mult, act_shift);
    }
  }
}

static void head_stage(const act_t hq[HEAD_IN], acc_t out[HEAD_OUT]) {
  acc_t acc_s[HEAD_OUT], acc_b[HEAD_OUT];
#pragma HLS ARRAY_PARTITION variable=acc_s complete
#pragma HLS ARRAY_PARTITION variable=acc_b complete
HINIT:
  for (int o = 0; o < HEAD_OUT; ++o) {
#pragma HLS PIPELINE II=1
    acc_s[o] = 0;
    acc_b[o] = 0;
  }
HIN:
  for (int i = 0; i < HEAD_IN; ++i) {
    uidx_t u = (uidx_t)(hq[i] + 8);
    lut_t b[GRAM_BASIS_DIM];
#pragma HLS ARRAY_PARTITION variable=b complete
HLUT:
    for (int k = 0; k < GRAM_BASIS_DIM; ++k) {
#pragma HLS UNROLL
      b[k] = hd_lut_b[u][k];
    }
    lut_t s = hd_lut_s[u];
HOUT:
    for (int o = 0; o < HEAD_OUT; ++o) {
#pragma HLS PIPELINE II=1
      kidx_t sid = hd_basis_ids[o * HEAD_IN + i];               // K3 fetch
      bidx_t bid = hd_base_ids[o * HEAD_IN + i];                // K3 fetch
      acc_t ss = 0;
HMAC:
      for (int k = 0; k < GRAM_BASIS_DIM; ++k) {
#pragma HLS UNROLL
        ss += (acc_t)hd_basis_codebook_q[sid][k] * (acc_t)b[k]; // K3 fetch
      }
      acc_s[o] += ss;
      acc_b[o] += (acc_t)hd_base_codebook_q[bid] * (acc_t)s;    // K3 fetch
    }
  }
  acc_t v[HEAD_OUT], y[HEAD_OUT];
HPRE:
  for (int o = 0; o < HEAD_OUT; ++o) {
#pragma HLS PIPELINE II=1
    wide_t t = (wide_t)acc_s[o] * (wide_t)hd_mult_basis
             + (wide_t)acc_b[o] * (wide_t)hd_mult_base;
    v[o] = (acc_t)((t + ((wide_t)1 << (hd_shift - 1))) >> hd_shift);
  }
  int_layernorm<HEAD_OUT>(v, hd_gamma_q, hd_beta_q, hd_eps_int, y);
HACT:
  for (int o = 0; o < HEAD_OUT; ++o) {
#pragma HLS PIPELINE II=1
    lut_t so = silu_interp(y[o], silu_tab);
    out[o] = ((acc_t)so) << 6;
  }
}

void kan_top(const act_t in[IMG_H * IMG_W], acc_t out[HEAD_OUT]) {
#pragma HLS INTERFACE ap_memory port=in
#pragma HLS INTERFACE ap_memory port=out
#pragma HLS INTERFACE ap_ctrl_hs port=return
  static act_t f0[CONV1_IN_CH][IMG_H][IMG_W];
  static act_t f1[CONV1_OUT_CH][F1_H][F1_W];
  static act_t f2[CONV2_OUT_CH][F2_H][F2_W];
LOAD:
  for (int y = 0; y < IMG_H; ++y) {
    for (int x = 0; x < IMG_W; ++x) {
#pragma HLS PIPELINE II=1
      f0[0][y][x] = in[y * IMG_W + x];
    }
  }
  conv_stage<CONV1_IN_CH, CONV1_OUT_CH, IMG_H, IMG_W, F1_H, F1_W>(
      f0, f1, c1_lut_b, c1_lut_s, c1_poly_codebook_q, c1_base_codebook_q,
      c1_poly_ids, c1_base_ids, c1_mult_poly, c1_mult_base, c1_shift,
      c1_gamma_q, c1_beta_q, c1_eps_int, c1_act_mult, c1_act_shift);
  conv_stage<CONV2_IN_CH, CONV2_OUT_CH, F1_H, F1_W, F2_H, F2_W>(
      f1, f2, c2_lut_b, c2_lut_s, c2_poly_codebook_q, c2_base_codebook_q,
      c2_poly_ids, c2_base_ids, c2_mult_poly, c2_mult_base, c2_shift,
      c2_gamma_q, c2_beta_q, c2_eps_int, c2_act_mult, c2_act_shift);
  act_t hq[HEAD_IN];
POOL:
  for (int o = 0; o < HEAD_IN; ++o) {
#pragma HLS PIPELINE II=1
    acc_t psum = 0;
    for (int y = 0; y < F2_H; ++y) {
      for (int x = 0; x < F2_W; ++x) {
        psum += (acc_t)f2[o][y][x];
      }
    }
    wide_t t = (wide_t)psum * (wide_t)pool_mult;
    wide_t r = (t + ((wide_t)1 << (pool_shift - 1))) >> pool_shift;
    if (r > (wide_t)HW_QP) r = (wide_t)HW_QP;
    if (r < (wide_t)HW_QN) r = (wide_t)HW_QN;
    hq[o] = (act_t)r;
  }
  head_stage(hq, out);
}
