// Deterministic integer LayerNorm + interpolated integer SiLU for the GRAM
// designs. Bit-exact twins of funcodekan/hw/gram_spec.py (torch) and
// funcodekan/hw/gram_fixed_point.py (numpy) -- the frozen algorithms of
// docs/HW_DESIGN_CONTRACT.md "GRAM integer datapath". Do not modify one
// implementation without the others.
#ifndef FUNCODE_HW_INT_LAYERNORM_H
#define FUNCODE_HW_INT_LAYERNORM_H

#include "hw_config.h"

#define GRAM_NORM_FRAC 12   // Q4.12 normalized values / pre-LN grid
#define GRAM_GAMMA_FRAC 12  // gamma int16 Q4.12 (trained gammas reach ~2)
#define GRAM_SILU_FRAC 10   // SiLU table entries Q6.10

// floor division (Python // semantics) for b > 0; C '/' truncates to zero.
static inline wide_t floor_div64(wide_t a, wide_t b) {
#pragma HLS INLINE
  wide_t q = a / b;
  if ((a % b != 0) && (a < 0)) q -= 1;
  return q;
}

// floor(sqrt(x)) for x in [0, 2^62): bit-by-bit restoring, fixed 32 rounds.
static inline ap_uint<64> isqrt64(ap_uint<64> x) {
#pragma HLS INLINE off
  ap_uint<64> num = x;
  ap_uint<64> res = 0;
  ap_uint<64> bit = (ap_uint<64>)1 << 62;
ISQRT:
  for (int i = 0; i < 32; ++i) {
    ap_uint<64> t = res + bit;
    if (num >= t) {
      num -= t;
      res = (res >> 1) + bit;
    } else {
      res >>= 1;
    }
    bit >>= 2;
  }
  return res;
}

// Integer LayerNorm over N values at Q.12 (v), affine gamma (Q2.14) /
// beta (Q.26), eps_int = round(N^2 * 2^24 * 1e-5). Output y at Q.12.
template <int N>
void int_layernorm(const acc_t v[N], const lut_t gamma_q[N],
                   const int beta_q[N], long long eps_int, acc_t y[N]) {
  wide_t s = 0;
LN_SUM:
  for (int o = 0; o < N; ++o) {
#pragma HLS PIPELINE II=1
    s += (wide_t)v[o];
  }
  wide_t c[N];
  wide_t qsum = 0;
LN_CENTER:
  for (int o = 0; o < N; ++o) {
#pragma HLS PIPELINE II=1
    c[o] = (wide_t)N * (wide_t)v[o] - s;
    qsum += c[o] * c[o];
  }
  ap_uint<64> r_in = (ap_uint<64>)(qsum / N) + (ap_uint<64>)eps_int;
  wide_t r = (wide_t)isqrt64(r_in);
LN_NORM:
  for (int o = 0; o < N; ++o) {
#pragma HLS PIPELINE II=1
    wide_t num = 2 * c[o] * ((wide_t)1 << GRAM_NORM_FRAC) + r;
    wide_t n_fx = floor_div64(num, 2 * r);
    wide_t t = n_fx * (wide_t)gamma_q[o] + (wide_t)beta_q[o];
    y[o] = (acc_t)floor_div64(t + ((wide_t)1 << (GRAM_GAMMA_FRAC - 1)),
                              (wide_t)1 << GRAM_GAMMA_FRAC);
  }
}

// Integer INSTANCE norm: same frozen kernel as int_layernorm but with a
// SCALAR gamma/beta (per-channel affine over N spatial positions).
template <int N>
void int_instnorm(const acc_t v[N], lut_t gamma_q, int beta_q,
                  long long eps_int, acc_t y[N]) {
  wide_t s = 0;
IN_SUM:
  for (int p = 0; p < N; ++p) {
#pragma HLS PIPELINE II=1
    s += (wide_t)v[p];
  }
  wide_t c[N];
  wide_t qsum = 0;
IN_CENTER:
  for (int p = 0; p < N; ++p) {
#pragma HLS PIPELINE II=1
    c[p] = (wide_t)N * (wide_t)v[p] - s;
    qsum += c[p] * c[p];
  }
  ap_uint<64> r_in = (ap_uint<64>)(qsum / N) + (ap_uint<64>)eps_int;
  wide_t r = (wide_t)isqrt64(r_in);
IN_NORM:
  for (int p = 0; p < N; ++p) {
#pragma HLS PIPELINE II=1
    wide_t num = 2 * c[p] * ((wide_t)1 << GRAM_NORM_FRAC) + r;
    wide_t n_fx = floor_div64(num, 2 * r);
    wide_t t = n_fx * (wide_t)gamma_q + (wide_t)beta_q;
    y[p] = (acc_t)floor_div64(t + ((wide_t)1 << (GRAM_GAMMA_FRAC - 1)),
                              (wide_t)1 << GRAM_GAMMA_FRAC);
  }
}

// Interpolated integer SiLU: y Q4.12 -> Q6.10. tab has 257 entries over
// [-8, 8) step 1/16 (generated into each gram params.h as silu_tab).
static inline lut_t silu_interp(acc_t y, const lut_t tab[257]) {
#pragma HLS INLINE
  const int lo = -(8 << GRAM_NORM_FRAC);
  int yc = (int)y;
  if (yc < lo) yc = lo;
  if (yc > -lo - 1) yc = -lo - 1;
  int u = yc - lo;
  int i = u >> 8;
  int frac = u & 255;
  int d = (int)tab[i + 1] - (int)tab[i];
  return (lut_t)((int)tab[i] + ((d * frac + 128) >> 8));
}

// Hidden-layer activation requant: Q6.10 SiLU value -> INT4 code.
static inline act_t act_requant(lut_t s, mult_t m, int shift) {
#pragma HLS INLINE
  wide_t t = (wide_t)s * (wide_t)m;
  wide_t r = (t + ((wide_t)1 << (shift - 1))) >> shift;
  if (r > (wide_t)HW_QP) r = (wide_t)HW_QP;
  if (r < (wide_t)HW_QN) r = (wide_t)HW_QN;
  return (act_t)r;
}

#endif // FUNCODE_HW_INT_LAYERNORM_H
