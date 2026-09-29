// FP32 GRAM (KAGN) polynomial basis + SiLU + float LayerNorm for the G1
// design. Exact replica of funcodekan/models/variants.py
// GramPolynomialLayer: z = tanh(x); P0=1, P1=z, Pd = z*P(d-1) - 0.1*P(d-2);
// output = SiLU(LayerNorm(base + nonlinear)). Degree 3 -> 4 basis terms.
#ifndef FUNCODE_HW_GRAM_BASIS_FP32_H
#define FUNCODE_HW_GRAM_BASIS_FP32_H

#include <cmath>

#define GRAM_DEGREE 3
#define GRAM_BDIM 4
#define GRAM_LN_EPS 1e-5f

static inline float gram_silu_f32(float x) {
#pragma HLS INLINE
  return x / (1.0f + expf(-x));
}

// tanh via the expf identity: the tanhf FPO core deadlocked in Verilog
// cosim (RTL sim time overran the csynth latency bound ~330x with no
// vector progress; csim/csynth clean), while expf is proven across the
// spline fp32 design's 9h cosim. Numerics differ from libm tanhf by ~1
// ulp — far inside the 1e-3 golden tolerance.
static inline float gram_tanh_f32(float x) {
#pragma HLS INLINE
  return 1.0f - 2.0f / (expf(2.0f * x) + 1.0f);
}

static inline void gram_basis_f32(float x, float basis[GRAM_BDIM]) {
#pragma HLS INLINE
  float z = gram_tanh_f32(x);
  basis[0] = 1.0f;
  basis[1] = z;
GRAM_REC:
  for (int d = 2; d <= GRAM_DEGREE; ++d) {
    basis[d] = z * basis[d - 1] - 0.1f * basis[d - 2];
  }
}

// Float LayerNorm + SiLU over N values (sequential reductions, float32).
template <int N>
void gram_ln_silu_f32(const float pre[N], const float gamma[N],
                      const float beta[N], float out[N]) {
  float s = 0.0f;
LNF_SUM:
  for (int o = 0; o < N; ++o) {
#pragma HLS PIPELINE II=1
    s += pre[o];
  }
  float mu = s / (float)N;
  float q = 0.0f;
LNF_VAR:
  for (int o = 0; o < N; ++o) {
#pragma HLS PIPELINE II=1
    float d = pre[o] - mu;
    q += d * d;
  }
  float inv = 1.0f / sqrtf(q / (float)N + GRAM_LN_EPS);
LNF_OUT:
  for (int o = 0; o < N; ++o) {
#pragma HLS PIPELINE II=1
    float y = (pre[o] - mu) * inv * gamma[o] + beta[o];
    out[o] = gram_silu_f32(y);
  }
}

// Float INSTANCE norm + SiLU over N spatial positions with scalar
// gamma/beta (per-channel), sequential float32 reductions.
template <int N>
void gram_in_silu_f32(const float pre[N], float gamma, float beta,
                      float out[N]) {
  float s = 0.0f;
INF_SUM:
  for (int p = 0; p < N; ++p) {
#pragma HLS PIPELINE II=1
    s += pre[p];
  }
  float mu = s / (float)N;
  float q = 0.0f;
INF_VAR:
  for (int p = 0; p < N; ++p) {
#pragma HLS PIPELINE II=1
    float d = pre[p] - mu;
    q += d * d;
  }
  float inv = 1.0f / sqrtf(q / (float)N + GRAM_LN_EPS);
INF_OUT:
  for (int p = 0; p < N; ++p) {
#pragma HLS PIPELINE II=1
    float y = (pre[p] - mu) * inv * gamma + beta;
    out[p] = gram_silu_f32(y);
  }
}

#endif // FUNCODE_HW_GRAM_BASIS_FP32_H
