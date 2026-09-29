// FP32 Cox-de Boor B-spline basis + SiLU for the D1 design.
// Same recurrence, same eps, same half-open order-0 indicators as
// funcodekan/models/spline.py::SplineLayer.b_splines (grid_size=5, order=3,
// 12 knots). D2/D3 use precomputed LUTs instead (A4 insight) -- the
// per-precision-class basis algorithm difference is documented in
// docs/HW_FAIRNESS.md.
#ifndef FUNCODE_HW_BSPLINE_FP32_H
#define FUNCODE_HW_BSPLINE_FP32_H

#include <cmath>

#define BSPL_KNOTS 12
#define BSPL_ORDER 3
#define BSPL_DIM 8

static inline float silu_f32(float x) {
#pragma HLS INLINE
  return x / (1.0f + expf(-x));
}

static inline void bspline_basis_f32(float x, const float grid[BSPL_KNOTS],
                                     float basis[BSPL_DIM]) {
#pragma HLS INLINE
  float b[BSPL_KNOTS - 1];
BSPL_ORDER0:
  for (int t = 0; t < BSPL_KNOTS - 1; ++t) {
#pragma HLS UNROLL
    b[t] = (x >= grid[t] && x < grid[t + 1]) ? 1.0f : 0.0f;
  }
  const float eps = 1e-8f;
BSPL_REC:
  for (int k = 1; k <= BSPL_ORDER; ++k) {
    // ascending t reads the not-yet-updated b[t+1] -- matches the torch
    // vectorized recurrence exactly
BSPL_T:
    for (int t = 0; t + k < BSPL_KNOTS - 1; ++t) {
      float dp = grid[t + k] - grid[t];
      float dn = grid[t + k + 1] - grid[t + 1];
      b[t] = (x - grid[t]) / (dp + eps) * b[t]
           + (grid[t + k + 1] - x) / (dn + eps) * b[t + 1];
    }
  }
BSPL_COPY:
  for (int j = 0; j < BSPL_DIM; ++j) {
#pragma HLS UNROLL
    basis[j] = b[j];
  }
}

#endif // FUNCODE_HW_BSPLINE_FP32_H
