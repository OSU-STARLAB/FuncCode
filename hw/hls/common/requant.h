// Shared requantization datapath (D2 and D3 use these unchanged; fairness
// contract, docs/HW_FAIRNESS.md). Bit-exact twin of
// funcodekan/hw/fixed_point.py::QuantLayer.forward_codes.
#ifndef FUNCODE_HW_REQUANT_H
#define FUNCODE_HW_REQUANT_H

#include "hw_config.h"

// t = acc_s*M_s + acc_b*M_b ; round-half-up shift: (t + 2^(shift-1)) >> shift
static inline wide_t requant_round(acc_t acc_s, acc_t acc_b,
                                   mult_t m_s, mult_t m_b, int shift) {
#pragma HLS INLINE
  wide_t t = (wide_t)acc_s * (wide_t)m_s + (wide_t)acc_b * (wide_t)m_b;
  return (t + ((wide_t)1 << (shift - 1))) >> shift;
}

// hidden layers: clamp to INT4
static inline act_t requant_code(acc_t acc_s, acc_t acc_b,
                                 mult_t m_s, mult_t m_b, int shift) {
#pragma HLS INLINE
  wide_t r = requant_round(acc_s, acc_b, m_s, m_b, shift);
  if (r > (wide_t)HW_QP) r = (wide_t)HW_QP;
  if (r < (wide_t)HW_QN) r = (wide_t)HW_QN;
  return (act_t)r;
}

// output layer: INT32 logits at scale 2^-HW_LOGIT_FRAC_BITS, no clamp
static inline acc_t requant_logit(acc_t acc_s, acc_t acc_b,
                                  mult_t m_s, mult_t m_b, int shift) {
#pragma HLS INLINE
  return (acc_t)requant_round(acc_s, acc_b, m_s, m_b, shift);
}

#endif // FUNCODE_HW_REQUANT_H
