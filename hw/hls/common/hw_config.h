// Frozen numeric contract of the FuncCode-KAN HW designs.
// Mirrors funcodekan/hw/hw_spec.py -- NEVER change one side without the
// other (docs/HW_DESIGN_CONTRACT.md "Frozen design decisions").
#ifndef FUNCODE_HW_CONFIG_H
#define FUNCODE_HW_CONFIG_H

#include <ap_int.h>

typedef ap_int<4>   w4_t;    // INT4 weights / codebook entries (-8..7)
typedef ap_int<4>   act_t;   // INT4 activation codes (-8..7)
typedef ap_uint<4>  uidx_t;  // LUT index u = q + 8 (0..15)
typedef ap_uint<5>  sidx_t;  // spline codebook index (Ks = 32)
typedef ap_uint<4>  bidx_t;  // base codebook index (Kb = 16)
typedef ap_int<16>  lut_t;   // LUT entries: basis Q2.14, SiLU Q5.11
typedef ap_int<32>  acc_t;   // branch accumulators / INT32 logits
typedef ap_int<32>  mult_t;  // requant multipliers
typedef ap_int<64>  wide_t;  // requant wide accumulator

// plain types used by the generated golden.h (testbench side)
typedef signed char in_code_t;

// Storage types of the LARGE generated ROMs (weights / index streams).
// The csim compiler (mingw gcc 6.2) ICEs on >~100k class-type initializers,
// so csim uses POD storage; synthesis uses the true narrow ap types. The
// VALUES are identical (INT4 / 5-bit / 4-bit ranges), so csim stays
// bit-exact with the RTL (verified again by cosim, rung L3).
#ifdef __SYNTHESIS__
typedef ap_int<4>  w4rom_t;    // INT4 weight ROMs
typedef ap_uint<5> sidxrom_t;  // spline index stream ROM (Ks=32)
typedef ap_uint<4> bidxrom_t;  // base index stream ROM (Kb=16)
typedef ap_uint<6> kidxrom_t;  // KAGN-conv poly index ROM (Ks=64)
#else
typedef signed char   w4rom_t;
typedef unsigned char sidxrom_t;
typedef unsigned char bidxrom_t;
typedef unsigned char kidxrom_t;
#endif
typedef ap_uint<6> kidx_t;     // KAGN-conv poly index compute type

enum {
  HW_QN = -8,
  HW_QP = 7,
  HW_BASIS_FRAC_BITS = 14,
  HW_SILU_FRAC_BITS = 10,
  HW_LOGIT_FRAC_BITS = 16
};

// D2 weight ROMs are nibble-packed POD uint32 words (exactly 4 bits per
// weight; see docs/HW_FAIRNESS.md): decode two's-complement INT4 nibble.
static inline int w4_unpack(unsigned int word, int nib) {
#pragma HLS INLINE
  int v = (int)((word >> (nib * 4)) & 0xFu);
  return (v ^ 8) - 8;
}

#endif // FUNCODE_HW_CONFIG_H
