// Shared C testbench for all three designs (fairness: ONE testbench).
// Reads the generated golden.h of the design under test, runs TB_N vectors
// through kan_top, and compares:
//   - D2/D3 (default): INT32 logits must be BIT-EXACT vs the emulator
//     golden outputs (verification rung L2 via csim, L3 via cosim).
//   - D1 (-DDESIGN_FP32): float logits within 1e-3, argmax exact.
// TB_N defaults to GOLDEN_N (1000); cosim runs pass a smaller -DTB_N.

#include <cmath>
#include <cstdio>

#include "hw_config.h"
#include "golden.h"

#ifndef TB_N
#define TB_N GOLDEN_N
#endif

#ifdef DESIGN_FP32
void kan_top(const float in[GOLDEN_IN], float out[GOLDEN_OUT]);
#else
void kan_top(const act_t in[GOLDEN_IN], acc_t out[GOLDEN_OUT]);
#endif

int main() {
  int logit_mismatches = 0;
  int argmax_agree = 0;
  int correct = 0;

  for (int n = 0; n < TB_N; ++n) {
#ifdef DESIGN_FP32
    float in[GOLDEN_IN];
    float out[GOLDEN_OUT];
    for (int i = 0; i < GOLDEN_IN; ++i) in[i] = golden_inputs[n][i];
    kan_top(in, out);
    int mm = 0;
    for (int c = 0; c < GOLDEN_OUT; ++c) {
      if (fabsf(out[c] - golden_outputs[n][c]) > 1e-3f) ++mm;
    }
    int am = 0, gm = 0;
    for (int c = 1; c < GOLDEN_OUT; ++c) {
      if (out[c] > out[am]) am = c;
      if (golden_outputs[n][c] > golden_outputs[n][gm]) gm = c;
    }
#else
    act_t in[GOLDEN_IN];
    acc_t out[GOLDEN_OUT];
    for (int i = 0; i < GOLDEN_IN; ++i) in[i] = (act_t)golden_inputs[n][i];
    kan_top(in, out);
    int mm = 0;
    for (int c = 0; c < GOLDEN_OUT; ++c) {
      if (out[c] != golden_outputs[n][c]) ++mm;
    }
    int am = 0, gm = 0;
    for (int c = 1; c < GOLDEN_OUT; ++c) {
      if (out[c] > out[am]) am = c;
      if (golden_outputs[n][c] > golden_outputs[n][gm]) gm = c;
    }
#endif
    if (mm) {
      ++logit_mismatches;
      if (logit_mismatches <= 10) {
        printf("MISMATCH vector %d: %d logit(s) differ\n", n, mm);
      }
    }
    if (am == gm) ++argmax_agree;
    if (am == (int)golden_labels[n]) ++correct;
  }

  printf("TB_RESULT n=%d vectors_with_logit_mismatch=%d argmax_agree=%d "
         "accuracy_pct=%.2f\n",
         TB_N, logit_mismatches, argmax_agree,
         100.0 * correct / (double)TB_N);
  if (logit_mismatches == 0 && argmax_agree == TB_N) {
    printf("TB_PASS\n");
    return 0;
  }
  printf("TB_FAIL\n");
  return 1;
}
