// cnn_hw.h — CNN accelerator in fixed point (Phase 2b, step 3)
//
// Same network as cnn_ref.h (C++ double reference), with the ap_fixed types
// chosen in VELA-TN-004 §5: 16-bit weights and activations, round to nearest
// (AP_RND) and saturation (AP_SAT), exact 40-bit accumulators.
//
// Bit-exact reference model: tools/cnn_fixed_point_analysis.py (emulation
// with the TN004 formats). Verified by verif/tests/test_hls_cnn_hw.py.
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2b: CNN accelerator (Vitis HLS)

#ifndef CNN_HW_H
#define CNN_HW_H

#include <ap_fixed.h>

#include "cnn_params.h"

// ap_fixed<W, I>: W bits, I integer bits (sign included). See TN004 §5.
typedef ap_fixed<16, 5, AP_RND, AP_SAT> input_t;    // normalized image, +-16
typedef ap_fixed<16, 3, AP_RND, AP_SAT> weight_t;   // weights and biases, +-4
typedef ap_fixed<16, 6, AP_RND, AP_SAT> act1_t;     // conv1 / pool1, +-32
typedef ap_fixed<16, 7, AP_RND, AP_SAT> act2_t;     // conv2 / pool2, +-64
typedef ap_fixed<16, 7, AP_RND, AP_SAT> act3_t;     // conv3 / pool3, +-64
typedef ap_fixed<40, 14>                acc_t;      // accumulators, exact
typedef ap_fixed<16, 7, AP_RND, AP_SAT> logit_t;    // FC outputs, +-64

// Top function (future HLS IP).
//   image : preprocessed image (center crop + per-image normalization, done
//           by the PS), CHW
//   logits: 10 class scores (for the probability computed by the PS)
//   return: predicted class, argmax of the exact (acc_t) scores
int cnn_hw(const input_t image[IMG_C][IMG_H][IMG_W], logit_t logits[FC_OUT]);

#ifndef __SYNTHESIS__
// Simulation only: intermediate layers of the last call, for the tests.
struct CnnHwTrace {
    act1_t pool1[C1][P1H][P1W];
    act2_t pool2[C2][P2H][P2W];
    act3_t pool3[C3][P3H][P3W];
    acc_t  scores[FC_OUT];          // exact FC outputs, before logit_t
};
extern CnnHwTrace *cnn_hw_trace;   // set by a testbench, null by default
#endif

#endif // CNN_HW_H
