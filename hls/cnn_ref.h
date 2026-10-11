// cnn_ref.h — CNN reference in C++ double precision (Phase 2b, step 1)
//
// Same computation as the golden model python/cnn.py, written as plain loops:
//   - conv 3x3 "same" (zero padding), CHW layout, weights [C_out][C_in][3][3]
//   - ReLU
//   - maxpool 3x3, stride 2, TensorFlow SAME padding: out = ceil(in / 2),
//     the missing row/column is added at the bottom/right and ignored (-inf)
//   - flatten in HWC order: vec[(h * W + w) * C + c] = x[c][h][w]
//   - fully connected 180 -> 10, softmax
//
// It is the C++ reference that the fixed-point HLS version (cnn_hw.cpp) will
// be compared to. Verified against the golden model by
// verif/tests/test_hls_cnn_ref.py (layer by layer and on CIFAR-10 images).
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2b: CNN accelerator (Vitis HLS)

#ifndef CNN_REF_H
#define CNN_REF_H

#include "cnn_params.h"

// Intermediate results of one inference (for layer-by-layer comparison)
struct CnnTrace {
    double conv1[C1][H1][W1];      // before ReLU
    double pool1[C1][P1H][P1W];
    double conv2[C2][H2][W2];      // before ReLU
    double pool2[C2][P2H][P2W];
    double conv3[C3][H3][W3];      // before ReLU
    double pool3[C3][P3H][P3W];
    double vec[FC_IN];             // flattened, HWC order
};

// Full forward pass. Returns the predicted class (argmax of the logits).
// `trace` may be null.
int cnn_forward_ref(const double image[IMG_C][IMG_H][IMG_W],
                    double logits[FC_OUT], double probs[FC_OUT],
                    CnnTrace *trace);

#endif // CNN_REF_H
