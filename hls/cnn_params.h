// cnn_params.h — CNN dimensions (shared by the reference and the HLS versions)
//
// Image 3x24x24 (CHW)
//   -> Conv1 3x3, 64 + ReLU -> MaxPool 3x3/2 SAME -> 64x12x12
//   -> Conv2 3x3, 32 + ReLU -> MaxPool 3x3/2 SAME -> 32x6x6
//   -> Conv3 3x3, 20 + ReLU -> MaxPool 3x3/2 SAME -> 20x3x3
//   -> flatten in HWC order -> 180 -> FC 180x10 -> softmax
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2b: CNN accelerator (Vitis HLS)

#ifndef CNN_PARAMS_H
#define CNN_PARAMS_H

const int IMG_C = 3;
const int IMG_H = 24;
const int IMG_W = 24;

const int C1 = 64, H1 = 24, W1 = 24;   // conv1 output
const int P1H = 12, P1W = 12;          // pool1 output

const int C2 = 32, H2 = 12, W2 = 12;   // conv2 output
const int P2H = 6, P2W = 6;            // pool2 output

const int C3 = 20, H3 = 6, W3 = 6;     // conv3 output
const int P3H = 3, P3W = 3;            // pool3 output

const int FC_IN  = P3H * P3W * C3;     // 180, HWC order
const int FC_OUT = 10;                 // CIFAR-10 classes

const int K = 3;                       // conv kernel size
const int POOL = 3, POOL_STRIDE = 2;   // maxpool window and stride

#endif // CNN_PARAMS_H
