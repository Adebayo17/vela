// cnn_hw_main.cpp — Batch runner for the fixed-point CNN (cnn_hw.cpp)
//
// Driven by verif/tests/test_hls_cnn_hw.py:
//   cnn_hw_main <images.bin> <N> <out.bin> [trace.bin]
//
//   images.bin : N x 3 x 24 x 24 float64, little endian, CHW, preprocessed
//                images; converted to input_t (round to nearest, saturate)
//   out.bin    : per image 21 float64: 10 logits (logit_t), 10 exact scores
//                (acc_t, exactly representable in float64), class
//   trace.bin  : optional, pool1, pool2, pool3 of the first image (float64)
//
// Same role as the C-simulation testbench of Vitis HLS, compiled with g++
// and the Vitis ap_fixed headers.
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cnn_hw.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s images.bin N out.bin [trace.bin]\n", argv[0]);
        return 2;
    }
    const long n = std::atol(argv[2]);
    const size_t img_size = (size_t)IMG_C * IMG_H * IMG_W;

    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) { std::perror(argv[1]); return 1; }
    std::vector<double> images(img_size * n);
    if (std::fread(images.data(), sizeof(double), images.size(), fin) != images.size()) {
        std::fprintf(stderr, "ERROR: %s too short for %ld images\n", argv[1], n);
        return 1;
    }
    std::fclose(fin);

    FILE *fout = std::fopen(argv[3], "wb");
    if (!fout) { std::perror(argv[3]); return 1; }
    static input_t image[IMG_C][IMG_H][IMG_W];
    static CnnHwTrace trace;
    for (long i = 0; i < n; i++) {
        const double *src = &images[i * img_size];
        for (int c = 0; c < IMG_C; c++)
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    image[c][y][x] = src[(c * IMG_H + y) * IMG_W + x];

        cnn_hw_trace = &trace;
        logit_t logits[FC_OUT];
        const int pred = cnn_hw(image, logits);

        double record[2 * FC_OUT + 1];
        for (int o = 0; o < FC_OUT; o++) {
            record[o] = logits[o].to_double();
            record[FC_OUT + o] = trace.scores[o].to_double();
        }
        record[2 * FC_OUT] = pred;
        std::fwrite(record, sizeof(double), 2 * FC_OUT + 1, fout);

        if (i == 0 && argc > 4) {
            FILE *ftr = std::fopen(argv[4], "wb");
            if (!ftr) { std::perror(argv[4]); return 1; }
            for (int c = 0; c < C1; c++)
                for (int y = 0; y < P1H; y++)
                    for (int x = 0; x < P1W; x++) {
                        const double v = trace.pool1[c][y][x].to_double();
                        std::fwrite(&v, sizeof(double), 1, ftr);
                    }
            for (int c = 0; c < C2; c++)
                for (int y = 0; y < P2H; y++)
                    for (int x = 0; x < P2W; x++) {
                        const double v = trace.pool2[c][y][x].to_double();
                        std::fwrite(&v, sizeof(double), 1, ftr);
                    }
            for (int c = 0; c < C3; c++)
                for (int y = 0; y < P3H; y++)
                    for (int x = 0; x < P3W; x++) {
                        const double v = trace.pool3[c][y][x].to_double();
                        std::fwrite(&v, sizeof(double), 1, ftr);
                    }
            std::fclose(ftr);
        }
    }
    std::fclose(fout);
    std::printf("DONE: %ld images\n", n);
    return 0;
}
