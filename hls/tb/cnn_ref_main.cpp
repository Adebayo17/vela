// cnn_ref_main.cpp — Batch runner for the C++ double reference
//
// Driven by verif/tests/test_hls_cnn_ref.py:
//   cnn_ref_main <images.bin> <N> <out.bin> [trace.bin]
//
//   images.bin : N x 3 x 24 x 24 float64, little endian, CHW, preprocessed
//                images (python/data_loader.preprocess)
//   out.bin    : per image 21 float64: 10 logits, 10 probabilities, class
//   trace.bin  : optional, CnnTrace of the first image (all float64, in the
//                order of the struct fields)
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cnn_ref.h"

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
    static CnnTrace trace;
    for (long i = 0; i < n; i++) {
        double logits[FC_OUT], probs[FC_OUT], record[2 * FC_OUT + 1];
        const auto *img = reinterpret_cast<const double (*)[IMG_H][IMG_W]>(&images[i * img_size]);
        const int pred = cnn_forward_ref(img, logits, probs, i == 0 ? &trace : nullptr);
        for (int o = 0; o < FC_OUT; o++) {
            record[o] = logits[o];
            record[FC_OUT + o] = probs[o];
        }
        record[2 * FC_OUT] = pred;
        std::fwrite(record, sizeof(double), 2 * FC_OUT + 1, fout);
    }
    std::fclose(fout);

    if (argc > 4 && n > 0) {
        FILE *ftr = std::fopen(argv[4], "wb");
        if (!ftr) { std::perror(argv[4]); return 1; }
        std::fwrite(&trace, sizeof(CnnTrace), 1, ftr);
        std::fclose(ftr);
    }
    std::printf("DONE: %ld images\n", n);
    return 0;
}
