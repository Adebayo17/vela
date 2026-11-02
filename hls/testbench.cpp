// testbench.cpp — Vitis HLS testbench of cnn_hw() (C-simulation and C/RTL co-simulation)
//
//   testbench <data_dir> [N]
//
// Reads <data_dir>/tb_images.bin and <data_dir>/tb_expected.bin, written by
// tools/gen_hls_tb_data.py from the bit-exact model of the fixed-point CNN
// (TN004 formats), runs cnn_hw() on the first N images and requires exact
// equality of the class and of the 10 logits. Returns 0 if all match: Vitis
// HLS reports the C-simulation / co-simulation as passed only in that case.
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cnn_hw.h"

static bool read_doubles(const std::string &path, std::vector<double> &buf) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { std::perror(path.c_str()); return false; }
    const size_t got = std::fread(buf.data(), sizeof(double), buf.size(), f);
    std::fclose(f);
    if (got != buf.size()) {
        std::fprintf(stderr, "ERROR: %s: %zu values, expected %zu\n", path.c_str(), got, buf.size());
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <data_dir> [N]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int n = argc > 2 ? std::atoi(argv[2]) : 20;
    const size_t img_size = (size_t)IMG_C * IMG_H * IMG_W;
    const int rec = 2 * FC_OUT + 1;

    std::vector<double> images(img_size * n), expected((size_t)rec * n);
    if (!read_doubles(dir + "/tb_images.bin", images) ||
        !read_doubles(dir + "/tb_expected.bin", expected))
        return 1;

    static input_t image[IMG_C][IMG_H][IMG_W];
    int errors = 0;
    for (int i = 0; i < n; i++) {
        const double *src = &images[i * img_size];
        for (int c = 0; c < IMG_C; c++)
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    image[c][y][x] = src[(c * IMG_H + y) * IMG_W + x];

        logit_t logits[FC_OUT];
        const int pred = cnn_hw(image, logits);

        const double *exp = &expected[(size_t)i * rec];
        bool ok = pred == (int)exp[2 * FC_OUT];
        for (int o = 0; o < FC_OUT; o++)
            ok = ok && logits[o].to_double() == exp[o];
        if (!ok) {
            errors++;
            std::printf("MISMATCH image %d: class %d (expected %d)\n", i, pred, (int)exp[2 * FC_OUT]);
        }
    }
    std::printf("%s: %d/%d images bit-exact\n", errors ? "FAIL" : "PASS", n - errors, n);
    return errors ? 1 : 0;
}
