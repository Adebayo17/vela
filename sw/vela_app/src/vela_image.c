/*
 * vela_image.c — Portable image processing of the VELA ARM application
 *
 * See vela_image.h. Reference: python/data_loader.py (crop, normalization),
 * verif/models/cnn_hw_model.py (input_raw, pack_image_words,
 * unpack_logit_words).
 *
 * Part of VELA — Vision Edge Learning Accelerator (Phase 3, VELA-TN-007 §9)
 */

#include "vela_image.h"

#include <math.h>

#define CROP_OFFSET  ((VELA_CIFAR_SIZE - VELA_CROP) / 2)   /* 4, as crop_center */
#define INPUT_FRAC   11                                    /* input_t = ap_fixed<16, 5> */
#define LOGIT_FRAC   9                                     /* logit_t = ap_fixed<16, 7> */

static uint8_t crop_pixel(const uint8_t *pixels, int c, int y, int x)
{
    return pixels[(c * VELA_CIFAR_SIZE + y + CROP_OFFSET) * VELA_CIFAR_SIZE + x + CROP_OFFSET];
}

int16_t vela_input_raw(double x)
{
    double scaled = floor(x * (double)(1 << INPUT_FRAC) + 0.5);
    if (scaled > 32767.0)
        return 32767;
    if (scaled < -32768.0)
        return -32768;
    return (int16_t)scaled;
}

void vela_preprocess(const uint8_t pixels[VELA_CIFAR_PIXELS], int16_t raw[VELA_IN_VALUES])
{
    double sum = 0.0, sq = 0.0, mean, std, min_std;
    int c, y, x, i;

    for (c = 0; c < 3; c++)
        for (y = 0; y < VELA_CROP; y++)
            for (x = 0; x < VELA_CROP; x++)
                sum += crop_pixel(pixels, c, y, x);
    mean = sum / VELA_IN_VALUES;

    for (c = 0; c < 3; c++)
        for (y = 0; y < VELA_CROP; y++)
            for (x = 0; x < VELA_CROP; x++) {
                double d = crop_pixel(pixels, c, y, x) - mean;
                sq += d * d;
            }
    std = sqrt(sq / VELA_IN_VALUES);                       /* population std */
    min_std = 1.0 / sqrt((double)VELA_IN_VALUES);
    if (std < min_std)
        std = min_std;

    i = 0;
    for (c = 0; c < 3; c++)
        for (y = 0; y < VELA_CROP; y++)
            for (x = 0; x < VELA_CROP; x++)
                raw[i++] = vela_input_raw((crop_pixel(pixels, c, y, x) - mean) / std);
}

void vela_pack_words(const int16_t raw[VELA_IN_VALUES], uint32_t words[VELA_IN_WORDS])
{
    int n;
    for (n = 0; n < VELA_IN_WORDS; n++)
        words[n] = (uint32_t)(uint16_t)raw[2 * n] | ((uint32_t)(uint16_t)raw[2 * n + 1] << 16);
}

void vela_unpack_logits(const uint32_t words[VELA_CLASSES / 2], int16_t logits[VELA_CLASSES])
{
    int n;
    for (n = 0; n < VELA_CLASSES / 2; n++) {
        logits[2 * n]     = (int16_t)(uint16_t)(words[n] & 0xFFFFu);
        logits[2 * n + 1] = (int16_t)(uint16_t)(words[n] >> 16);
    }
}

void vela_softmax(const int16_t logits[VELA_CLASSES], double probs[VELA_CLASSES])
{
    double z[VELA_CLASSES], zmax, total = 0.0;
    int k;

    for (k = 0; k < VELA_CLASSES; k++)
        z[k] = logits[k] / (double)(1 << LOGIT_FRAC);
    zmax = z[0];
    for (k = 1; k < VELA_CLASSES; k++)
        if (z[k] > zmax)
            zmax = z[k];
    for (k = 0; k < VELA_CLASSES; k++) {
        probs[k] = exp(z[k] - zmax);                       /* shifted: no overflow */
        total += probs[k];
    }
    for (k = 0; k < VELA_CLASSES; k++)
        probs[k] /= total;
}

void vela_image_x2(const uint8_t pixels[VELA_CIFAR_PIXELS], volatile uint32_t *src)
{
    const int plane = VELA_CIFAR_SIZE * VELA_CIFAR_SIZE;
    int x, y;

    for (y = 0; y < VELA_SRC_SIDE; y++)
        for (x = 0; x < VELA_SRC_SIDE; x++) {
            int p = (y / 2) * VELA_CIFAR_SIZE + x / 2;
            src[y * VELA_SRC_SIDE + x] = ((uint32_t)pixels[p] << 16)
                                       | ((uint32_t)pixels[plane + p] << 8)
                                       | (uint32_t)pixels[2 * plane + p];
        }
}
