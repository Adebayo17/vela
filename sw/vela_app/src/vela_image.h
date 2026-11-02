/*
 * vela_image.h — Portable image processing of the VELA ARM application
 *
 * No hardware access: compiled for the ARM (Vitis) and on the host, where
 * verif/tests/test_sw_host.py checks it against the Python reference
 * (python/data_loader.py, verif/models/cnn_hw_model.py).
 *
 * CIFAR-10 record (test_batch.bin): 1 label byte + 3 072 bytes, the R, G and
 * B planes of a 32x32 image (row by row). Functions take the 3 072 pixels.
 *
 * Part of VELA — Vision Edge Learning Accelerator (Phase 3, VELA-TN-007 §9)
 */

#ifndef VELA_IMAGE_H
#define VELA_IMAGE_H

#include <stdint.h>

#define VELA_CIFAR_SIZE     32                          /* CIFAR-10 image side */
#define VELA_CIFAR_PIXELS   (3 * VELA_CIFAR_SIZE * VELA_CIFAR_SIZE)   /* 3 072 */
#define VELA_RECORD_BYTES   (1 + VELA_CIFAR_PIXELS)                    /* 3 073 */
#define VELA_CROP           24                          /* CNN input side */
#define VELA_IN_VALUES      (3 * VELA_CROP * VELA_CROP)                /* 1 728 */
#define VELA_IN_WORDS       (VELA_IN_VALUES / 2)                       /*   864 */
#define VELA_CLASSES        10
#define VELA_SRC_SIDE       64                          /* image zone: CIFAR x 2 */

/*
 * CNN preprocessing, identical to python/data_loader.preprocess followed by
 * cnn_hw_model.input_raw (VELA-TN-005 §5):
 *   center crop 24x24, m' = (m - mean) / max(std, 1/sqrt(N)) over the 1 728
 *   values (population std), raw = clamp(floor(m' * 2^11 + 0.5)) in int16.
 * Output in CHW order. Computed in double precision.
 */
void vela_preprocess(const uint8_t pixels[VELA_CIFAR_PIXELS], int16_t raw[VELA_IN_VALUES]);

/* input_t raw code of one preprocessed value (round half up, saturate). */
int16_t vela_input_raw(double x);

/*
 * 32-bit words of the CNN IP image window (offset 0x1000): element 2n in
 * bits 15:0, element 2n+1 in bits 31:16 (cnn_hw_model.pack_image_words).
 */
void vela_pack_words(const int16_t raw[VELA_IN_VALUES], uint32_t words[VELA_IN_WORDS]);

/* 10 logit_t raw codes from the 5 words of the logits window (offset 0x20). */
void vela_unpack_logits(const uint32_t words[VELA_CLASSES / 2], int16_t logits[VELA_CLASSES]);

/* Softmax of the logits (logit = raw / 2^9), probabilities in [0, 1]. */
void vela_softmax(const int16_t logits[VELA_CLASSES], double probs[VELA_CLASSES]);

/*
 * Image x 2 for the source memory (VELA-TN-007 §5): each pixel duplicated
 * into a 2x2 square, 64x64 words 0x00RRGGBB at y * 64 + x.
 */
void vela_image_x2(const uint8_t pixels[VELA_CIFAR_PIXELS], volatile uint32_t *src);

#endif /* VELA_IMAGE_H */
