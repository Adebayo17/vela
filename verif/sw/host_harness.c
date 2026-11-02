/*
 * host_harness.c — Runs the portable ARM code (sw/vela_app/src) on the host
 *
 * Built and driven by verif/tests/test_sw_host.py:
 *   host_harness pre  < records.bin > out.bin
 *       stdin : N x 3 072 pixel bytes (CIFAR-10 planes, no label)
 *       stdout: per image, 1 728 int16 raw codes then 864 uint32 words
 *   host_harness x2   < pixels.bin > out.bin      4 096 uint32 source words
 *   host_harness soft < logits.bin > out.bin      10 int16 in, 10 double out
 *   host_harness unpack < words.bin > out.bin     5 uint32 in, 10 int16 out
 *   host_harness raw  < values.bin > out.bin      N double in, N int16 out
 *   host_harness text > out.bin
 *       96 x 64 display words pre-filled with a marker (0xA5A50000 | index),
 *       then text drawn at in-zone, partly-out and fully-out positions,
 *       including x = 0 (image zone) and negative coordinates.
 *
 * Little-endian binary I/O (host = x86, like the ARM).
 *
 * Part of VELA — Vision Edge Learning Accelerator (verification only)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vela_image.h"
#include "vela_text.h"

static int read_exact(void *buf, size_t n)
{
    return fread(buf, 1, n, stdin) == n;
}

int main(int argc, char **argv)
{
    if (argc < 2)
        return 2;

    if (!strcmp(argv[1], "pre")) {
        uint8_t pixels[VELA_CIFAR_PIXELS];
        int16_t raw[VELA_IN_VALUES];
        uint32_t words[VELA_IN_WORDS];
        while (read_exact(pixels, sizeof pixels)) {
            vela_preprocess(pixels, raw);
            vela_pack_words(raw, words);
            fwrite(raw, sizeof raw, 1, stdout);
            fwrite(words, sizeof words, 1, stdout);
        }
    } else if (!strcmp(argv[1], "x2")) {
        uint8_t pixels[VELA_CIFAR_PIXELS];
        static uint32_t src[VELA_SRC_SIDE * VELA_SRC_SIDE];
        if (!read_exact(pixels, sizeof pixels))
            return 1;
        vela_image_x2(pixels, src);
        fwrite(src, sizeof src, 1, stdout);
    } else if (!strcmp(argv[1], "soft")) {
        int16_t logits[VELA_CLASSES];
        double probs[VELA_CLASSES];
        if (!read_exact(logits, sizeof logits))
            return 1;
        vela_softmax(logits, probs);
        fwrite(probs, sizeof probs, 1, stdout);
    } else if (!strcmp(argv[1], "unpack")) {
        uint32_t words[VELA_CLASSES / 2];
        int16_t logits[VELA_CLASSES];
        if (!read_exact(words, sizeof words))
            return 1;
        vela_unpack_logits(words, logits);
        fwrite(logits, sizeof logits, 1, stdout);
    } else if (!strcmp(argv[1], "raw")) {
        double x;
        while (read_exact(&x, sizeof x)) {
            int16_t r = vela_input_raw(x);
            fwrite(&r, sizeof r, 1, stdout);
        }
    } else if (!strcmp(argv[1], "text")) {
        static uint32_t disp[VELA_DISP_W * VELA_DISP_H];
        int i;
        for (i = 0; i < VELA_DISP_W * VELA_DISP_H; i++)
            disp[i] = 0xA5A50000u | (uint32_t)i;
        vela_text_fill(disp, VELA_BLACK);
        vela_text_draw(disp, VELA_TEXT_X0, 4, "CNN", VELA_YELLOW, VELA_BLACK);
        vela_text_draw(disp, VELA_TEXT_X0, 20, "TRUCK", VELA_WHITE, VELA_BLACK);
        vela_text_draw(disp, VELA_TEXT_X0, 32, "73%", VELA_CYAN, VELA_BLACK);
        vela_text_draw(disp, VELA_TEXT_X0, 54, "9999", VELA_GREY, VELA_BLACK);
        vela_text_draw(disp, 0, 10, "IMAGE ZONE", VELA_WHITE, VELA_BLACK);      /* starts in image zone */
        vela_text_draw(disp, 40, 44, "ABCDEFGHIJ", VELA_WHITE, VELA_BLACK);     /* crosses x = 64 */
        vela_text_draw(disp, 90, 60, "XY", VELA_WHITE, VELA_BLACK);             /* off right / bottom */
        vela_text_draw(disp, -20, -4, "NEGATIVE", VELA_WHITE, VELA_BLACK);      /* off left / top */
        vela_text_draw(disp, VELA_TEXT_X0, 100, "LOW", VELA_WHITE, VELA_BLACK); /* fully below */
        fwrite(disp, sizeof disp, 1, stdout);
    } else {
        return 2;
    }
    return 0;
}
