/*
 * main.c — VELA ARM bare-metal application (Phase 3, VELA-TN-007 §9)
 *
 * At start-up: mounts the SD card, opens test_batch.bin (CIFAR-10 binary
 * test set), runs the VALIDATION MODE (first VELA_GOLDEN_N images: CNN IP
 * class vs the golden model, VELA-NFR-012), then the interactive loop:
 *
 *   SW0 = 1  CNN mode    : image x 2 shown through filter_engine (copy), class
 *                          and probability in the text zone, LEDs = class
 *   SW0 = 0  filter mode : image x 2 filtered by filter_engine, filter name in
 *                          the text zone, LEDs = filter number
 *   BTN0 / BTN1 : previous / next image      BTN2 / BTN3 : previous / next filter
 *
 * Buttons and switches are sampled every 10 ms and debounced in software: a
 * new level is accepted after two equal consecutive samples (TN007 §7, §13).
 * Display rule (TN007 §5.1): the software writes only the text zone of the
 * display memory (vela_text.c clips to it); the image zone is written by
 * filter_engine only, in both modes.
 *
 * Console: UART0, 115 200 Bd (USB-UART of the Arty Z7-20).
 *
 * Part of VELA — Vision Edge Learning Accelerator
 */

#include <stdio.h>

#include "ff.h"
#include "xil_printf.h"

#include "golden_ref.h"
#include "vela_hw.h"
#include "vela_image.h"
#include "vela_text.h"

#define DATASET_FILE     "0:/test_batch.bin"
#define DATASET_IMAGES   10000
#define SAMPLE_US        10000u           /* button sampling period: 10 ms */

static const char *const CLASS_NAME[VELA_CLASSES] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};
/* 5 characters at most: the text zone is 32 pixels wide (6-pixel cells) */
static const char *const CLASS_SHORT[VELA_CLASSES] = {
    "PLANE", "AUTO", "BIRD", "CAT", "DEER", "DOG", "FROG", "HORSE", "SHIP", "TRUCK"
};
static const char *const FILTER_NAME[VELA_MODE_COUNT] = {
    "original", "grayscale", "threshold", "gaussian", "sobel", "median"
};
static const char *const FILTER_SHORT[VELA_MODE_COUNT] = {
    "ORIG", "GRAY", "THRES", "GAUSS", "SOBEL", "MED"
};

static FATFS fs;
static FIL dataset;
static uint8_t record[VELA_RECORD_BYTES];

/* Read image `index` of test_batch.bin into `record` (label + pixels). */
static int load_record(int index)
{
    UINT got;
    if (f_lseek(&dataset, (FSIZE_t)index * VELA_RECORD_BYTES) != FR_OK)
        return -1;
    if (f_read(&dataset, record, VELA_RECORD_BYTES, &got) != FR_OK || got != VELA_RECORD_BYTES)
        return -1;
    return 0;
}

/* Classify the loaded record; probability (0..1) of the class in *prob. */
static int classify(double *prob, uint32_t *write_us, uint32_t *infer_us)
{
    int16_t raw[VELA_IN_VALUES], logits[VELA_CLASSES];
    uint32_t words[VELA_IN_WORDS];
    double probs[VELA_CLASSES];
    int cls;

    vela_preprocess(record + 1, raw);
    vela_pack_words(raw, words);
    cls = vela_cnn_run(words, logits, write_us, infer_us);
    if (cls < 0 || cls >= VELA_CLASSES)
        return -1;
    vela_softmax(logits, probs);
    *prob = probs[cls];
    return cls;
}

/* Show the loaded record in the image zone through filter_engine. */
static int show_image(enum vela_mode mode)
{
    vela_image_x2(record + 1, vela_src_mem());
    return vela_engine_run(mode);
}

static void show_text(const char *title, const char *line1, const char *line2, int index)
{
    char buf[8];
    volatile uint32_t *disp = vela_disp_mem();

    vela_text_fill(disp, VELA_BLACK);
    vela_text_draw(disp, VELA_TEXT_X0 + 1, 2, title, VELA_YELLOW, VELA_BLACK);
    vela_text_draw(disp, VELA_TEXT_X0 + 1, 18, line1, VELA_WHITE, VELA_BLACK);
    vela_text_draw(disp, VELA_TEXT_X0 + 1, 30, line2, VELA_CYAN, VELA_BLACK);
    snprintf(buf, sizeof buf, "%d", index);
    vela_text_draw(disp, VELA_TEXT_X0 + 1, 54, buf, VELA_GREY, VELA_BLACK);
}

/* ── Validation mode (VELA-NFR-012) ─────────────────────────────────────── */

static void validation(void)
{
    int i, agree = 0, hw_errors = 0;

    xil_printf("\r\nValidation: first %d test images, CNN IP vs golden model\r\n", VELA_GOLDEN_N);
    xil_printf("  idx label      golden     FPGA       prob   write_us infer_us\r\n");
    for (i = 0; i < VELA_GOLDEN_N; i++) {
        double prob = 0.0;
        uint32_t wr = 0, inf = 0;
        int cls = (load_record(i) == 0) ? classify(&prob, &wr, &inf) : -1;
        int ok = cls == VELA_GOLDEN_CLASS[i];

        agree += ok;
        if (cls != VELA_GOLDEN_EXPECTED[i])
            hw_errors++;               /* differs from the bit-exact IP model */
        xil_printf("  %3d %-10s %-10s %-10s %3d %%  %8u %8u %s\r\n", i,
                   CLASS_NAME[VELA_GOLDEN_LABEL[i]], CLASS_NAME[VELA_GOLDEN_CLASS[i]],
                   cls >= 0 ? CLASS_NAME[cls] : "ERROR", (int)(prob * 100.0 + 0.5),
                   (unsigned)wr, (unsigned)inf, ok ? "" : "<-- differs");
    }
    xil_printf("VALIDATION %d / %d agree with the golden model: VELA-NFR-012 %s\r\n",
               agree, VELA_GOLDEN_N, agree >= 9 ? "PASS" : "FAIL");
    xil_printf("           %d differ from the bit-exact IP model (expected 0)\r\n", hw_errors);
}

/* ── Interactive loop ──────────────────────────────────────────────────── */

static void refresh(int cnn_mode, int index, enum vela_mode filter)
{
    char pct[8];

    if (load_record(index) != 0) {
        show_text("SD", "ERROR", "", index);
        return;
    }
    if (cnn_mode) {
        double prob = 0.0;
        uint32_t wr = 0, inf = 0;
        int cls = classify(&prob, &wr, &inf);
        int eng = show_image(VELA_MODE_COPY);
        if (cls < 0) {
            show_text("CNN", "ERROR", "", index);
            return;
        }
        snprintf(pct, sizeof pct, "%d%%", (int)(prob * 100.0 + 0.5));
        show_text("CNN", CLASS_SHORT[cls], pct, index);
        vela_leds((uint32_t)cls);
        xil_printf("CNN    image %4d  label %-10s  class %-10s  %3d %%  infer %u us  engine %d us\r\n",
                   index, CLASS_NAME[record[0] % VELA_CLASSES], CLASS_NAME[cls],
                   (int)(prob * 100.0 + 0.5), (unsigned)inf, eng);
    } else {
        int eng = show_image(filter);
        show_text("FILT", FILTER_SHORT[filter], "", index);
        vela_leds((uint32_t)filter);
        xil_printf("FILTER image %4d  %-10s  engine %d us\r\n", index, FILTER_NAME[filter], eng);
    }
}

int main(void)
{
    uint32_t stable, last, sample;
    uint64_t next;
    int index = 0, cnn_mode;
    enum vela_mode filter = VELA_MODE_SOBEL;

    xil_printf("\r\nVELA - Vision Edge Learning Accelerator (Phase 3)\r\n");
    vela_leds(0);
    vela_text_fill(vela_disp_mem(), VELA_BLACK);

    if (f_mount(&fs, "0:/", 1) != FR_OK || f_open(&dataset, DATASET_FILE, FA_READ) != FR_OK) {
        xil_printf("ERROR: cannot open %s on the SD card\r\n", DATASET_FILE);
        show_text("NO SD", "CARD", "", 0);
        for (;;)
            ;
    }

    validation();

    stable = last = vela_inputs();
    cnn_mode = (stable & VELA_SW0) != 0;
    refresh(cnn_mode, index, filter);
    next = vela_time_us() + SAMPLE_US;

    for (;;) {
        uint32_t pressed;
        int dirty = 0;

        if (vela_time_us() < next)
            continue;
        next += SAMPLE_US;

        /* Debounce: accept a level seen on two consecutive samples */
        sample = vela_inputs();
        pressed = 0;
        if (sample == last && sample != stable) {
            pressed = sample & ~stable & 0xFu;           /* buttons: new presses */
            stable = sample;
            if (((stable & VELA_SW0) != 0) != cnn_mode) {
                cnn_mode = (stable & VELA_SW0) != 0;
                dirty = 1;
            }
        }
        last = sample;

        if (pressed & VELA_BTN(0)) { index = (index + DATASET_IMAGES - 1) % DATASET_IMAGES; dirty = 1; }
        if (pressed & VELA_BTN(1)) { index = (index + 1) % DATASET_IMAGES; dirty = 1; }
        if (pressed & VELA_BTN(2)) { filter = (enum vela_mode)((filter + VELA_MODE_COUNT - 1) % VELA_MODE_COUNT); dirty = 1; }
        if (pressed & VELA_BTN(3)) { filter = (enum vela_mode)((filter + 1) % VELA_MODE_COUNT); dirty = 1; }

        if (dirty)
            refresh(cnn_mode, index, filter);
    }
}
