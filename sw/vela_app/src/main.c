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

#include "diskio.h"
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

/*
 * SD card diagnosis after FR_NO_FILESYSTEM: raw sector 0 (MBR) and the first
 * sector of partition 1. The Zynq BootROM and FatFs need an MBR partition
 * table with a FAT16 / FAT32 partition: type 0xEE = GPT (protective MBR),
 * 0x07 = exFAT or NTFS, 0x0B / 0x0C = FAT32, 0x06 / 0x0E = FAT16.
 */
static void sd_diagnose(void)
{
    static BYTE sector[512] __attribute__((aligned(32)));
    char fs[9] = {0};
    DWORD lba;
    int p;

    if (disk_initialize(0) & STA_NOINIT || disk_read(0, sector, 0, 1) != RES_OK) {
        xil_printf("  sector 0: read failed\r\n");
        return;
    }
    xil_printf("  sector 0: signature %02x%02x (55aa expected)\r\n", sector[510], sector[511]);
    for (p = 0; p < 4; p++) {
        const BYTE *e = sector + 446 + 16 * p;
        lba = e[8] | (e[9] << 8) | (e[10] << 16) | ((DWORD)e[11] << 24);
        xil_printf("  partition %d: type 0x%02x, first sector %u\r\n", p + 1, e[4], (unsigned)lba);
    }
    lba = sector[454] | (sector[455] << 8) | (sector[456] << 16) | ((DWORD)sector[457] << 24);
    if (lba && disk_read(0, sector, lba, 1) == RES_OK) {
        for (p = 0; p < 8; p++)                       /* OEM name: "EXFAT   ", "NTFS    "... */
            fs[p] = (sector[3 + p] >= 32 && sector[3 + p] < 127) ? (char)sector[3 + p] : '.';
        xil_printf("  partition 1 boot sector: OEM name \"%s\"", fs);
        for (p = 0; p < 8; p++)                       /* FAT32 type string at offset 82 */
            fs[p] = (sector[82 + p] >= 32 && sector[82 + p] < 127) ? (char)sector[82 + p] : '.';
        xil_printf(", FAT32 field \"%s\"\r\n", fs);
        /* BPB fields checked by FatFs before accepting the volume */
        xil_printf("  BPB: jump %02x, signature %02x%02x, bytes/sector %u, sectors/cluster %u,\r\n",
                   sector[0], sector[510], sector[511],
                   (unsigned)(sector[11] | (sector[12] << 8)), (unsigned)sector[13]);
        xil_printf("       reserved %u, FATs %u, root entries %u, total16 %u, FAT size16 %u,\r\n",
                   (unsigned)(sector[14] | (sector[15] << 8)), (unsigned)sector[16],
                   (unsigned)(sector[17] | (sector[18] << 8)),
                   (unsigned)(sector[19] | (sector[20] << 8)),
                   (unsigned)(sector[22] | (sector[23] << 8)));
        xil_printf("       total32 %u, FAT size32 %u, version %u, root cluster %u\r\n",
                   (unsigned)(sector[32] | (sector[33] << 8) | (sector[34] << 16) | ((DWORD)sector[35] << 24)),
                   (unsigned)(sector[36] | (sector[37] << 8) | (sector[38] << 16) | ((DWORD)sector[39] << 24)),
                   (unsigned)(sector[42] | (sector[43] << 8)),
                   (unsigned)(sector[44] | (sector[45] << 8) | (sector[46] << 16) | ((DWORD)sector[47] << 24)));
        /* The FAT type is set by the cluster count only (Microsoft FAT spec):
         * < 65 525 clusters is FAT16, whatever the "FAT32" label says; a
         * "FAT32" volume with too few clusters (e.g. 1 GB with 32 KB clusters)
         * is rejected by FatFs and by the Zynq BootROM */
        {
            DWORD rsv = sector[14] | (sector[15] << 8);
            DWORD fatsz = sector[36] | (sector[37] << 8) | (sector[38] << 16) | ((DWORD)sector[39] << 24);
            DWORD total = sector[32] | (sector[33] << 8) | (sector[34] << 16) | ((DWORD)sector[35] << 24);
            DWORD spc = sector[13] ? sector[13] : 1;
            DWORD clusters = (total - rsv - sector[16] * fatsz) / spc;
            xil_printf("  clusters %u: %s\r\n", (unsigned)clusters,
                       clusters < 65525u
                       ? "too few for FAT32 (< 65525) -> reformat with smaller clusters (4 KB) or as FAT16"
                       : "valid FAT32 cluster count");
        }
    }
}

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

    {
        FRESULT mount = f_mount(&fs, "0:/", 1);
        FRESULT open = (mount == FR_OK) ? f_open(&dataset, DATASET_FILE, FA_READ) : FR_OK;
        if (mount != FR_OK || open != FR_OK) {
            /* FatFs codes: 3 FR_NOT_READY (card not detected / not answering),
             * 13 FR_NO_FILESYSTEM (no FAT16/FAT32 volume: exFAT, no partition
             * table...), 4 FR_NO_FILE (file not at the root of the card) */
            xil_printf("ERROR: SD card: f_mount = %d, f_open(%s) = %d\r\n",
                       (int)mount, DATASET_FILE, (int)open);
            if (mount == FR_NO_FILESYSTEM)
                sd_diagnose();
            show_text("NO SD", mount != FR_OK ? "MOUNT" : "FILE",
                      "", mount != FR_OK ? (int)mount : (int)open);
            for (;;)
                ;
        }
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
