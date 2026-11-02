/*
 * vela_hw.h — Hardware access of the VELA ARM application (Zynq, bare-metal)
 *
 * Addresses fixed by tools/fpga.tcl (VELA-TN-007 §4, checked at each build).
 * The PL region 0x4000_0000 is mapped as device memory by the standalone BSP:
 * accesses are not cached, no flush is needed.
 *
 * Part of VELA — Vision Edge Learning Accelerator (Phase 3, VELA-TN-007 §9)
 */

#ifndef VELA_HW_H
#define VELA_HW_H

#include <stdint.h>

#include "vela_image.h"

/* ── Address map (TN007 §4) ───────────────────────────────────────────── */
#define VELA_CNN_BASE        0x40000000u   /* cnn_hw s_axi_control */
#define VELA_GPIO_IO_BASE    0x41200000u   /* ch1: BTN[3:0] SW[5:4], ch2: LED[3:0] */
#define VELA_GPIO_CTRL_BASE  0x41210000u   /* ch1: vela_pl ctrl, ch2: vela_pl status */
#define VELA_SRC_BASE        0x42000000u   /* source BRAM, 64 x 64 words 0x00RRGGBB */
#define VELA_DISP_BASE       0x42020000u   /* display BRAM, 96 x 64 words RGB565 */

/* ── CNN IP registers (TN005 §4) ──────────────────────────────────────── */
#define CNN_CTRL             0x0000u       /* bit 0 ap_start, bit 1 ap_done (clear on read) */
#define CNN_RETURN           0x0010u       /* predicted class */
#define CNN_LOGITS           0x0020u       /* 10 x logit_t, 2 per word */
#define CNN_IMAGE            0x1000u       /* 1 728 x input_t, 2 per word, CHW */

/* ── AXI GPIO registers ──────────────────────────────────────────────── */
#define GPIO_DATA            0x0000u
#define GPIO_DATA2           0x0008u

/* Buttons and switches (axi_gpio_io channel 1, TN007 §7) */
#define VELA_BTN(n)          (1u << (n))   /* BTN0..BTN3 */
#define VELA_SW0             (1u << 4)
#define VELA_SW1             (1u << 5)

/* Filter engine modes (TN007 §6.2) */
enum vela_mode {
    VELA_MODE_COPY = 0, VELA_MODE_GRAY, VELA_MODE_THRESHOLD,
    VELA_MODE_GAUSSIAN, VELA_MODE_SOBEL, VELA_MODE_MEDIAN, VELA_MODE_COUNT
};

static inline volatile uint32_t *vela_src_mem(void)  { return (volatile uint32_t *)VELA_SRC_BASE; }
static inline volatile uint32_t *vela_disp_mem(void) { return (volatile uint32_t *)VELA_DISP_BASE; }

/* Microseconds since power-up (Cortex-A9 global timer). */
uint64_t vela_time_us(void);

/*
 * One inference (TN005 §5): writes the 864 image words, starts the IP, polls
 * ap_done, reads the class and the logits. Durations in microseconds: image
 * writes and inference. Returns the class, or -1 after a 100 ms timeout.
 */
int vela_cnn_run(const uint32_t words[VELA_IN_WORDS], int16_t logits[VELA_CLASSES],
                 uint32_t *write_us, uint32_t *infer_us);

/*
 * One filter_engine run in the given mode (start edge on ctrl bit 0, mode on
 * bits 3:1), waits for done. Returns the duration in microseconds, or -1
 * after a 10 ms timeout.
 */
int vela_engine_run(enum vela_mode mode);

/* Raw buttons / switches (6 bits) and LEDs (4 bits). */
uint32_t vela_inputs(void);
void vela_leds(uint32_t value);

#endif /* VELA_HW_H */
