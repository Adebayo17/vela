/*
 * vela_hw.c — Hardware access of the VELA ARM application, see vela_hw.h
 *
 * Part of VELA — Vision Edge Learning Accelerator (Phase 3, VELA-TN-007 §9)
 */

#include "vela_hw.h"

#include "xil_io.h"
#include "xiltimer.h"          /* XTime_GetTime, COUNTS_PER_SECOND (Vitis 2024.2 SDT BSP) */

#define CNN_TIMEOUT_US     100000u     /* 100 ms (expected ~6 ms, NFR-008) */
#define ENGINE_TIMEOUT_US  10000u      /* 10 ms (expected ~41 us) */

#define CTRL_START         0x1u
#define STATUS_BUSY        0x1u
#define STATUS_DONE        0x2u
#define CNN_AP_DONE        0x2u

uint64_t vela_time_us(void)
{
    XTime t;
    XTime_GetTime(&t);
    /* COUNTS_PER_SECOND = CPU clock / 2 (global timer); the BSP macro has no parentheses */
    return (uint64_t)t / ((COUNTS_PER_SECOND) / 1000000u);
}

int vela_cnn_run(const uint32_t words[VELA_IN_WORDS], int16_t logits[VELA_CLASSES],
                 uint32_t *write_us, uint32_t *infer_us)
{
    uint32_t logit_words[VELA_CLASSES / 2];
    uint64_t t0, t1, t2;
    int n;

    t0 = vela_time_us();
    for (n = 0; n < VELA_IN_WORDS; n++)
        Xil_Out32(VELA_CNN_BASE + CNN_IMAGE + 4u * (uint32_t)n, words[n]);
    t1 = vela_time_us();

    Xil_Out32(VELA_CNN_BASE + CNN_CTRL, CTRL_START);
    /* ap_done is clear on read: stop at the first read that returns it set */
    while (!(Xil_In32(VELA_CNN_BASE + CNN_CTRL) & CNN_AP_DONE))
        if (vela_time_us() - t1 > CNN_TIMEOUT_US)
            return -1;
    t2 = vela_time_us();

    for (n = 0; n < VELA_CLASSES / 2; n++)
        logit_words[n] = Xil_In32(VELA_CNN_BASE + CNN_LOGITS + 4u * (uint32_t)n);
    vela_unpack_logits(logit_words, logits);

    *write_us = (uint32_t)(t1 - t0);
    *infer_us = (uint32_t)(t2 - t1);
    return (int)Xil_In32(VELA_CNN_BASE + CNN_RETURN);
}

int vela_engine_run(enum vela_mode mode)
{
    uint32_t ctrl = ((uint32_t)mode & 0x7u) << 1;
    uint64_t t0 = vela_time_us();
    uint32_t status;

    /* Rising edge of start with the mode already set (TN007 §7) */
    Xil_Out32(VELA_GPIO_CTRL_BASE + GPIO_DATA, ctrl);
    Xil_Out32(VELA_GPIO_CTRL_BASE + GPIO_DATA, ctrl | CTRL_START);
    Xil_Out32(VELA_GPIO_CTRL_BASE + GPIO_DATA, ctrl);

    /* done is cleared by the start edge, a few cycles before this first read */
    do {
        status = Xil_In32(VELA_GPIO_CTRL_BASE + GPIO_DATA2);
        if (vela_time_us() - t0 > ENGINE_TIMEOUT_US)
            return -1;
    } while ((status & STATUS_BUSY) || !(status & STATUS_DONE));

    return (int)(vela_time_us() - t0);
}

uint32_t vela_inputs(void)
{
    return Xil_In32(VELA_GPIO_IO_BASE + GPIO_DATA) & 0x3Fu;
}

void vela_leds(uint32_t value)
{
    Xil_Out32(VELA_GPIO_IO_BASE + GPIO_DATA2, value & 0xFu);
}
