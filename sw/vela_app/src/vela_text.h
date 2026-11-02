/*
 * vela_text.h — Text zone of the display memory (portable)
 *
 * The display memory holds 96 x 64 RGB565 pixels, one per 32-bit word at
 * y * 96 + x (VELA-TN-007 §5). The software owns only the TEXT ZONE,
 * columns 64..95 (§5.1, VELA-FR-013): every function here clips to it, so
 * the image zone written by filter_engine is never touched, whatever the
 * coordinates or the string length.
 *
 * Font: 5 x 7 pixels, 1 pixel spacing (6 x 8 cell): 5 characters per line in
 * the 32-pixel zone. Digits, A-Z (lower case drawn as upper case), space and
 * . - : / % ; other characters are drawn as spaces.
 *
 * Part of VELA — Vision Edge Learning Accelerator (Phase 3, VELA-TN-007 §9)
 */

#ifndef VELA_TEXT_H
#define VELA_TEXT_H

#include <stdint.h>

#define VELA_DISP_W        96
#define VELA_DISP_H        64
#define VELA_TEXT_X0       64          /* first column of the text zone */
#define VELA_CHAR_W        6           /* glyph 5 + spacing 1 */
#define VELA_CHAR_H        8           /* glyph 7 + spacing 1 */

/* RGB565 colours */
#define VELA_BLACK   0x0000u
#define VELA_WHITE   0xFFFFu
#define VELA_YELLOW  0xFFE0u
#define VELA_CYAN    0x07FFu
#define VELA_GREY    0x8410u

/* Fill the whole text zone (columns 64..95, rows 0..63) with one colour. */
void vela_text_fill(volatile uint32_t *disp, uint16_t color);

/* Set one pixel if (x, y) is inside the text zone; ignored otherwise. */
void vela_text_pixel(volatile uint32_t *disp, int x, int y, uint16_t color);

/*
 * Draw a string with its top-left corner at (x, y), 6 x 8 cells with the
 * background colour, clipped to the text zone. Returns the x after the last
 * character.
 */
int vela_text_draw(volatile uint32_t *disp, int x, int y, const char *s,
                   uint16_t fg, uint16_t bg);

#endif /* VELA_TEXT_H */
