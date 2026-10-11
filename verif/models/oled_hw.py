"""
oled_hw.py — Reference model of the OLED display path (spi_master + oled_controller).

What the Pmod OLEDrgb (SSD1331) must receive on its SPI bus, as a list of
(dc, byte) transactions: dc = 0 for commands, dc = 1 for pixel data.

Power-up order (SSD1331 datasheet Rev 1.2 section 7.9, Digilent OLEDrgb_DevInit):
logic supply (pmoden) -> reset -> init commands -> panel supply (vccen)
-> Display ON (0xAF) -> pixels. VCC must stay off until after the reset.

Specification values used by the tests (VELA-NFR-001: 100 MHz clock):
    SCLK            6.25 MHz   -> half period = 8 cycles at 100 MHz
    power-up wait   20 ms      (pmoden high -> reset)
    reset pulse     100 us     (oled_res low; datasheet minimum 3 us)
    post-reset      100 us     (oled_res high -> first command)
    VCC settle      25 ms      (vccen high -> Display ON)
    display turn-on 100 ms     (Display ON -> first pixel accepted)
"""

# REQ: VELA-FR-009

SPI_FREQ = 6_250_000
POWER_UP_S = 20e-3
RESET_S = 100e-6
VCC_ON_S = 25e-3
DISPLAY_ON_S = 100e-3

# SSD1331 command bytes sent by oled_controller: the 30 bytes of init_rom
# (rtl/oled_controller.v), then Display ON (0xAF) once VCC is on.
SSD1331_INIT = [
    0xAE,              # display off
    0xA0, 0x72,        # remap: RGB, 65k colors (RGB565)
    0xA1, 0x00,        # display start line
    0xA2, 0x00,        # display offset
    0xA4,              # normal display
    0xA8, 0x3F,        # multiplex ratio: 64
    0xAD, 0x8E,        # master configuration: external VCC
    0xB0, 0x0B,        # power save mode: disabled
    0xB1, 0x31,        # phase 1/2 period
    0xB3, 0xF0,        # clock divider / oscillator frequency
    0x8A, 0x64,        # second pre-charge speed A
    0x8B, 0x78,        # second pre-charge speed B
    0x8C, 0x64,        # second pre-charge speed C
    0x15, 0x00, 0x5F,  # column address: 0..95
    0x75, 0x00, 0x3F,  # row address: 0..63
    0xAF,              # display on
]


def sclk_half_period(clk_freq, spi_freq=SPI_FREQ):
    """Expected SCLK half period in clk cycles for the specified frequencies."""
    return round(clk_freq / spi_freq / 2)


def pixel_bytes(pixels):
    """RGB565 pixels -> data transactions: high byte then low byte, dc = 1."""
    out = []
    for p in pixels:
        out += [(1, (p >> 8) & 0xFF), (1, p & 0xFF)]
    return out


def expected_transactions(pixels):
    """Full SPI traffic: init commands (dc = 0) then the pixel data (dc = 1)."""
    return [(0, b) for b in SSD1331_INIT] + pixel_bytes(pixels)


def decode_monitor(words):
    """
    Split the 48-bit words written by verif/tb/spi_monitor.v and tb_oled_controller.v.

    Returns:
        spi:    list of (dc, byte, cycle) for type-0 words (SPI bytes)
        events: list of (signal_id, value, cycle) for type-1 words
    """
    spi, events = [], []
    for w in words:
        kind, a, b, cycle = w >> 44, (w >> 40) & 0xF, (w >> 32) & 0xFF, w & 0xFFFFFFFF
        (spi if kind == 0 else events).append((a, b, cycle))
    return spi, events
