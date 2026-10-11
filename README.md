# VELA — Vision Edge Learning Accelerator

> **VELA** : **V**ision **E**dge **L**earning **A**ccelerator — plateforme embarquée de traitement d'image et d'inférence CNN sur FPGA.

Image processing and CNN inference platform on a Xilinx Zynq-7020 SoC (Digilent Arty Z7-20).  
Results displayed live on a Pmod OLED RGB (SSD1331, 96x64px).

Two operating modes, selectable via on-board switches and buttons:

- **Image filters** — Grayscale, Sobel edge detection, Gaussian blur, Median filter, Threshold (binary)
- **CNN classification** — CIFAR-10 classifier (10 categories) on 24x24 images

## Architecture overview

```
                     ┌─────────────────────────────────────────────┐
                     │              Zynq XC7Z020 SoC               │
                     │                                             │
  ┌──────────┐       │  ┌──────────────────────────────────────┐   │
  │ SD Card  │───────│──│         ARM Cortex-A9 (PS)           │   │
  └──────────┘       │  │  bare-metal: SD read, dispatch,      │   │
                     │  │  mode FSM, OLED update               │   │
  ┌──────────┐       │  └──────────┬───────────────────────────┘   │
  │ Buttons  │───────│─────────────│                               │
  │ Switches │       │             │ AXI                           │
  └──────────┘       │  ┌──────────▼───────────────────────────┐   │
                     │  │            FPGA Fabric (PL)          │   │
                     │  │                                      │   │
                     │  │  ┌─────────────┐  ┌───────────────┐  │   │
                     │  │  │ Filter IPs  │  │   CNN IP      │  │   │
                     │  │  │ (Verilog)   │  │   (HLS C++)   │  │   │
                     │  │  └──────┬──────┘  └───────┬───────┘  │   │
                     │  │         └────────┬────────┘          │   │
                     │  │                  │                   │   │
                     │  │         ┌────────▼────────┐          │   │
                     │  │         │ OLED Controller │          │   │
                     │  │         │ (Verilog + SPI) │          │   │
                     │  │         └────────┬────────┘          │   │
                     │  └──────────────────│───────────────────┘   │
                     └─────────────────────│───────────────────────┘
                                           │ SPI
                                  ┌────────▼────────┐
                                  │  Pmod OLED RGB  │
                                  │  SSD1331 96x64  │
                                  └─────────────────┘
```

## CNN architecture

```
Image 24x24x3 (CHW)
  -> Conv1(3x3, 64 filters) + ReLU + MaxPool(3x3, stride 2)  -> 12x12x64
  -> Conv2(3x3, 32 filters) + ReLU + MaxPool(3x3, stride 2)  -> 6x6x32
  -> Conv3(3x3, 20 filters) + ReLU + MaxPool(3x3, stride 2)  -> 3x3x20
  -> Reshape (HWC flatten)                                     -> 180
  -> Fully Connected (180 -> 10)                               -> 10
  -> Softmax                                                   -> probabilities
```

Pre-trained weights from `CNN_coeff_3x3.txt` (TensorFlow format, transposed to CHW at load time).

## Image filters

| Filter    | Kernel | Implementation | Description                                            |
| --------- | ------ | -------------- | ------------------------------------------------------ |
| Grayscale | —      | Verilog RTL    | RGB to luminance (0.299R + 0.587G + 0.114B)            |
| Sobel     | 3x3    | Verilog RTL    | Edge detection (horizontal + vertical gradient)        |
| Gaussian  | 3x3    | Verilog RTL    | Smoothing / noise reduction ([1,2,1;2,4,2;1,2,1] / 16) |
| Median    | 3x3    | Verilog RTL    | Non-linear noise removal (sort 9 values, pick middle)  |
| Threshold | —      | Verilog RTL    | Binary output (pixel > threshold ? white : black)      |

All filters are implemented as pure Verilog RTL modules — simple enough to describe at register level and a good demonstration of RTL design skills. The CNN, being more complex, is implemented in Vitis HLS C++ with `ap_fixed` arithmetic.

## On-board controls

```
SW0 : Mode       0 = Filters    1 = CNN classification
SW1 : Source     0 = SD card    1 = ESP32-CAM (future)

BTN0 : Previous image
BTN1 : Next image
BTN2 : Previous filter (filter mode) / unused (CNN mode)
BTN3 : Next filter (filter mode)     / unused (CNN mode)
```

Every button press triggers an immediate OLED refresh — no confirm step.

## Project structure

```
vela/
├── python/                     # Phase 1 — Python golden models
│   ├── cnn.py                  # CNN layers (conv2d, relu, maxpool, fc, softmax)
│   ├── filters.py              # Image filters (grayscale, sobel, gaussian, median, threshold)
│   ├── data_loader.py          # CIFAR-10 binary reader + preprocessing
│   ├── weights_loader.py       # TF weight file parser with CHW transposition
│   ├── main.py                 # CLI entry point for CNN validation
│   ├── gui_simulator.py        # Board simulator: OLED view, HW (bit-exact) / golden views
│   └── tests/                  # Golden model tests
│       ├── conftest.py         # Fixtures (data paths, weights, synthetic images)
│       ├── test_cnn.py         # CNN unit + system tests
│       └── test_filters.py     # Filter unit tests
├── rtl/                        # Phase 2a — Synthesizable Verilog only (filters + display)
│   ├── grayscale.v             # RGB to grayscale conversion
│   ├── sobel_filter.v          # Sobel edge detection (3x3)
│   ├── gaussian_filter.v       # Gaussian blur (3x3)
│   ├── median_filter.v         # Median filter (3x3)
│   ├── threshold.v             # Binary threshold
│   ├── window3x3.v             # 3x3 sliding window (line buffers) shared by 3x3 filters
│   ├── oled_controller.v       # SSD1331 OLED display driver
│   └── spi_master.v            # SPI master for OLED communication
├── verif/                      # RTL verification (co-simulation vs bit-exact models)
│   ├── models/
│   │   ├── filters_hw.py       # Bit-exact integer models of the RTL filters
│   │   ├── oled_hw.py          # Expected SPI traffic of the OLED (init + RGB565 pixels)
│   │   └── pipeline_hw.py      # Board output model: RTL filter chain -> RGB565 (GUI HW view)
│   ├── tb/                     # File-driven Verilog testbenches (tb_<module>.v)
│   │   └── spi_monitor.v       # Passive SPI decoder/checker (mode 0, SCLK period)
│   └── tests/
│       ├── rtl_sim.py          # iverilog/vvp runner for file-driven testbenches
│       ├── test_rtl_*.py       # One co-simulation test file per RTL module
│       └── test_rtl_lint.py    # Verilator lint + Verilog-2005 check of every RTL file
├── hls/                        # Phase 2b — Vitis HLS C++ (CNN only)
│   ├── cnn_hw.cpp              # CNN accelerator (ap_fixed)
│   ├── cnn_hw.h                # Interface & type definitions
│   └── testbench.cpp           # C simulation testbench
├── data/                       # Data files (not tracked in git)
│   ├── CNN_coeff_3x3.txt       # Pre-trained weights
│   └── cifar-10-batches-bin/
│       ├── test_batch.bin      # 10000 test images
│       └── batches.meta.txt
├── tools/
│   ├── traceability_scanner.py # Requirements <-> tests matrix (make trace)
│   ├── synth.tcl               # Vivado OOC synthesis + P&R of each module (make synth)
│   ├── synth_summary.py        # Synthesis summary + NFR-002/004 check
│   └── vela_testing.py         # Shared @requirement decorator for all test suites
├── docs/
│   ├── requirement/            # Requirements (VELA-FR-xxx)
│   └── TN/
│       ├── TN001.md            # Technical note: CHW vs HWC memory layouts
│       ├── TN002.md            # Technical note: RTL verification method (co-simulation)
│       └── TN003.md            # Technical note: RTL module specification (ports, latency, handshake)
├── Makefile                    # Task runner (test, run, coverage, clean)
└── pytest.ini                  # pytest configuration
```

## Quick start

### Requirements

- Python 3.8+
- NumPy
- pytest (for tests)

### Data setup

Download [CIFAR-10 binary version](https://www.cs.toronto.edu/~kriz/cifar.html) and extract into `data/cifar-10-batches-bin/`.  
Place `CNN_coeff_3x3.txt` in `data/`.

### Run

```bash
# Single image inference
cd python
python3 main.py --single 0

# Validate on 100 images (default)
python3 main.py

# Validate on 1000 images (Sprint 1 exit criterion: >= 60%)
python3 main.py --images 1000 --confusion
```

### Board simulator (GUI)

```bash
make gui          # 100 random images
make gui-small    # 20 images, faster start
```

Simulates the Pmod OLED (96x64) and the board controls. Two views, toggled with **H**:

- **HW** (default): what the board will display. Raw 8-bit pixels go through the
  bit-exact models of the RTL chain (`verif/models/pipeline_hw.py`: grayscale →
  filter → RGB565), shown without rescaling. **S** cycles the Sobel `MAG_SHIFT`
  (0-3; RTL default **2**: ~1 % of saturated pixels on CIFAR-10 images, versus
  33-48 % with 0).
- **GOLDEN**: floating-point golden model on the CNN-normalized image, rescaled
  to [0, 255] for display.

Other keys: ←/→ image, ↑/↓ filter, M mode CNN/filters, D debug panel, Q quit.

### Tests

```bash
# Fast unit tests (no CIFAR-10 data needed, only weights file)
make test

# All tests including accuracy validation (needs CIFAR-10 data)
make test-all

# RTL co-simulation vs bit-exact model (needs Icarus Verilog) + Verilator lint
make rtl-test
make rtl-lint

# Same RTL tests + waveforms for GTKWave (build/waves/<test>/), K = pytest -k filter
make rtl-waves K=grayscale
gtkwave build/waves/test_rtl_grayscale__test_matches_model_with_valid_gaps/wave.gtkw

# Synthesis + place & route of each RTL module (Vivado, out of context, 100 MHz)
make synth                         # all modules, ~6 min; reports in build/synth/
make synth MODULES="sobel_filter"  # a subset

# Requirements traceability matrix
make trace
```

## Roadmap

### Phase 1a — Python golden model: CNN
- [x] `data_loader.py` — CIFAR-10 reader + center crop + normalization
- [x] `weights_loader.py` — TF weight parser with CHW transposition
- [x] `cnn.py` — All layers (conv2d, relu, maxpool, fc, softmax) + forward pass
- [x] `main.py` — CLI with single/batch/confusion modes
- [x] Test suite with `@requirement` traceability
- [x] Fix MaxPool SAME padding (TF `ceil(H/stride)` behavior)
- [x] Pass all unit tests on VM
- [x] Validate accuracy >= 60% on 1000 images

### Phase 1b — Python golden model: image filters
- [x] `filters.py` — Grayscale, Sobel, Gaussian, Median, Threshold
- [x] `test_filters.py` — Unit tests (synthetic images, known outputs)
- [x] Visual validation on sample CIFAR-10 images

### Phase 2a — Verilog RTL: image filters
- [x] `grayscale.v` — Fixed-point RGB to luminance
- [x] `window3x3.v` — Shared 3x3 window generator (line buffers, zero padding, `ready_in` flush)
- [x] `sobel_filter.v` — 3x3 Sobel with line buffers (magnitude >> `MAG_SHIFT` (default 2), saturated at 255)
- [x] `gaussian_filter.v` — 3x3 Gaussian with line buffers
- [x] `median_filter.v` — 3x3 median (19-CAS sorting network, 3 pipeline stages)
- [x] `threshold.v` — Comparator + binary output
- [x] Testbenches with Python golden model comparison (`make rtl-test`, waveforms: `make rtl-waves`)
- [x] Synthesis in Vivado, resource/timing check (`make synth`: all modules meet 100 MHz, filters use 1.4 % LUT, 0 DSP)
- [x] Display path verified: `spi_master` (SPI mode 0, 6.25 MHz) and `oled_controller` (SSD1331 datasheet power-up order: VCC after reset and init, 25 ms before Display ON; RGB565 pixels), parameterized for 100 MHz

### Phase 2b — HLS C++: CNN accelerator (Vitis HLS 2024.2)
- [ ] Port CNN to C++ double precision (reference)
- [ ] Analyze value ranges for `ap_fixed` sizing
- [ ] Port to `ap_fixed` for Vitis HLS synthesis
- [ ] Optimize with pipeline/unroll directives
- [ ] Export CNN IP to Vivado

### Phase 3 — FPGA integration (Vivado 2024.2)
- [ ] Block design: Zynq PS + CNN IP + filter IPs + OLED controller
- [ ] Button/switch debounce (~10ms) + mode FSM
- [ ] ARM bare-metal application (SD card read, filter/CNN dispatch, OLED update)
- [ ] Bitstream generation and on-board validation
- [ ] Target: FPGA result matches Python on >= 9/10 test images

## Design choices

| Aspect     | Choice                                  | Rationale                                                            |
| ---------- | --------------------------------------- | -------------------------------------------------------------------- |
| Filters    | Verilog RTL                             | Simple enough for register-level design; demonstrates RTL skills     |
| CNN        | Vitis HLS C++                           | Complex datapath with many MACs; HLS handles scheduling + pipelining |
| Arithmetic | `ap_fixed` (HLS), fixed-point (Verilog) | No floating-point unit on FPGA fabric                                |
| Display    | SPI to SSD1331                          | Existing proven driver from Sobel project                            |
| Controls   | Immediate refresh                       | OLED is the only feedback — user sees result instantly               |

## Hardware

| Component             | Role                                           |
| --------------------- | ---------------------------------------------- |
| Digilent Arty Z7-20   | Zynq XC7Z020 SoC (ARM Cortex-A9 + FPGA fabric) |
| Pmod OLED RGB         | SSD1331 96x64px display (SPI)                  |
| SD card               | Image storage for on-board inference           |
| Freenove ESP32-S3 CAM | Live image source (future)                     |

## Tools

- **Vivado 2024.2** — FPGA synthesis, implementation, bitstream
- **Vitis HLS 2024.2** — C++ to RTL synthesis (CNN only)
- **Python 3 + NumPy** — Golden model and validation
- **Make** — Task automation (via Git Bash on Windows)

## License

Academic project — no license specified.
