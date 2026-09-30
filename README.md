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
│   └── tests/                  # pytest test suite
│       ├── conftest.py         # Fixtures and @requirement decorator
│       ├── test_cnn.py         # CNN unit + system tests
│       └── test_filters.py     # Filter unit tests
├── rtl/                        # Phase 2a — Verilog RTL (image filters + display)
│   ├── grayscale.v             # RGB to grayscale conversion
│   ├── sobel_filter.v          # Sobel edge detection (3x3)
│   ├── gaussian_filter.v       # Gaussian blur (3x3)
│   ├── median_filter.v         # Median filter (3x3)
│   ├── threshold.v             # Binary threshold
│   ├── oled_controller.v       # SSD1331 OLED display driver
│   └── spi_master.v            # SPI master for OLED communication
├── hls/                        # Phase 2b — Vitis HLS C++ (CNN only)
│   ├── cnn_hw.cpp              # CNN accelerator (ap_fixed)
│   ├── cnn_hw.h                # Interface & type definitions
│   └── testbench.cpp           # C simulation testbench
├── data/                       # Data files (not tracked in git)
│   ├── CNN_coeff_3x3.txt       # Pre-trained weights
│   └── cifar-10-batches-bin/
│       ├── test_batch.bin      # 10000 test images
│       └── batches.meta.txt
├── docs/
│   └── TN/TN001.md            # Technical note: CHW vs HWC memory layouts
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

### Tests

```bash
# Fast unit tests (no CIFAR-10 data needed, only weights file)
make test

# All tests including accuracy validation (needs CIFAR-10 data)
make test-all
```

## Roadmap

### Phase 1a — Python golden model: CNN
- [x] `data_loader.py` — CIFAR-10 reader + center crop + normalization
- [x] `weights_loader.py` — TF weight parser with CHW transposition
- [x] `cnn.py` — All layers (conv2d, relu, maxpool, fc, softmax) + forward pass
- [x] `main.py` — CLI with single/batch/confusion modes
- [x] Test suite with `@requirement` traceability
- [ ] Fix MaxPool SAME padding (TF `ceil(H/stride)` behavior)
- [ ] Pass all unit tests on VM
- [ ] Validate accuracy >= 60% on 1000 images

### Phase 1b — Python golden model: image filters
- [ ] `filters.py` — Grayscale, Sobel, Gaussian, Median, Threshold
- [ ] `test_filters.py` — Unit tests (synthetic images, known outputs)
- [ ] Visual validation on sample CIFAR-10 images

### Phase 2a — Verilog RTL: image filters
- [ ] `grayscale.v` — Fixed-point RGB to luminance
- [ ] `sobel_filter.v` — 3x3 Sobel with line buffers
- [ ] `gaussian_filter.v` — 3x3 Gaussian with line buffers
- [ ] `median_filter.v` — 3x3 median (sorting network)
- [ ] `threshold.v` — Comparator + binary output
- [ ] Testbenches with Python golden model comparison
- [ ] Synthesis in Vivado, resource/timing check

### Phase 2b — HLS C++: CNN accelerator (Vitis 2024.1)
- [ ] Port CNN to C++ double precision (reference)
- [ ] Analyze value ranges for `ap_fixed` sizing
- [ ] Port to `ap_fixed` for Vitis HLS synthesis
- [ ] Optimize with pipeline/unroll directives
- [ ] Export CNN IP to Vivado

### Phase 3 — FPGA integration (Vivado 2024.1)
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

- **Vivado 2024.1** — FPGA synthesis, implementation, bitstream
- **Vitis HLS 2024.1** — C++ to RTL synthesis (CNN only)
- **Python 3 + NumPy** — Golden model and validation
- **Make** — Task automation (via Git Bash on Windows)

## License

Academic project — no license specified.
