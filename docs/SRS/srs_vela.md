# VELA — Vision Edge Learning Accelerator
## Software & System Requirements Specification (SRS)

> **Document ID:** VELA-SRS-001 

> **Version:** 1.0 

> **Status:** Draft 

> **Date:** 2026-05-10 

> **Author:** Gaël BOYA

---

## Revision history

| Version   | Date          | Author    | Changes           |
| :---      | :---          | :---      | :---              |
| 0.1       | 2025-01-01    | Gaël BOYA | Initial draft     |
| 1.0       | 2025-01-01    | Gaël BOYA | First baseline    |
 
---

## 1. Introduction

### 1.1 Purpose

This document defines the system and software requirements for **VELA** (Vision Edge Learning Accelerator), a reconfigurable hardware-software platform that performs real-time image classificatiion using Convolutional Neural Network (CNN) entirely on an embedded FPGA SoC, without any cloud dependency.

### 1.2 Problem statement

Modern AI-based image classification systems rely on cloud infrastructure for inference. This creates four classes of engineering problems that **VELA** aims to solve: 

| Problem class         | Description                                                                                                   | Affected sectors                                  |
| :---                  | :---                                                                                                          | :---                                              |
| **Latency**           | Round-trip to cloud introduces 50–500 ms of variable delay                                                    | Robotics, autonomous vehicles, industrial lines   |
| **Connectivity**      | Remote or RF-denied environments have no reliable network                                                     | Drones, agriculture, defence                      |
| **Data sovereignty**  | Images of patients, production lines, or restricted areas cannot leave the premises                           | Medical, defence, industry                        |
| **Power budget**      | GPU-class inference hardware (70–300 W) is incompatible with battery-powered or space-constrained deployments | IoT, wearables, drones                            |

**VELA** addresses all four simultaneously by implementing inference entirely on a Xilinx Zynq-7020 SoC, combining an ARM Cortex-A9 processing system with FPGA programmable logic on a single chip.

### 1.3 Scope

**VELA** covers three layers of a complete edge inference system: 

- **Algorithm layer** — a validated Python reference model (golden model)
- **Accelerator layer** — a fixed-point C++ implementation synthesised to RTL via High-Level-Synthesis (HLS)
- **Integration layer** — a complete SoC integrating the CNN accelerator, image input, and display output on the Arty Z7-20 development board

The platform uses the CIFAR-10 image classification task as its primary benchmark. The architecture is intentionally generic: the CNN topology, input resolution, and number of output classes are parameterisable to accomodate domain-specific retraining.

### 1.4 Definitions and acronyms

| Term          | Definition                                                                        |
| :---          | :---                                                                              |
| CNN           | Convolutional Neural Network                                                      |
| FPGA          | Field-Programmable Gate Array                                                     |
| HLS           | High-Level Synthesis                                                              |
| PS            | Processing System (ARM side of Zynq)                                              |
| PL            | Programmable Logic (FPGA side of Zynq)                                            |
| SoC           | System on Chip                                                                    |
| AXI           | Advanced eXtensible Interface (ARM bus protocol)                                  |
| BRAM          | Block RAM (on-chip memory in Xilinx FPGAs)                                        |
| IVV           | Integration, Verification and Validation                                          |
| Golden model  | Software reference implementation used as ground truth for hardware verification  |
| Inference     | Forward pass through a trained neural network (no training involved)              |
| Edge          | Computation performed locally on the device, not in the cloud                     |
 
### 1.5 Stakeholders
 
| Stakeholder                           | Role                                          | Primary concern                       |
| :---                                  | :---                                          | :---                                  |
| System integrator                     | Adopts VELA as a platform building block      | Portability, documentation quality    |
| Domain client                         | Provides domain-specific trained weights      | Accuracy, adaptability                |
| Hardware engineer                     | Implements the PL accelerator                 | Timing closure, resource budget       |
| Software engineer                     | Develops the ARM bare-metal application       | API clarity, driver reliability       |
| Certification authority (optional)    | Reviews the system for safety-critical use    | Traceability, test coverage           |

--- 

## 2. Overall system description

### 2.1 System context

**VELA** is a **platform**, not a product. It provides the inference execution engine. The domain-specific elements (trained weights, image source, application logic) are provided by the integrator or client.

```
┌──────────────────────────────────────────────────────┐
│                   Client domain                      │
│  ┌─────────────┐   ┌──────────────┐                  │
│  │ Image source│   │ Trained CNN  │                  │
│  │(SD / camera)│   │   weights    │                  │
│  └──────┬──────┘   └──────┬───────┘                  │
│         │                 │                          │
│  ┌──────▼─────────────────▼────────────────────────┐ │
│  │              VELA Platform                      │ │
│  │  ┌──────────┐  ┌─────────────┐  ┌───────────┐   │ │
│  │  │ ARM PS   │  │ CNN accel.  │  │  Display  │   │ │
│  │  │ (driver) │  │  (PL/HLS)   │  │(OLED/HDMI)│   │ │
│  │  └──────────┘  └─────────────┘  └───────────┘   │ │
│  └─────────────────────────────────────────────────┘ │
│         │                                            │
│  ┌──────▼──────┐                                     │
│  │  Application│  classification result + confidence │
│  └─────────────┘                                     │
└──────────────────────────────────────────────────────┘
```

### 2.2 Target hardware

| Component     | Specification                                                 |
| :---          | :---                                                          |
| SoC           | Xilinx Zynq XC7Z020-1CLG400C                                  |
| Board         | Digilent Arty Z7-20                                           |
| DDR3          | 512 MiB, 16-bit bus @ 1050 Mbps                               |
| Flash         | 16 MB Quad-SPI                                                |
| Storage       | microSD (FAT32)                                               |
| Display       | Digilent Pmod OLEDrgb (SSD1331, 96×64 px, SPI)                |
| FPGA fabric   | 53,200 LUT, 106,400 FF, 630 KB BRAM, 220 DSP                  |
| Clock         | 50 MHz PS input → up to 650 MHz CPU, configurable PL clock    |
 
### 2.3 Development tools
 
| Tool              | Version   | Purpose                                   |
| :---              | :---      | :---                                      |
| Python            | ≥ 3.10    | Golden model, IVV scripts                 |
| NumPy             | ≥ 1.24    | Array computation                         |
| pytest            | ≥ 7.0     | Unit and integration testing              |
| Vitis HLS         | 2024.1    | C++ → RTL synthesis                       |
| Vivado            | 2024.1    | Block design, implementation, bitstream   |
| Vitis (embedded)  | 2024.1    | ARM bare-metal application                |
| Git               | ≥ 2.40    | Version control                           |

### 2.4 Assumptions and constraints

- Training of the CNN is **out-the-scope**. Pre-trained weights are provided as input files (`CNN_coeff_3x3.txt`).
- The platform targets inference only (forward pass). Non on-device learning is performed.
- The development board (Arty Z7-20) is the sole target for this version. Porting to other Zynq devices is architecturally possible but not validated here.
- The Vivado WebPACk free licence is sufficient for the XC7Z020 target.
- Power supply is USB (500 mA) or external 7-15 V.

---

## 3. Functional requirements

Requirements are identified as **VELA-FR-XXX** (Functional Requirement).
 
### 3.1 Golden model (Python)
 
| ID            | Requirement                                                                                                                                                   | Priority  |
| :---          | :---                                                                                                                                                          | :---      |
| VELA-FR-001   | The system shall provide a Python implementation of the complete CNN forward pass, including convolution, ReLU, MaxPool, fully-connected, and softmax layers. | Must      |
| VELA-FR-002   | The Python model shall load pre-trained weights from a text file (`CNN_coeff_3x3.txt`) and apply them without modification.                                   | Must      |
| VELA-FR-003   | The Python model shall preprocess input images: centre crop to 24×24 pixels, and apply per-image normalisation (µ/σ globally across all channels).            | Must      |
| VELA-FR-004   | The Python model shall output the predicted class index (0–9) and the full softmax probability vector for each input image.                                   | Must      |
| VELA-FR-005   | The Python model shall achieve a top-1 classification accuracy of at least 60% on the CIFAR-10 test set (10,000 images).                                      | Must      |
| VELA-FR-006   | The Python model shall support batch evaluation with configurable image count and produce a confusion matrix.                                                 | Should    |
 
### 3.2 HLS accelerator (C++ / Vitis HLS)
 
| ID            | Requirement                                                                                                                                                           | Priority  |
| :---          | :---                                                                                                                                                                  | :---      |
| VELA-FR-010   | The system shall provide a C++ implementation of the CNN forward pass using `ap_fixed` fixed-point arithmetic synthesisable by Vitis HLS.                             | Must      |
| VELA-FR-011   | The C++ implementation shall be separated into a top-level function (`cnn_top`) and a testbench (`testbench.cpp`). The top-level function shall contain no file I/O.  | Must      |
| VELA-FR-012   | The C++ implementation shall achieve a top-1 accuracy of at least 58% on the CIFAR-10 test set (acceptable degradation due to quantisation).                          | Must      |
| VELA-FR-013   | The HLS synthesis shall complete without timing violations at a target clock of 100 MHz.                                                                              | Must      |
| VELA-FR-014   | The accelerator shall use no more than 80% of available DSP blocks on the XC7Z020.                                                                                    | Must      |
| VELA-FR-015   | The accelerator shall use no more than 70% of available BRAM on the XC7Z020.                                                                                          | Must      |
| VELA-FR-016   | The HLS implementation shall apply at least one pipeline directive (`#pragma HLS PIPELINE`) to the innermost convolution loop.                                        | Should    |
| VELA-FR-017   | The synthesised IP shall be exportable as a Vivado-compatible IP core.                                                                                                | Must      |
 
### 3.3 SoC integration (Vivado + ARM)
 
| ID            | Requirement                                                                                                                                           | Priority  |
| :---          | :---                                                                                                                                                  | :---      |
| VELA-FR-020   | The system shall implement a complete Zynq SoC integrating the CNN accelerator IP, ARM PS, AXI interconnect, and BRAM for image and weight storage.   | Must      |
| VELA-FR-021   | The ARM processor shall read an input image from a microSD card (FAT32 filesystem) and write it to BRAM via AXI.                                      | Must      |
| VELA-FR-022   | The ARM processor shall trigger the CNN accelerator and read back the classification result via AXI-Lite registers.                                   | Must      |
| VELA-FR-023   | The system shall display the input image and the top-1 predicted class label on the Pmod OLEDrgb (96×64 pixels) via SPI.                              | Must      |
| VELA-FR-024   | The user shall be able to navigate to the next image by pressing BTN1 on the Arty Z7-20.                                                              | Should    |
| VELA-FR-025   | LED0 shall be illuminated during CNN computation and extinguished when the result is ready.                                                           | Should    |
| VELA-FR-026   | The system shall transmit the classification result (class index, class name, confidence in %) over UART at 115200 baud for debugging.                | Must      |
 
### 3.4 Display and output
 
| ID            | Requirement                                                                                                                               | Priority  |
| :---          | :---                                                                                                                                      | :---      |
| VELA-FR-030   | The OLED display shall show the input image (upscaled ×2, 48×48 px) in the left half of the screen.                                       | Should    |
| VELA-FR-031   | The OLED display shall show the predicted class name and confidence percentage in the right half of the screen using a 5×7 bitmap font.   | Must      |
| VELA-FR-032   | The display shall update within 1 second of BTN1 being pressed.                                                                           | Should    |

--- 

## 4. Non-functional requirements

Requirements are identified as **VELA-NFR-XXX**.
 
| ID            | Requirement                                                                                                                                                   | Category        | Priority |
| :---          | :---                                                                                                                                                          | :---            | :---     |
| VELA-NFR-001  | End-to-end inference latency (image read → result on display) shall not exceed 2 seconds.                                                                     | Performance     | Must     |
| VELA-NFR-002  | The system shall operate from USB power (≤ 500 mA at 5V = 2.5 W).                                                                                             | Power           | Should   |
| VELA-NFR-003  | All source code shall be version-controlled in a Git repository with meaningful commit messages.                                                              | Maintainability | Must     |
| VELA-NFR-004  | The Python golden model shall achieve ≥ 90% unit test coverage on individual CNN layers.                                                                      | Reliability     | Must     |
| VELA-NFR-005  | The hardware–software interface (AXI register map) shall be documented in a dedicated section of the architecture document.                                   | Traceability    | Must     |
| VELA-NFR-006  | The Vivado project shall be reconstructible from a TCL script without manual intervention.                                                                    | Reproducibility | Should   |
| VELA-NFR-007  | The platform shall be adaptable to different CNN topologies by replacing weight files and modifying HLS parameters, without redesigning the SoC block diagram.| Portability     | Should   |
| VELA-NFR-008  | No proprietary or licensed third-party IP shall be used. All IP blocks shall be either Xilinx standard library IP or original RTL.                            | Licensing       | Must     |

---

## 5. Implementation constraints

### 5.1 CNN architecture (fixed for this version)

The CNN topology is fixed by the provided weight files:

```
Input        : 3 × 24 × 24 (RGB, centre-cropped, normalised)
Conv1        : kernel 3×3, 64 filters, padding same, stride 1
ReLU1
MaxPool1     : window 3×3, stride 2  → 64 × 12 × 12
Conv2        : kernel 3×3, 32 filters, padding same, stride 1
ReLU2
MaxPool2     : window 3×3, stride 2  → 32 × 6 × 6
Conv3        : kernel 3×3, 20 filters, padding same, stride 1
ReLU3
MaxPool3     : window 3×3, stride 2  → 20 × 3 × 3
Reshape      : → 180
FC           : 180 → 10
Softmax      : → 10 class probabilities
```

Total parameters: see `weights_loader.py::ARCH` for exacts counts per layer.

### 5.2 Fixed-point format (initial)

| Layer output  | Proposed format   | Rationale                         |
| :---          | :---              | :---                              |
| Input image   | `ap_fixed<8,1>`   | Normalised ∈ [-4, 4] after µ/σ    |
| Conv outputs  | `ap_fixed<16,8>`  | Accumulation of products          |
| FC output     | `ap_fixed<16,8>`  | Dot product of 180 elements       |
| Softmax       | `ap_fixed<16,1>`  | Output ∈ [0, 1]                   |
 
These formats shall be validated against the fixed-point range analysis document (`fixed_point_analysis.md`) before synthesis.
 
---
 
## 6. External interfaces
 
### 6.1 Software interfaces
 
| Interface             | Direction | Format                    | Description                           |
| :---                  | :---      | :---                      | :---                                  |
| `CNN_coeff_3x3.txt`   | Input     | ASCII float, one per line | Pre-trained weights and biases        |
| `test_batch.bin`      | Input     | CIFAR-10 binary format    | Input images and labels               |
| UART                  | Output    | ASCII, 115200 8N1         | Debug log: class, confidence          |
| `ivv_refs/*.bin`      | I/O       | Raw float32 binary        | IVV reference files (Python → C++)    |
 
### 6.2 Hardware interfaces
 
| Interface     | Connector     | Protocol                      | Description                       |
| :---          | :---          | :---                          | :---                              |
| Pmod OLEDrgb  | JA (pins 1–8) | SPI mode 3, up to 6.67 MHz    | Display output                    |
| microSD       | J9            | SDIO 4-bit                    | Image and weight storage          |
| USB-UART      | J14           | CDC/ACM, 115200 baud          | Debug output                      |
| JTAG          | J14 / J15     | Cascaded JTAG                 | Programming and debug             |
| BTN0–BTN3     | On-board      | GPIO PL                       | Reset (BTN0), next image (BTN1)   |
| LED0–LED3     | On-board      | GPIO PL                       | Status indicators                 |
 
---
 
## 7. Traceability matrix (requirements → tests)
 
| Requirement   | Test ID               | Test type                 | Sprint    |
| :---          | :---                  | :---                      | :---      |
| VELA-FR-001   | VELA-T-001 to T-004   | Unit (pytest)             | Sprint 1  |
| VELA-FR-002   | VELA-T-005            | Unit (pytest)             | Sprint 1  |
| VELA-FR-003   | VELA-T-006            | Unit (pytest)             | Sprint 1  |
| VELA-FR-005   | VELA-T-010            | System (pytest)           | Sprint 1  |
| VELA-FR-010   | VELA-T-020            | Integration (testbench)   | Sprint 2  |
| VELA-FR-012   | VELA-T-021            | Integration (testbench)   | Sprint 2  |
| VELA-FR-013   | VELA-T-022            | Synthesis report          | Sprint 2  |
| VELA-FR-014   | VELA-T-023            | Synthesis report          | Sprint 2  |
| VELA-FR-020   | VELA-T-030            | Integration (DRC)         | Sprint 3  |
| VELA-FR-023   | VELA-T-031            | System (visual)           | Sprint 3  |
| VELA-FR-026   | VELA-T-032            | System (UART log)         | Sprint 3  |
| VELA-NFR-001  | VELA-T-040            | Performance (stopwatch)   | Sprint 3  |
| VELA-NFR-004  | VELA-T-041            | Coverage (pytest-cov)     | Sprint 1  |
 
Full test specifications are in the IVVP document (VELA-IVVP-001).