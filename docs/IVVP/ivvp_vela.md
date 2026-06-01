# VELA — Vision Edge Learning Accelerator
## Integration, Verification and Validation Plan (IVVP)

> **Document ID:** VELA-IVVP-001

> **Version:** 1.0 

> **Status:** Draft 

> **Date:** 2026-05-10 

> **Author:** Gaël BOYA

> **Related document:** VELA-SRS-001 (Requirements)

--- 

## 1. Purpose and scope

This document defines how every requirement stated in VELA-SRS-001 is tested, by what means, under what conditions, and with what acceptance criteria. It also defines the integration order — the sequence in which sub-systems are assembled and verified before the full system is validated.

The fundamental principle: **each implementation phase produces a reference that is used to verify the next phase**.

```
Python golden model  →  verifies  →  C++ HLS implementation
C++ HLS output       →  verifies  →  FPGA hardware output
FPGA hardware        →  validates →  system-level requirement
```

--- 

## 2. Test identification scheme

Tests are identified as **VELA-T-XXX** where XXX is a three-digit number.

Each test record contains:
- **ID** — unique identifier
- **Type** — Unit / Integration / System / Performance
- **Covers** — requirement(s) from SRS
- **Preconditions** — what must be true before the test runs
- **Inputs** — test data or stimuli
- **Expected output** — exact criterion for pass/fail
- **Pass criterion** — binary go/no-go statement
- **Tool** — how the test is executed
- **Sprint** — when this test is performed

--- 

## 3. Phase 1 - Python golden model tests

### VELA-T-001 · Conv2d identity kernel

| Field                 | Value                                                                                     |
| :---                  | :---                                                                                      |
| **Type**              | Unit                                                                                      |
| **Covers**            | VELA-FR-001                                                                               |
| **Preconditions**     | `cnn.py` importable, NumPy installed                                                      |
| **Inputs**            | 1×5×5 input filled with `arange(25)`, 3×3 identity kernel (centre=1, rest=0), zero bias   |
| **Expected output**   | `conv2d_fast(img, kern, bias)` == `img` to within 1e-5                                    |
| **Pass criterion**    | `np.testing.assert_allclose(output, input, atol=1e-5)` passes                             |
| **Tool**              | `pytest tests/test_layers.py::test_conv2d_identity`                                       |
| **Sprint**            | 1                                                                                         |
 
---
 
### VELA-T-002 · Conv2d fast vs reference consistency
 
| Field                 | Value                                                         |
| :---                  | :---                                                          |
| **Type**              | Unit                                                          |
| **Covers**            | VELA-FR-001                                                   |
| **Preconditions**     | Both `conv2d()` and `conv2d_fast()` implemented               |
| **Inputs**            | Random 3×8×8 feature map, random 4×3×3×3 kernel, random bias  |
| **Expected output**   | `conv2d_fast` and `conv2d` produce identical results          |
| **Pass criterion**    | `np.testing.assert_allclose(fast, ref, atol=1e-4)` passes     |
| **Tool**              | `pytest tests/test_layers.py::test_conv2d_consistency`        |
| **Sprint**            | 1                                                             |
 
---
 
### VELA-T-003 · ReLU clips negative values
 
| Field                 | Value                                                     |
| :---                  | :---                                                      |
| **Type**              | Unit                                                      |
| **Covers**            | VELA-FR-001                                               |
| **Inputs**            | `[-3.0, -0.1, 0.0, 0.5, 2.0]`                             |
| **Expected output**   | `[0.0, 0.0, 0.0, 0.5, 2.0]`                               |
| **Pass criterion**    | `np.testing.assert_array_equal(relu(x), expected)` passes |
| **Tool**              | `pytest tests/test_layers.py::test_relu`                  |
| **Sprint**            | 1                                                         |
 
---
 
### VELA-T-004 · MaxPool returns correct maximum
 
| Field                 | Value                                                                             |
| :---                  | :---                                                                              |
| **Type**              | Unit                                                                              |
| **Covers**            | VELA-FR-001                                                                       |
| **Inputs**            | 1×4×4 map `[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]`, pool=2, stride=2    |
| **Expected output**   | `[[[6, 8], [14, 16]]]`                                                            |
| **Pass criterion**    | `np.testing.assert_array_equal(output, expected)` passes                          |
| **Tool**              | `pytest tests/test_layers.py::test_maxpool`                                       |
| **Sprint**            | 1                                                                                 |
 
---
 
### VELA-T-005 · Softmax sums to 1
 
| Field                 | Value                                             |
| :---                  | :---                                              |
| **Type**              | Unit                                              |
| **Covers**            | VELA-FR-001                                       |
| **Inputs**            | `[1.0, 2.0, 3.0, 4.0, 5.0]`                       |
| **Expected output**   | sum of probabilities == 1.0, argmax == 4          |
| **Pass criterion**    | `abs(sum(probs) - 1.0) < 1e-6` and `argmax == 4`  |
| **Tool**              | `pytest tests/test_layers.py::test_softmax`       |
| **Sprint**            | 1                                                 |
 
---
 
### VELA-T-006 · Image preprocessing shape and statistics
 
| Field | Value |
|-------|-------|
| **Type** | Unit |
| **Covers** | VELA-FR-003 |
| **Inputs** | Single CIFAR-10 raw image (3×32×32) |
| **Expected output** | Output shape (3×24×24), mean ≈ 0.0 (±0.01), std ≈ 1.0 (±0.1) |
| **Pass criterion** | Shape correct AND `abs(mean) < 0.01` AND `abs(std - 1.0) < 0.1` |
| **Tool** | `pytest tests/test_layers.py::test_preprocess` |
| **Sprint** | 1 |
 
---
 
### VELA-T-007 · Weight file loading and shape validation
 
| Field | Value |
|-------|-------|
| **Type** | Unit |
| **Covers** | VELA-FR-002 |
| **Preconditions** | `CNN_coeff_3x3.txt` accessible at configured path |
| **Inputs** | `CNN_coeff_3x3.txt` |
| **Expected output** | Dict with keys `conv1_w (64,3,3,3)`, `conv1_b (64,)`, `conv2_w (32,64,3,3)`, `conv2_b (32,)`, `conv3_w (20,32,3,3)`, `conv3_b (20,)`, `fc_w (10,180)`, `fc_b (10,)` |
| **Pass criterion** | All shapes match exactly, no exception raised |
| **Tool** | `pytest tests/test_weights.py::test_weight_shapes` |
| **Sprint** | 1 |
 
---
 
### VELA-T-010 · System accuracy on 1000 images (golden model)
 
| Field | Value |
|-------|-------|
| **Type** | System |
| **Covers** | VELA-FR-005 |
| **Preconditions** | `CNN_coeff_3x3.txt` and `test_batch.bin` accessible |
| **Inputs** | First 1000 images from CIFAR-10 `test_batch.bin` |
| **Expected output** | Top-1 accuracy ≥ 60% |
| **Pass criterion** | `correct / 1000 >= 0.60` |
| **Tool** | `pytest tests/test_system.py::test_accuracy_1000` (or `python top.py --images 1000`) |
| **Sprint** | 1 |
| **Note** | This is the exit criterion for Sprint 1. Do not proceed to Sprint 2 until this passes. |
 
---
 
### VELA-T-011 · Per-class accuracy report
 
| Field | Value |
|-------|-------|
| **Type** | System |
| **Covers** | VELA-FR-006 |
| **Inputs** | 1000 images from test_batch.bin |
| **Expected output** | Confusion matrix, per-class accuracy printed to stdout |
| **Pass criterion** | No class has 0% accuracy (would indicate a systematic data ordering bug) |
| **Tool** | `python top.py --images 1000 --confusion` |
| **Sprint** | 1 |
 
---

 
## 4. Phase 2 — HLS C++ tests
 
### VELA-T-020 · C++ double vs Python output agreement
 
| Field | Value |
|-------|-------|
| **Type** | Integration |
| **Covers** | VELA-FR-010 |
| **Preconditions** | Python IVV reference files generated (`python generate_ivv_references.py`) |
| **Inputs** | 10 reference images from `ivv_refs/image_N.bin` |
| **Expected output** | C++ double output matches Python scores to within 1e-4 per element |
| **Pass criterion** | `max_delta < 1e-4` across all 10 × 10 output elements |
| **Tool** | `make all && ./top_cnn_reference` (reads ivv_refs/) |
| **Sprint** | 2 |
 
---
 
### VELA-T-021 · ap_fixed vs Python output agreement
 
| Field | Value |
|-------|-------|
| **Type** | Integration |
| **Covers** | VELA-FR-012 |
| **Preconditions** | VELA-T-020 passed. Fixed-point formats chosen per `fixed_point_analysis.md` |
| **Inputs** | Same 10 reference images as VELA-T-020 |
| **Expected output** | ap_fixed output argmax matches Python argmax on ≥ 9/10 images |
| **Pass criterion** | Concordance rate ≥ 90% on reference set |
| **Tool** | `make all && ./top_cnn_hardware` (reads ivv_refs/) |
| **Sprint** | 2 |
| **Note** | Argmax match is more meaningful than absolute score match for fixed-point. |
 
---
 
### VELA-T-022 · HLS synthesis timing closure
 
| Field | Value |
|-------|-------|
| **Type** | Synthesis |
| **Covers** | VELA-FR-013 |
| **Preconditions** | Vitis HLS 2024.1 installed. `cnn_top` function correct per VELA-T-021 |
| **Inputs** | Vitis HLS project targeting `xc7z020clg400-1` at 100 MHz |
| **Expected output** | Synthesis report shows no timing violation (WNS ≥ 0 ns) |
| **Pass criterion** | HLS synthesis report: `Timing: Met` |
| **Tool** | Vitis HLS GUI or `vitis_hls -f run_hls.tcl` |
| **Sprint** | 2 |
 
---
 
### VELA-T-023 · HLS resource budget
 
| Field | Value |
|-------|-------|
| **Type** | Synthesis |
| **Covers** | VELA-FR-014, VELA-FR-015 |
| **Inputs** | HLS synthesis report |
| **Expected output** | DSP ≤ 176 (80% of 220), BRAM ≤ 270 KB (≈ 80 blocks, 43% of 630 KB) |
| **Pass criterion** | Both limits respected simultaneously |
| **Tool** | Vitis HLS synthesis report (`solution/syn/report/cnn_top_csynth.rpt`) |
| **Sprint** | 2 |
 
---
 
### VELA-T-024 · IP export integrity
 
| Field | Value |
|-------|-------|
| **Type** | Integration |
| **Covers** | VELA-FR-017 |
| **Inputs** | Exported IP directory `impl_3_cnn_hw/ip/cnn_hw_v1_0/` |
| **Expected output** | `component.xml` present, IP added to Vivado IP catalog without error |
| **Pass criterion** | Vivado reports "IP is up to date" with no critical errors |
| **Tool** | Vivado → IP Catalog → Add Repository |
| **Sprint** | 2 |
 
---
 
## 5. Phase 3 — SoC integration tests
 
### VELA-T-030 · Block design DRC clean
 
| Field | Value |
|-------|-------|
| **Type** | Integration |
| **Covers** | VELA-FR-020 |
| **Preconditions** | All IP blocks added, connections made, Zynq preset applied |
| **Inputs** | Vivado block design |
| **Expected output** | Design Rule Check reports 0 critical errors |
| **Pass criterion** | Vivado DRC: 0 errors (warnings acceptable) |
| **Tool** | Vivado → Run Connection Automation → Validate Design (F6) |
| **Sprint** | 3 |
 
---
 
### VELA-T-031 · UART debug output on first image
 
| Field | Value |
|-------|-------|
| **Type** | System |
| **Covers** | VELA-FR-026 |
| **Preconditions** | Bitstream programmed on Arty Z7-20. microSD with test images inserted. UART terminal open (115200 8N1) |
| **Inputs** | Power-on or BTN0 reset |
| **Expected output** | UART prints `{"image":0, "class":"airplane", "confidence":73}` (values will vary) within 2 seconds |
| **Pass criterion** | Valid JSON received with class name from CIFAR-10 set (10 known names) |
| **Tool** | PuTTY / minicom / `python -m serial.tools.miniterm /dev/ttyUSB1 115200` |
| **Sprint** | 3 |
 
---
 
### VELA-T-032 · OLED display visible output
 
| Field | Value |
|-------|-------|
| **Type** | System |
| **Covers** | VELA-FR-023, VELA-FR-031 |
| **Preconditions** | VELA-T-031 passed |
| **Inputs** | System running, image 0 loaded |
| **Expected output** | OLED shows a coloured image patch and a class label in text |
| **Pass criterion** | Visual inspection: text is readable, image is non-black, display does not flicker |
| **Tool** | Visual inspection |
| **Sprint** | 3 |
 
---
 
### VELA-T-033 · Hardware vs Python concordance (10 images)
 
| Field | Value |
|-------|-------|
| **Type** | Validation |
| **Covers** | VELA-FR-022, VELA-NFR-004 |
| **Preconditions** | VELA-T-031 and VELA-T-032 passed |
| **Inputs** | 10 images from test_batch.bin (images 0–9, covering at least 5 different classes) |
| **Expected output** | FPGA predicted class matches Python predicted class on ≥ 9/10 images |
| **Pass criterion** | Concordance rate ≥ 90% |
| **Tool** | `python ivv_fpga.py` (reads UART log, compares with Python reference) |
| **Sprint** | 3 |
| **Note** | This is the primary IVV exit criterion. A failure here indicates a fixed-point or memory ordering bug, not a CNN accuracy problem. |
 
---
 
### VELA-T-040 · End-to-end latency
 
| Field | Value |
|-------|-------|
| **Type** | Performance |
| **Covers** | VELA-NFR-001 |
| **Preconditions** | VELA-T-033 passed |
| **Inputs** | BTN1 press (next image trigger) |
| **Measurement** | Time from BTN1 press to OLED update (measured with stopwatch or UART timestamp) |
| **Expected output** | Total latency ≤ 2000 ms |
| **Pass criterion** | Average of 5 consecutive measurements ≤ 2000 ms |
| **Tool** | Stopwatch or UART timestamp differential |
| **Sprint** | 3 |
 
---
 
### VELA-T-041 · Python test coverage
 
| Field | Value |
|-------|-------|
| **Type** | Static analysis |
| **Covers** | VELA-NFR-004 |
| **Inputs** | `pytest --cov=. --cov-report=term-missing tests/` |
| **Expected output** | Line coverage of `cnn.py` and `data_loader.py` ≥ 90% |
| **Pass criterion** | `pytest-cov` report: coverage ≥ 90% on core modules |
| **Tool** | `pip install pytest-cov && pytest --cov` |
| **Sprint** | 1 |
 
---
 
## 6. IVV reference file generation
 
The following Python script generates binary reference files used by the C++ testbench (VELA-T-020, VELA-T-021) and the FPGA validation script (VELA-T-033). Run it once after Sprint 1 is complete and commit the outputs to `ivv_refs/`.
 
```python
# generate_ivv_references.py
import os
import numpy as np
from data_loader import load_dataset, CIFAR10_CLASSES
from weights_loader import load_weights
from cnn import cnn_forward
 
DATA_DIR     = "./data/cifar-10-batches-bin"
WEIGHTS_FILE = "./data/CNN_coeff_3x3.txt"
OUT_DIR      = "./ivv_refs"
N_IMAGES     = 10
 
os.makedirs(OUT_DIR, exist_ok=True)
images, labels = load_dataset(DATA_DIR, max_images=N_IMAGES)
weights = load_weights(WEIGHTS_FILE)
 
manifest = []
for i, (img, lbl) in enumerate(zip(images, labels)):
    probs, pred = cnn_forward(img, weights)
    img.astype(np.float32).tofile(f"{OUT_DIR}/image_{i}.bin")
    probs.astype(np.float32).tofile(f"{OUT_DIR}/scores_{i}.bin")
    manifest.append({
        "index": i,
        "true_label": int(lbl),
        "true_class": CIFAR10_CLASSES[lbl],
        "pred_label": int(pred),
        "pred_class": CIFAR10_CLASSES[pred],
        "confidence": float(probs[pred])
    })
    print(f"[{i:2d}] true={CIFAR10_CLASSES[lbl]:<12} pred={CIFAR10_CLASSES[pred]:<12} conf={probs[pred]:.3f}")
 
import json
with open(f"{OUT_DIR}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n[OK] {N_IMAGES} reference files written to {OUT_DIR}/")
```
 
---
 
## 7. Sprint exit criteria summary
 
| Sprint | Exit criterion | Test(s) | Status |
|--------|---------------|---------|--------|
| Sprint 0 | Repository structure clean, impl_1 reconstructible | Manual | ⬜ |
| Sprint 1 | Python accuracy ≥ 60% on 1000 images, all unit tests green | T-001 to T-011 | ⬜ |
| Sprint 2 | HLS timing met, IP exported, ap_fixed concordance ≥ 90% | T-020 to T-024 | ⬜ |
| Sprint 3 | FPGA concordance ≥ 90%, latency ≤ 2 s, OLED visible | T-030 to T-041 | ⬜ |
 
---
 
## 8. Known risks and mitigations
 
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Fixed-point overflow in Conv layers | Medium | High | Run range analysis before choosing ap_fixed formats (VELA-T-021 detects this) |
| HLS timing closure failure at 100 MHz | Low | Medium | Reduce to 50 MHz as fallback; document the trade-off |
| BRAM insufficient for all weights | Low | High | Partition weights across multiple BRAM blocks or store in DDR3 |
| OLED SPI timing incompatible with HLS clock domain | Medium | Medium | Use dedicated SPI controller in PL (reused from impl_1), independent of CNN clock |
| Reshape order mismatch Python ↔ C++ | High | High | VELA-T-020 detects this immediately; fix before HLS synthesis |
 
---
 
*End of document — VELA-IVVP-001 v1.0*