"""
weights_loader.py — Loads CNN weights from CNN_coeff_3x3.txt file.

File format (numpy print export from TensorFlow):
─────────────────────────────────────────────────
  tensor_name:  conv1/biases
  [-0.961  -0.239  ...]           ← space-separated floats, no commas
 
  tensor_name:  conv1/weights
  [[[[  7.76e-01  -2.48e-01 ...   ← multi-dimensional, no commas
 
The file contains 8 named blocks in this order:
  conv1/biases    → (64,)
  conv1/weights   → (3, 3, 3, 64)     TensorFlow: (kH, kW, C_in, C_out)
  conv2/biases    → (32,)
  conv2/weights   → (3, 3, 64, 32)    TensorFlow: (kH, kW, C_in, C_out)
  conv3/biases    → (20,)
  conv3/weights   → (3, 3, 32, 20)    TensorFlow: (kH, kW, C_in, C_out)
  local3/biases   → (10,)
  local3/weights  → (180, 10)         TensorFlow: (C_in, C_out)
 
CRITICAL — axis transposition:
  TensorFlow conv kernel : (kH, kW, C_in, C_out)
  Required for conv2d_fast: (C_out, C_in, kH, kW)
  → Transposition: .reshape(kH,kW,C_in,C_out).transpose(3,2,0,1)
 
  TensorFlow FC weights  : (n_in, n_out) = (180, 10)
  Required for fully_connected: (n_out, n_in) = (10, 180)
  → Transposition: .reshape(180,10).T
 
REQ: VELA-FR-002
"""

import os
import re
import numpy as np


## ———— Architecture reference ————————————————————————————————————————————————————— ##
# Maps TensorFlow tensor names to:
#   - tf_shape      : shape as stored in the file (TensorFlow convention)
#   - out_shhape    : shape expected by conv2d_fast / fully_connected (CNN convention)
#   - transpose     : axes permutation to apply after reshape, or None

ARCH = {
    "conv1/biases": {
        "tf_shape":     (64,),
        "out_shape":    (64,),
        "transpose":    None
    },
    "conv1/weights": {
        "tf_shape":     (3, 3, 3, 64),                  # (kH, kW, C_in, C_out)
        "out_shape":    (64, 3, 3, 3),                  # (C_out, C_in, kH, kW)
        "transpose":    (3, 2, 0, 1)
    },
    "conv2/biases": {
        "tf_shape":     (32,),
        "out_shape":    (32,),
        "transpose":    None
    },
    "conv2/weights": {
        "tf_shape":     (3, 3, 64, 32),                 # (kH, kW, C_in, C_out)
        "out_shape":    (32, 64, 3, 3),                 # (C_out, C_in, kH, kW)
        "transpose":    (3, 2, 0, 1)
    },
    "conv3/biases": {
        "tf_shape":     (20,),
        "out_shape":    (20,),
        "transpose":    None
    },
    "conv3/weights": {
        "tf_shape":     (3, 3, 32, 20),                 # (kH, kW, C_in, C_out)
        "out_shape":    (20, 32, 3, 3),                 # (C_out, C_in, kH, kW)
        "transpose":    (3, 2, 0, 1)
    },
    "local3/biases": {
        "tf_shape":     (10,),
        "out_shape":    (10,),
        "transpose":    None
    },
    "local3/weights": {
        "tf_shape":     (180, 10),                      # (n_in, n_out) TensorFlow
        "out_shape":    (10, 180),                      # (n_out, n_in) for W @ x
        "transpose":    (1, 0)                          # simple .T
    },
}

# Public key aliases (used throughout the codebase):
# Maps internal tf names -> clean public names
KEY_MAP = {
    "conv1/biases":     "conv1_b",
    "conv1/weights":    "conv1_w",
    "conv2/biases":     "conv2_b",
    "conv2/weights":    "conv2_w",
    "conv3/biases":     "conv3_b",
    "conv3/weights":    "conv3_w",
    "local3/biases":    "fc_b",
    "local3/weights":   "fc_w"
}


## ———— Parsing ———————————————————————————————————————————————————————————————————— ##

def _parse_blocks(filepath: str) -> dict[str, np.ndarray]:
    """
    Parse the CNN_coeff_3x3.txt file into named raw float arrays.
 
    Strategy:
      1. Split content on 'tensor_name:' markers.
      2. For each block, extract all float values with regex
         (handles both space-separated flat arrays and nested brackets).
      3. Validate the count against ARCH expectations.
 
    Returns:
        dict mapping tf_name → flat np.ndarray float32
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
 
    # Split on tensor_name markers
    raw_blocks = re.split(r'tensor_name:\s+', content)
    raw_blocks = [b.strip() for b in raw_blocks if b.strip()]
 
    if len(raw_blocks) != 8:
        raise ValueError(
            f"Expected 8 tensor blocks in {filepath}, found {len(raw_blocks)}.\n"
            "The file may be corrupted or use a different format."
        )
 
    result = {}
    float_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
 
    for block in raw_blocks:
        lines   = block.split('\n')
        tf_name = lines[0].strip()
        body    = '\n'.join(lines[1:])
 
        if tf_name not in ARCH:
            raise ValueError(
                f"Unknown tensor name '{tf_name}' in {filepath}.\n"
                f"Expected one of: {list(ARCH.keys())}"
            )
 
        floats = [float(x) for x in float_pattern.findall(body)]
        arr    = np.array(floats, dtype=np.float32)
 
        # Validate count
        expected_count = 1
        for dim in ARCH[tf_name]["tf_shape"]:
            expected_count *= dim
 
        if len(arr) != expected_count:
            raise ValueError(
                f"Tensor '{tf_name}': expected {expected_count} values, "
                f"found {len(arr)}.\n"
                f"Expected shape: {ARCH[tf_name]['tf_shape']}"
            )
 
        result[tf_name] = arr
 
    return result

## ———— Public API ————————————————————————————————————————————————————————————————— ##

def load_weights(filepath: str) -> dict[str, np.ndarray]:
    """
    Load and return CNN weights ready for use in conv2d_fast() and
    fully_connected().
 
    Performs:
      1. Parse all 8 tensor blocks from the file.
      2. Reshape each tensor to its TensorFlow shape.
      3. Transpose kernel weights from TF (kH,kW,C_in,C_out)
         to CHW (C_out,C_in,kH,kW).
      4. Transpose FC weights from TF (n_in,n_out) to (n_out,n_in).
      5. Return a dict with clean public key names.
 
    Args:
        filepath : path to CNN_coeff_3x3.txt
 
    Returns:
        dict with keys:
          conv1_w : np.ndarray (64, 3,  3, 3)
          conv1_b : np.ndarray (64,)
          conv2_w : np.ndarray (32, 64, 3, 3)
          conv2_b : np.ndarray (32,)
          conv3_w : np.ndarray (20, 32, 3, 3)
          conv3_b : np.ndarray (20,)
          fc_w    : np.ndarray (10, 180)
          fc_b    : np.ndarray (10,)
 
    Raises:
        FileNotFoundError : if filepath does not exist
        ValueError        : if the file structure does not match expectations
 
    REQ: VELA-FR-002
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weight file not found: {filepath}")
 
    raw = _parse_blocks(filepath)
    weights = {}
 
    for tf_name, spec in ARCH.items():
        arr = raw[tf_name]
 
        # Reshape to TensorFlow shape
        arr = arr.reshape(spec["tf_shape"])
 
        # Transpose to CHW convention if needed
        if spec["transpose"] is not None:
            arr = arr.transpose(spec["transpose"])
 
        # Verify final shape
        assert arr.shape == spec["out_shape"], (
            f"Internal error: '{tf_name}' has shape {arr.shape} "
            f"after transpose, expected {spec['out_shape']}"
        )
 
        public_key = KEY_MAP[tf_name]
        weights[public_key] = arr.astype(np.float32)
 
    return weights
 
 
def load_class_names(meta_filepath: str) -> list[str]:
    """
    Load CIFAR-10 class names from batches.meta.txt.
 
    File format: one class name per line, in label order (0=first line).
 
    Args:
        meta_filepath : path to batches.meta.txt
 
    Returns:
        list of 10 class name strings
 
    REQ: VELA-FR-002
    """
    if not os.path.exists(meta_filepath):
        raise FileNotFoundError(
            f"CIFAR-10 meta file not found: {meta_filepath}\n"
            "Expected: batches.meta.txt in the cifar-10-batches-bin directory."
        )
 
    with open(meta_filepath, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
 
    if len(names) != 10:
        raise ValueError(
            f"Expected 10 class names in {meta_filepath}, found {len(names)}."
        )
 
    return names
 
 
def print_weights_summary(weights: dict) -> None:
    """
    Print a formatted summary of loaded weights (shape and statistics).
    """
    print("─" * 68)
    print(f"  {'Key':<10} {'Shape':<20} {'Min':>9} {'Max':>9} {'Mean':>9}")
    print("─" * 68)
    for key, arr in weights.items():
        print(f"  {key:<10} {str(arr.shape):<20} "
              f"{arr.min():9.4f} {arr.max():9.4f} {arr.mean():9.4f}")
    print("─" * 68)
    total = sum(a.size for a in weights.values())
    print(f"  Total parameters: {total:,}")


## ———— Self-test ────────────────────────────────────────────────────────────────── ##
 
if __name__ == "__main__":
    import sys
 
    WEIGHTS_FILE = "../../data/CNN_coeff_3x3.txt"
    META_FILE    = "../../data/cifar-10-batches-bin/batches.meta.txt"
 
    print("=== weights_loader.py self-test ===\n")
 
    # 1. Load weights
    try:
        w = load_weights(WEIGHTS_FILE)
        print("[OK] Weights loaded successfully\n")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
 
    # 2. Print summary
    print_weights_summary(w)
 
    # 3. Verify all shapes
    print("\n=== Shape verification ===")
    expected_shapes = {
        "conv1_w": (64, 3,  3, 3),
        "conv1_b": (64,),
        "conv2_w": (32, 64, 3, 3),
        "conv2_b": (32,),
        "conv3_w": (20, 32, 3, 3),
        "conv3_b": (20,),
        "fc_w":    (10, 180),
        "fc_b":    (10,),
    }
    all_ok = True
    for key, exp_shape in expected_shapes.items():
        actual = w[key].shape
        ok = actual == exp_shape
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {key:<10} {str(actual):<20} (expected {exp_shape})")
        if not ok:
            all_ok = False
 
    # 4. Load class names
    print("\n=== Class names ===")
    try:
        classes = load_class_names(META_FILE)
        for i, name in enumerate(classes):
            print(f"  {i}: {name}")
        print(f"\n[OK] {len(classes)} class names loaded")
    except Exception as e:
        print(f"[WARN] Could not load class names: {e}")
 
    print(f"\n=== Result: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ===")
    sys.exit(0 if all_ok else 1)
