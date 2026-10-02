"""
gui_simulator.py — VELA FPGA Board Simulator

Simulates the Arty Z7-20 + Pmod OLED behavior:
  - 96×64 OLED display (×6 upscale, nearest-neighbor)
  - Mode switch: CNN inference / Image filters
  - Image navigation and filter selection via keyboard

Controls:
  ←/→     Previous/Next image     (BTN0/BTN1)
  ↑/↓     Previous/Next filter    (BTN2/BTN3)
  M       Toggle mode CNN/Filter  (SW0)
  D       Toggle debug panel
  Q/Esc   Quit

REQ: VELA-FR-007
"""

import argparse
import tkinter as tk
from tkinter import font as tkfont
import numpy as np

from data_loader import load_batch, crop_center, normalize, CIFAR10_CLASSES
from weights_loader import load_weights
from cnn import cnn_forward
from filters import grayscale, sobel, gaussian, median, threshold


# ─── Constants ───────────────────────────────────────────────────────────────

OLED_W, OLED_H = 96, 64             # Pmod OLED SSD1331 resolution
SCALE          = 6                  # Upscale factor for display
IMG_SIZE       = 24                 # CIFAR post-crop
FILTERS        = ["original", "grayscale", "sobel", "gaussian", "median", "threshold"]


# ─── OledCanvas ──────────────────────────────────────────────────────────────

class OledCanvas:
    """
    Simulates the 96×64 Pmod OLED (SSD1331) as a Tk Canvas.
    Renders pixel-perfect with nearest-neighbor upscaling.
    """

    def __init__(self, parent):
        self.canvas = tk.Canvas(
            parent,
            width=OLED_W * SCALE,
            height=OLED_H * SCALE,
            bg="black",
            highlightthickness=2,
            highlightbackground="#444444"
        )
        self._photo = None

    def render(self, image_chw, text_line=""):
        """
        Draw a CHW float image centered on the OLED, with optional text below.

        Args:
            image_chw: np.ndarray (C, H, W) — 1 or 3 channels, any value range
            text_line: string to render below the image
        """
        # Create black 96×64 RGB buffer
        oled = np.zeros((OLED_H, OLED_W, 3), dtype=np.uint8)

        # Normalize image to 0-255 range
        img = image_chw.copy()
        c, h, w = img.shape

        # Handle single-channel (grayscale/sobel/threshold) → replicate to RGB
        if c == 1:
            img = np.repeat(img, 3, axis=0)

        # Normalize to [0, 255]
        imin, imax = img.min(), img.max()
        if imax - imin > 1e-6:
            img = (img - imin) / (imax - imin) * 255.0
        else:
            img = np.clip(img, 0, 255)

        img_hwc = img.transpose(1, 2, 0).astype(np.uint8)  # CHW → HWC

        # Center the 24×24 image in the 96×64 OLED (leave room for text)
        # Vertically: offset to leave space for text at bottom
        y_offset = (OLED_H - 10 - h) // 2   # 10px reserved for text zone
        x_offset = (OLED_W - w) // 2

        oled[y_offset:y_offset + h, x_offset:x_offset + w, :] = img_hwc

        # Render text line as pixel font (simple 4×6 bitmap)
        if text_line:
            self._draw_text(oled, text_line, y=OLED_H - 8)

        # Upscale nearest-neighbor
        oled_scaled = np.repeat(np.repeat(oled, SCALE, axis=0), SCALE, axis=1)

        # Convert to Tk PhotoImage
        self._photo = tk.PhotoImage(
            width=OLED_W * SCALE,
            height=OLED_H * SCALE
        )
        # PPM header trick for fast pixel transfer
        header = f"P6 {OLED_W * SCALE} {OLED_H * SCALE} 255 ".encode()
        self._photo.configure(data=header + oled_scaled.tobytes())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _draw_text(self, buf, text, y, color=(0, 200, 0)):
        """
        Minimal pixel font renderer (4×6 glyphs).
        Only supports uppercase, digits, space, dot, percent, colon.
        """
        glyphs = _get_pixel_font()
        x = 2
        for ch in text.upper():
            glyph = glyphs.get(ch)
            if glyph is None:
                x += 4
                continue
            for row_i, row in enumerate(glyph):
                for col_i, pixel in enumerate(row):
                    if pixel and y + row_i < buf.shape[0] and x + col_i < buf.shape[1]:
                        buf[y + row_i, x + col_i] = color
            x += len(glyph[0]) + 1



# ─── Police bitmap ───────────────────────────────────────────────────────────

def _get_pixel_font():
    """
    Minimal 4×6 pixel font for OLED text rendering.
    Returns dict mapping char → list of rows (each row = list of 0/1).
    """
    # Each glyph is 4 wide × 6 tall (some chars 3 wide)
    return {
        'A': [[0,1,0],[1,0,1],[1,1,1],[1,0,1],[1,0,1],[0,0,0]],
        'B': [[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,1,0],[0,0,0]],
        'C': [[0,1,1],[1,0,0],[1,0,0],[1,0,0],[0,1,1],[0,0,0]],
        'D': [[1,1,0],[1,0,1],[1,0,1],[1,0,1],[1,1,0],[0,0,0]],
        'E': [[1,1,1],[1,0,0],[1,1,0],[1,0,0],[1,1,1],[0,0,0]],
        'F': [[1,1,1],[1,0,0],[1,1,0],[1,0,0],[1,0,0],[0,0,0]],
        'G': [[0,1,1],[1,0,0],[1,0,1],[1,0,1],[0,1,1],[0,0,0]],
        'H': [[1,0,1],[1,0,1],[1,1,1],[1,0,1],[1,0,1],[0,0,0]],
        'I': [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1],[0,0,0]],
        'K': [[1,0,1],[1,1,0],[1,0,0],[1,1,0],[1,0,1],[0,0,0]],
        'L': [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,1,1],[0,0,0]],
        'M': [[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,0,0,0]],
        'N': [[1,0,1],[1,1,1],[1,1,1],[1,0,1],[1,0,1],[0,0,0]],
        'O': [[0,1,0],[1,0,1],[1,0,1],[1,0,1],[0,1,0],[0,0,0]],
        'P': [[1,1,0],[1,0,1],[1,1,0],[1,0,0],[1,0,0],[0,0,0]],
        'R': [[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,0,1],[0,0,0]],
        'S': [[0,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,0,0]],
        'T': [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,0]],
        'U': [[1,0,1],[1,0,1],[1,0,1],[1,0,1],[0,1,0],[0,0,0]],
        'V': [[1,0,1],[1,0,1],[1,0,1],[0,1,0],[0,1,0],[0,0,0]],
        'W': [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1],[0,1,1,0],[0,0,0,0]],
        'X': [[1,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,1],[0,0,0]],
        '0': [[0,1,0],[1,0,1],[1,0,1],[1,0,1],[0,1,0],[0,0,0]],
        '1': [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1],[0,0,0]],
        '2': [[1,1,0],[0,0,1],[0,1,0],[1,0,0],[1,1,1],[0,0,0]],
        '3': [[1,1,0],[0,0,1],[0,1,0],[0,0,1],[1,1,0],[0,0,0]],
        '4': [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1],[0,0,0]],
        '5': [[1,1,1],[1,0,0],[1,1,0],[0,0,1],[1,1,0],[0,0,0]],
        '6': [[0,1,1],[1,0,0],[1,1,0],[1,0,1],[0,1,0],[0,0,0]],
        '7': [[1,1,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,0,0]],
        '8': [[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[0,0,0]],
        '9': [[0,1,0],[1,0,1],[0,1,1],[0,0,1],[1,1,0],[0,0,0]],
        ' ': [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
        '.': [[0],[0],[0],[0],[1],[0]],
        '%': [[1,0,1],[0,0,1],[0,1,0],[1,0,0],[1,0,1],[0,0,0]],
        ':': [[0],[1],[0],[1],[0],[0]],
        '-': [[0,0,0],[0,0,0],[1,1,1],[0,0,0],[0,0,0],[0,0,0]],
    }


# ─── VelaSimulator ───────────────────────────────────────────────────────────

class VelaSimulator:
    """Main simulator window — Layout B with debug panel."""

    def __init__(self, images_raw, images_pp, labels, weights):
        """
        Args:
            images_raw : (N, 3, 32, 32) float32 — original images (for debug panel)
            images_pp  : (N, 3, 24, 24) float32 — preprocessed images
            labels     : (N,) int — true labels
            weights    : dict — CNN weights from load_weights()
        """
        self.images_raw = images_raw
        self.images_pp  = images_pp
        self.labels     = labels
        self.weights    = weights
        self.n_images   = len(labels)

        # State
        self.img_idx    = 0
        self.filter_idx = 0       # index in FILTERS list
        self.mode       = "cnn"   # "cnn" or "filter"
        self.show_debug = True    # debug panel visible

        # Cache CNN results to avoid recomputation
        self._cnn_cache = {}

        # ─── Window setup ────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("VELA Simulator")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)

        # ─── Main layout ────────────────────────────────────────────
        main_frame = tk.Frame(self.root, bg="#1a1a1a")
        main_frame.pack(padx=15, pady=15)

        # Left: OLED display
        oled_frame = tk.Frame(main_frame, bg="#1a1a1a")
        oled_frame.pack(side=tk.LEFT)

        oled_label = tk.Label(
            oled_frame, text="PMOD OLED 96×64",
            fg="#666666", bg="#1a1a1a", font=("Courier", 9)
        )
        oled_label.pack(pady=(0, 5))

        self.oled = OledCanvas(oled_frame)
        self.oled.canvas.pack()

        # Status bar below OLED
        self.status_var = tk.StringVar(value="")
        status_label = tk.Label(
            oled_frame, textvariable=self.status_var,
            fg="#888888", bg="#1a1a1a", font=("Courier", 10), anchor="w"
        )
        status_label.pack(fill=tk.X, pady=(5, 0))

        # ─── Separator ──────────────────────────────────────────────
        self.separator = tk.Frame(main_frame, width=2, bg="#444444")
        self.separator.pack(side=tk.LEFT, fill=tk.Y, padx=15)

        # ─── Right: Debug panel ──────────────────────────────────────
        self.debug_frame = tk.Frame(main_frame, bg="#1a1a1a", width=220)
        self.debug_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Original image display
        tk.Label(
            self.debug_frame, text="Original 32×32",
            fg="#666666", bg="#1a1a1a", font=("Courier", 9)
        ).pack(pady=(0, 5))

        self.orig_canvas = tk.Canvas(
            self.debug_frame, width=128, height=128,
            bg="black", highlightthickness=1, highlightbackground="#333333"
        )
        self.orig_canvas.pack()
        self._orig_photo = None

        # Info labels
        info_frame = tk.Frame(self.debug_frame, bg="#1a1a1a")
        info_frame.pack(pady=(15, 0), anchor="w")

        self.info_vars = {}
        for key in ["Mode", "Image", "Vrai", "Prédit", "Proba", "Filtre"]:
            row = tk.Frame(info_frame, bg="#1a1a1a")
            row.pack(fill=tk.X, pady=1)
            tk.Label(
                row, text=f"{key}:", width=8, anchor="e",
                fg="#888888", bg="#1a1a1a", font=("Courier", 10)
            ).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            tk.Label(
                row, textvariable=var, anchor="w",
                fg="#cccccc", bg="#1a1a1a", font=("Courier", 10, "bold")
            ).pack(side=tk.LEFT, padx=(5, 0))
            self.info_vars[key] = var

        # ─── Controls help ───────────────────────────────────────────
        controls = tk.Label(
            self.root,
            text="←→ images   ↑↓ filtres   M mode   D debug   Q quit",
            fg="#555555", bg="#1a1a1a", font=("Courier", 9)
        )
        controls.pack(pady=(5, 10))

        # ─── Key bindings ────────────────────────────────────────────
        self.root.bind("<Left>",    lambda e: self._prev_image())
        self.root.bind("<Right>",   lambda e: self._next_image())
        self.root.bind("<Up>",      lambda e: self._prev_filter())
        self.root.bind("<Down>",    lambda e: self._next_filter())
        self.root.bind("m",         lambda e: self._toggle_mode())
        self.root.bind("M",         lambda e: self._toggle_mode())
        self.root.bind("d",         lambda e: self._toggle_debug())
        self.root.bind("D",         lambda e: self._toggle_debug())
        self.root.bind("q",         lambda e: self.root.quit())
        self.root.bind("Q",         lambda e: self.root.quit())
        self.root.bind("<Escape>",  lambda e: self.root.quit())

        # Initial render
        self._update()

    # ─── Navigation ──────────────────────────────────────────────────

    def _prev_image(self):
        self.img_idx = (self.img_idx - 1) % self.n_images
        self._update()

    def _next_image(self):
        self.img_idx = (self.img_idx + 1) % self.n_images
        self._update()

    def _prev_filter(self):
        self.filter_idx = (self.filter_idx - 1) % len(FILTERS)
        self._update()

    def _next_filter(self):
        self.filter_idx = (self.filter_idx + 1) % len(FILTERS)
        self._update()

    def _toggle_mode(self):
        self.mode = "filter" if self.mode == "cnn" else "cnn"
        self._update()

    def _toggle_debug(self):
        self.show_debug = not self.show_debug
        if self.show_debug:
            self.separator.pack(side=tk.LEFT, fill=tk.Y, padx=15)
            self.debug_frame.pack(side=tk.LEFT, fill=tk.Y)
        else:
            self.separator.pack_forget()
            self.debug_frame.pack_forget()

    # ─── Rendering ───────────────────────────────────────────────────

    def _update(self):
        """Refresh OLED display and debug panel."""
        img_pp  = self.images_pp[self.img_idx]     # (3, 24, 24) preprocessed
        img_raw = self.images_raw[self.img_idx]     # (3, 32, 32) original
        label   = int(self.labels[self.img_idx])

        if self.mode == "cnn":
            self._render_cnn(img_pp, label)
        else:
            self._render_filter(img_pp, label)

        # Update debug panel
        self._update_debug(img_raw, label)

        # Status bar
        self.status_var.set(
            f"  [{self.img_idx + 1}/{self.n_images}]  "
            f"Mode: {self.mode.upper()}"
        )

    def _render_cnn(self, img_pp, label):
        """CNN mode: run inference, display class + probability."""
        # Check cache
        if self.img_idx not in self._cnn_cache:
            probs, pred = cnn_forward(img_pp, self.weights)
            self._cnn_cache[self.img_idx] = (probs, pred)

        probs, pred = self._cnn_cache[self.img_idx]
        class_name = CIFAR10_CLASSES[pred]
        confidence = probs[pred] * 100

        text = f"{class_name} {confidence:.0f}%"
        self.oled.render(img_pp, text)

        # Debug info
        self.info_vars["Prédit"].set(f"{class_name}")
        self.info_vars["Proba"].set(f"{confidence:.1f}%")
        self.info_vars["Filtre"].set("—")

    def _render_filter(self, img_pp, label):
        """Filter mode: apply selected filter, display result."""
        fname = FILTERS[self.filter_idx]

        if fname == "original":
            filtered = img_pp
        elif fname == "grayscale":
            filtered = grayscale(img_pp)
        elif fname == "sobel":
            filtered = sobel(img_pp)
        elif fname == "gaussian":
            filtered = gaussian(img_pp)
        elif fname == "median":
            filtered = median(img_pp)
        elif fname == "threshold":
            gray = grayscale(img_pp)
            filtered = threshold(gray)

        self.oled.render(filtered, fname)

        # Debug info
        self.info_vars["Prédit"].set("—")
        self.info_vars["Proba"].set("—")
        self.info_vars["Filtre"].set(fname)

    def _update_debug(self, img_raw, label):
        """Update the debug panel with original image and info."""
        if not self.show_debug:
            return

        # Render original 32×32 upscaled to 128×128
        img_hwc = img_raw.transpose(1, 2, 0).astype(np.uint8)
        img_big = np.repeat(np.repeat(img_hwc, 4, axis=0), 4, axis=1)

        self._orig_photo = tk.PhotoImage(width=128, height=128)
        header = f"P6 128 128 255 ".encode()
        self._orig_photo.configure(data=header + img_big.tobytes())
        self.orig_canvas.delete("all")
        self.orig_canvas.create_image(0, 0, anchor=tk.NW, image=self._orig_photo)

        # Info labels
        self.info_vars["Mode"].set(self.mode.upper())
        self.info_vars["Image"].set(f"#{self.img_idx}")
        self.info_vars["Vrai"].set(CIFAR10_CLASSES[label])

    # ─── Run ─────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VELA FPGA Board Simulator")
    parser.add_argument("--images", type=int, default=100,
                        help="Number of random images to load (default: 100)")
    parser.add_argument("--data-dir", type=str,
                        default="../data/cifar-10-batches-bin",
                        help="Path to CIFAR-10 binary directory")
    parser.add_argument("--weights", type=str,
                        default="../data/CNN_coeff_3x3.txt",
                        help="Path to CNN weights file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible selection")
    args = parser.parse_args()

    import os

    print("VELA Simulator — loading data...")

    # Load all images then sample randomly
    batch_path = os.path.join(args.data_dir, "test_batch.bin")
    images_raw, labels = load_batch(batch_path)  # (10000, 3, 32, 32)

    rng = np.random.default_rng(args.seed)
    n_total = len(labels)
    n_select = min(args.images, n_total)
    indices = rng.choice(n_total, size=n_select, replace=False)
    indices.sort()  # Keep index order for consistency

    images_raw = images_raw[indices]
    labels     = labels[indices]

    # Preprocess
    from data_loader import preprocess
    images_pp = np.array([preprocess(img) for img in images_raw], dtype=np.float32)

    # Load weights
    weights = load_weights(args.weights)

    print(f"Loaded {n_select} random images. Starting GUI...")

    app = VelaSimulator(images_raw, images_pp, labels, weights)
    app.run()


if __name__ == "__main__":
    main()