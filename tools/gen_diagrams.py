"""
gen_diagrams.py — Generate the SVG diagrams of docs/diagrams/.

All diagrams are plain SVG (no script), rendered by GitHub and any browser.
Animated ones use CSS keyframes inside the SVG; with prefers-reduced-motion
the animation is disabled and the still image stays complete and readable.

Diagrams:
    pixel_grid.svg          4x4 RGB image (TN001)
    chw_memory.svg          CHW layout in memory, stride of a pixel (TN001)
    hwc_memory.svg          HWC layout in memory (TN001)
    hw_methodology.svg      the 8 steps of the HW methodology, mapped to VELA (animated)
    vela_platform.svg       prototyping platform adapted to VELA (animated data flows)
    cnn_inference_chain.svg feature maps of the VELA CNN, layer by layer (animated)
    cnn_convolution.svg     multi-channel 3x3 convolution, two kernels (animated)
    cnn_maxpool.svg         3x3 / stride 2 maxpool with SAME padding (animated)
    cnn_fc_softmax.svg      HWC reshape, fully connected layer, softmax (animated)

Usage:
    python tools/gen_diagrams.py [out_dir]          (default: docs/diagrams)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ─── Style ───────────────────────────────────────────────────────────────────

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"
INK, MUTED, LINE, PAPER = "#1f2a2e", "#5b676b", "#c9d1ce", "#ffffff"
ACCENT, ACCENT_SOFT = "#a4562a", "#f6e9df"
CH = {"R": "#d64541", "G": "#3a9d5d", "B": "#3b6fd1"}           # channel colors
CH_SOFT = {"R": "#f7d6d4", "G": "#d3ecdb", "B": "#d6e1f6"}
STATUS = {                                                        # fill, stroke, label
    "done":    ("#e7f1ea", "#2f6f4e", "FAIT"),
    "phase3":  ("#fdf3e1", "#b7791f", "PHASE 3"),
    "future":  ("#eef0f1", "#8a949a", "FUTUR"),
}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(width, height, body, title, css=""):
    """Complete SVG document with a white background and an accessible title."""
    style = f"""
  text {{ font-family: {FONT}; }}
  .mono {{ font-family: {MONO}; }}
  .muted {{ fill: {MUTED}; }}
  {css}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}"""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{esc(title)}">\n'
            f'<title>{esc(title)}</title>\n'
            f'<style>{style}\n</style>\n'
            f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{MUTED}"/></marker>'
            f'<marker id="arrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{ACCENT}"/></marker></defs>\n'
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>\n'
            f'{body}\n</svg>\n')


def rect(x, y, w, h, fill="none", stroke=INK, sw=1, rx=0, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def text(x, y, s, size=13, anchor="middle", weight="normal", cls="", fill=None, extra=""):
    # Explicit fill attribute (a CSS "text { fill }" rule would override it);
    # the .muted class still wins over the attribute, as intended.
    f = f' fill="{fill or INK}"'
    c = f' class="{cls}"' if cls else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{c}{f} {extra}>{esc(s)}</text>')


def marker(arrow):
    """arrow: False (no arrowhead), True (grey arrowhead) or "A" (accent arrowhead)."""
    if not arrow:
        return ""
    return ' marker-end="url(#arrowA)"' if arrow == "A" else ' marker-end="url(#arrow)"'


def line(x1, y1, x2, y2, stroke=MUTED, sw=1.5, arrow=True, dash="", extra=""):
    m = marker(arrow)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}{m} {extra}/>')


def path(d, stroke=MUTED, sw=1.5, arrow=True, fill="none", dash="", extra=""):
    m = marker(arrow)
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{ds}{m} {extra}/>'


def sequence_css(name, n, period, dim=0.28):
    """
    Keyframes that highlight item k of n during [k/n, (k+1)/n] of the period:
    use with animation-delay = k * period / n on each item.
    """
    on = 100.0 / n
    return (f"@keyframes {name} {{ 0%, {on - 0.5:.2f}% {{ opacity: 1; }} "
            f"{on:.2f}%, 100% {{ opacity: {dim}; }} }}\n"
            f"  .{name} {{ animation: {name} {period}s linear infinite; }}")


# ─── TN001: image layouts ────────────────────────────────────────────────────

def pixel_grid():
    cell, gap, x0, y0 = 30, 18, 140, 70
    body = [text(20, 30, "Image RGB : H = 4, W = 4, C = 3  (48 valeurs)", 15, "start", "bold")]
    for w in range(4):
        body.append(text(x0 + w * (cell + gap) + cell / 2, y0 - 12, f"w = {w}", 12, cls="muted"))
    for h in range(4):
        y = y0 + h * (3 * cell + 14)
        body.append(text(x0 - 16, y + 1.5 * cell + 5, f"h = {h}", 12, "end", cls="muted"))
        for w in range(4):
            x = x0 + w * (cell + gap)
            for k, c in enumerate("RGB"):
                body.append(rect(x, y + k * cell, cell, cell, CH[c], PAPER, 1))
                body.append(text(x + cell / 2, y + k * cell + 19, f"{c}{h}{w}", 10, fill=PAPER,
                                 weight="bold"))
            body.append(rect(x - 2, y - 2, cell + 4, 3 * cell + 4, "none", INK, 1, 3))
    body.append(text(20, 520, "Chaque pixel (h, w) porte 3 valeurs : R, G, B.", 12, "start",
                     cls="muted"))
    return svg(440, 540, "\n".join(body), "Grille de pixels 4 x 4 RGB")


def memory_strip(order, title, formula, highlight, note):
    """48-cell memory strip; highlight = list of indices of pixel (h=1, w=2)."""
    cell, x0, y0 = 19, 30, 110
    body = [text(20, 30, title, 15, "start", "bold"),
            text(20, 54, formula, 13, "start", cls="mono"),
            text(20, 76, note, 12, "start", cls="muted")]
    for i, (c, h, w) in enumerate(order):
        x = x0 + i * cell
        hl = i in highlight
        body.append(rect(x, y0, cell - 1, 34, CH[c] if hl else CH_SOFT[c],
                         INK if hl else PAPER, 2 if hl else 1,
                         extra='class="hl"' if hl else ""))
        body.append(text(x + cell / 2 - 0.5, y0 + 15, c, 9, weight="bold",
                         fill=PAPER if hl else CH[c]))
        body.append(text(x + cell / 2 - 0.5, y0 + 28, f"{h}{w}", 8,
                         fill=PAPER if hl else MUTED))
        if i % 4 == 0 or i == 47:
            body.append(text(x + cell / 2, y0 + 50, str(i), 9, cls="muted"))
    for i in highlight:
        x = x0 + i * cell + cell / 2
        body.append(line(x, y0 - 16, x, y0 - 3, ACCENT, 1.5, "A"))
    body.append(text(x0, y0 + 74, "adresse →  (indices des valeurs du pixel h = 1, w = 2 en "
                     "surbrillance : " + ", ".join(map(str, highlight)) + ")", 12, "start",
                     cls="muted"))
    css = ("@keyframes pulse { 0%, 100% { stroke-width: 2; } 50% { stroke-width: 4; } }\n"
           "  .hl { animation: pulse 1.6s ease-in-out infinite; }")
    return svg(x0 * 2 + 48 * cell, 210, "\n".join(body), title, css)


def chw_memory():
    order = [(c, h, w) for c in "RGB" for h in range(4) for w in range(4)]
    hl = [c * 16 + 1 * 4 + 2 for c in range(3)]
    return memory_strip(order, "CHW (canaux d'abord) : 3 plans de 16 valeurs",
                        "index(c, h, w) = c·16 + h·4 + w      (16 = H·W)",
                        hl, "Les 3 valeurs d'un pixel sont espacées de 16 (pas d'un plan).")


def hwc_memory():
    order = [(c, h, w) for h in range(4) for w in range(4) for c in "RGB"]
    hl = [(1 * 4 + 2) * 3 + c for c in range(3)]
    return memory_strip(order, "HWC (canaux en dernier) : triplets RGB entrelacés",
                        "index(h, w, c) = (h·4 + w)·3 + c      (3 = C)",
                        hl, "Les 3 valeurs d'un pixel sont contiguës.")


# ─── HW methodology (SEI slides 9-16) mapped to VELA ─────────────────────────

METHODOLOGY = [
    ("Référence algorithmique", "done",
     ["Golden model Python : python/cnn.py", "poids CNN_coeff_3x3.txt, CIFAR-10",
      "précision 74,46 % (10 000 images)"]),
    ("Algorithme en virgule fixe", "done",
     ["Référence C++ double : hls/cnn_ref.cpp", "plages de valeurs → types ap_fixed (TN004)",
      "hls/cnn_hw.cpp + cnn_weights.h (16 bits)"]),
    ("Virgule fixe HLS synthétisable", "done",
     ["C-simulation Vitis HLS 2024.2", "+ g++ / pytest : bit-exact vs modèle TN004",
      "99,99 % d'accord avec le golden model"]),
    ("Vérification vs fichiers de référence", "done",
     ["Co-simulation C/RTL (xsim)", "tb_images.bin → RTL → comparé à tb_expected.bin",
      "3/3 bit-exact, latence 5,98 ms"]),
    ("Émulation avec entrées de référence", "phase3",
     ["IP sur la carte, images de référence", "chargées en mémoire, sortie comparée",
      "aux résultats attendus"]),
    ("Paramétrage du HW par SW", "phase3",
     ["ARM (PS) ↔ IP en AXI4-Lite (TN005)", "image @0x1000, ap_start, ap_done,",
      "classe @0x10 : IP prête, logiciel à écrire"]),
    ("Validation sur émulateur", "phase3",
     ["Carte SD → ARM (prétraitement)", "→ IP CNN → OLED SSD1331 96x64",
      "(au lieu de la sortie vidéo HDMI)"]),
    ("Système avec entrée/sortie vidéo", "future",
     ["Caméra ESP32-CAM (switch SW1)", "→ filtres / CNN → OLED",
      "flux continu : interface AXI-Stream"]),
]


def hw_methodology():
    cw, ch, gx, gy, x0, y0 = 520, 112, 30, 22, 30, 92
    body = [text(30, 36, "Étapes de la méthodologie HW appliquées à VELA", 20, "start", "bold"),
            text(30, 62, "D'après le cours SEI « Projet d'intégration des SoC » : chaque étape "
                 "est validée contre la précédente avant de passer à la suivante.", 13, "start",
                 cls="muted")]
    n = len(METHODOLOGY)
    for k, (title, status, lines) in enumerate(METHODOLOGY):
        col, row = k // 4, k % 4
        x, y = x0 + col * (cw + gx), y0 + row * (ch + gy)
        fill, stroke, label = STATUS[status]
        dash = 'stroke-dasharray="6 4"' if status == "future" else ""
        g = [rect(x, y, cw, ch, fill, stroke, 1.6, 8, dash),
             f'<circle cx="{x + 28}" cy="{y + 30}" r="16" fill="{stroke}"/>',
             text(x + 28, y + 35, str(k + 1), 15, weight="bold", fill=PAPER),
             text(x + 56, y + 35, title, 15, "start", "bold"),
             rect(x + cw - 92, y + 16, 78, 22, PAPER, stroke, 1.2, 11),
             text(x + cw - 53, y + 31, label, 11, weight="bold", fill=stroke)]
        for i, s in enumerate(lines):
            g.append(text(x + 56, y + 60 + i * 17, s, 12.5, "start",
                          cls="mono" if i == 0 and "/" in s else ""))
        body.append(f'<g class="step" style="animation-delay:{k * 2.5}s">' + "".join(g) + "</g>")
        if row < 3:
            body.append(line(x + 28, y + ch, x + 28, y + ch + gy - 2, MUTED, 1.5))
    # arrow from step 4 to step 5 (column change)
    x4, y4 = x0 + 28, y0 + 3 * (ch + gy) + ch
    x5, y5 = x0 + cw + gx + 28, y0
    body.append(path(f"M{x0 + cw} {y4 - ch / 2} C {x0 + cw + gx} {y4 - ch / 2}, "
                     f"{x5 - 50} {y5 + 30}, {x5 - 28} {y5 + 30}", MUTED, 1.5))
    yl = y0 + 4 * (ch + gy) + 6
    for i, (key, cap) in enumerate([("done", "fait et vérifié"), ("phase3", "phase 3 : intégration"),
                                    ("future", "extension future")]):
        fill, stroke, label = STATUS[key]
        body.append(rect(30 + i * 230, yl, 22, 16, fill, stroke, 1.4, 4))
        body.append(text(60 + i * 230, yl + 13, f"{label} : {cap}", 12, "start", cls="muted"))
    css = sequence_css("step", n, n * 2.5, 0.35)
    return svg(x0 * 2 + 2 * cw + gx, yl + 36, "\n".join(body),
               "Étapes de la méthodologie HW appliquées à VELA", css)


# ─── Prototyping platform (SEI slide 48) adapted to VELA ─────────────────────

def vela_platform():
    W, H = 1140, 712
    done_f, done_s, _ = STATUS["done"]
    ph_f, ph_s, _ = STATUS["phase3"]
    fu_f, fu_s, _ = STATUS["future"]
    dashed = 'stroke-dasharray="5 3"'
    b = [text(30, 36, "Plateforme de prototypage VELA (phase 3, TN007)", 20, "start", "bold"),
         text(30, 60, "Arty Z7-20 : Zynq XC7Z020. Blocs pleins = existants et vérifiés ; "
              "pointillés = à concevoir en phase 3 ou plus tard.", 13, "start", cls="muted")]
    # Zynq outline
    b.append(rect(150, 80, 830, 500, "#fafbfb", INK, 2, 12))
    b.append(text(965, 104, "Zynq XC7Z020", 14, "end", "bold"))
    # PS
    b.append(rect(170, 96, 560, 92, "#eef3f8", "#3b6fd1", 1.5, 8))
    b.append(text(190, 120, "Processing System (PS)", 13, "start", "bold", fill="#3b6fd1"))
    for x, w, title, sub in ((190, 170, "ARM Cortex-A9", "prétraitement, softmax, texte"),
                             (380, 150, "SD0 · UART", "images, console"),
                             (550, 160, "FCLK_CLK0", "100 MHz (NFR-001)")):
        b.append(rect(x, 132, w, 44, PAPER, INK, 1.2, 6))
        b.append(text(x + w / 2, 152, title, 13, weight="bold"))
        b.append(text(x + w / 2, 168, sub, 10.5, cls="muted"))
    # AXI bus
    b.append(rect(170, 208, 790, 34, "#f3efe9", ACCENT, 1.5, 17))
    b.append(text(565, 230, "M_AXI_GP0 → SmartConnect (AXI4-Lite) : décodage d'adresses, 5 esclaves",
                  13, weight="bold", fill=ACCENT))
    b.append(line(275, 176, 275, 206, ACCENT, 2, False))
    # CNN IP: receives the preprocessed image from the ARM, never reads the BRAMs
    b.append(rect(170, 270, 230, 130, done_f, done_s, 1.8, 10))
    b.append(text(285, 293, "IP CNN (Vitis HLS)", 14, weight="bold"))
    for i, s in enumerate(["reçoit de l'ARM l'image", "prétraitée 24×24 (@0x1000)",
                           "classe @0x10, logits @0x20", "5,98 ms · 22 DSP · 41 BRAM"]):
        b.append(text(285, 314 + i * 18, s, 11.5))
    b.append(line(285, 242, 285, 268, ACCENT, 2, "A"))
    b.append(text(291, 262, "image 24×24", 10, "start", fill=ACCENT))
    # AXI GPIO (buttons, switches, LEDs, filter engine control)
    b.append(rect(170, 430, 230, 66, ph_f, ph_s, 1.4, 8, dashed))
    b.append(text(285, 452, "AXI GPIO ×2", 12.5, weight="bold"))
    b.append(text(285, 469, "BTN · SW · LED", 10.5, cls="muted"))
    b.append(text(285, 484, "start · mode · busy · done", 10.5, cls="muted"))
    b.append(path("M410 242 L410 452 L402 452", MUTED, 1.4))
    b.append(line(400, 470, 438, 470, INK, 1.4))
    # BRAMs (port A: ARM through AXI BRAM controllers, port B: vela_pl)
    for x, title, sub, addr in ((470, "BRAM source", "64×64 RGB888", "@0x4200_0000"),
                                (780, "BRAM d'affichage", "image 64×64 | texte 32×64", "@0x4202_0000")):
        b.append(rect(x, 270, 180, 62, ph_f, ph_s, 1.4, 8, dashed))
        b.append(text(x + 90, 291, title, 12.5, weight="bold"))
        b.append(text(x + 90, 307, sub, 10.5, cls="muted"))
        b.append(text(x + 90, 322, addr, 10, cls="mono muted"))
    b.append(line(560, 242, 560, 268, ACCENT, 2, "A"))
    b.append(text(566, 262, "image ×2", 10, "start", fill=ACCENT))
    b.append(line(870, 242, 870, 268, ACCENT, 2, "A"))
    b.append(text(864, 262, "texte seul → zone texte", 10, "end", fill=ACCENT))
    # vela_pl (RTL)
    b.append(rect(425, 365, 545, 195, "#fbfbfb", INK, 1.2, 10, 'stroke-dasharray="6 4"'))
    b.append(text(440, 384, "vela_pl (RTL Verilog, module reference)", 12.5, "start", "bold"))
    b.append(rect(440, 398, 320, 142, ph_f, ph_s, 1.4, 8, dashed))
    b.append(text(600, 418, "filter_engine", 13, weight="bold"))
    for i, s in enumerate(["0 copie · 1 grayscale · 2 grayscale → threshold",
                           "3 gaussian ×3 (R, G, B) · 4 grayscale → sobel",
                           "5 median ×3 (R, G, B)   puis → RGB565"]):
        b.append(text(600, 440 + i * 18, s, 11))
    b.append(text(600, 506, "filtres existants (TN003) instanciés tels quels", 10.5, cls="muted"))
    b.append(text(600, 522, "image 64×64 en ≈ 41 µs → zone image seule", 10.5, cls="muted"))
    b.append(rect(780, 398, 180, 52, ph_f, ph_s, 1.4, 8, dashed))
    b.append(text(870, 420, "display_reader", 12.5, weight="bold"))
    b.append(text(870, 437, "lecture en boucle", 10.5, cls="muted"))
    b.append(rect(780, 478, 180, 62, done_f, done_s, 1.4, 8))
    b.append(text(870, 502, "oled_controller", 12.5, weight="bold"))
    b.append(text(870, 519, "spi_master, 6,25 MHz", 10.5, cls="muted"))
    b.append(line(560, 332, 560, 396, INK, 1.6))
    b.append(path("M740 398 L740 355 L830 355 L830 334", INK, 1.6))
    b.append(text(836, 350, "écriture", 10, "start", cls="muted"))
    b.append(line(910, 332, 910, 396, INK, 1.6))
    b.append(line(870, 450, 870, 476, INK, 1.6))
    # External: SD card, buttons, OLED, camera
    b.append(rect(20, 110, 110, 60, PAPER, INK, 1.4, 6))
    b.append(text(75, 136, "Carte SD", 13, weight="bold"))
    b.append(text(75, 154, "test_batch.bin", 10.5, cls="mono muted"))
    b.append(line(130, 140, 188, 152, INK, 1.6))
    b.append(rect(20, 420, 110, 72, PAPER, INK, 1.4, 6))
    b.append(text(75, 442, "BTN0-3", 12.5, weight="bold"))
    b.append(text(75, 460, "SW0-1", 12.5, weight="bold"))
    b.append(text(75, 478, "LED0-3", 12.5, weight="bold"))
    b.append(line(130, 456, 168, 456, INK, 1.4, False))
    b.append(rect(1000, 470, 120, 80, "#1f2a2e", INK, 1.5, 8))
    b.append(rect(1012, 482, 96, 56, "#2a3a3f", "#3a4a4f", 1, 3))
    b.append(text(1060, 514, "OLED 96x64", 11, fill=PAPER, weight="bold"))
    b.append(text(1060, 568, "Pmod OLEDrgb", 11.5, cls="muted"))
    b.append(text(1060, 583, "SSD1331, connecteur JA", 10.5, cls="muted"))
    b.append(line(960, 510, 998, 510, INK, 2))
    b.append(rect(1000, 380, 120, 70, fu_f, fu_s, 1.4, 8, 'stroke-dasharray="6 4"'))
    b.append(text(1060, 406, "Caméra", 13, weight="bold", fill=MUTED))
    b.append(text(1060, 424, "ESP32-CAM", 11, cls="muted"))
    b.append(text(1060, 440, "(SW1, futur)", 10.5, cls="muted"))
    b.append(path("M1000 415 L972 415", fu_s, 1.6, dash="6 4"))
    # The two modes
    b.append(rect(20, 596, 1100, 74, ACCENT_SOFT, ACCENT, 1.2, 8))
    b.append(text(36, 617, "Mode CNN (SW0 = 1) : SD → ARM (crop 24×24, normalisation) → IP CNN → "
                  "classe → ARM → zone texte ; image ×2 → BRAM source → filter_engine (copie) "
                  "→ zone image.", 12, "start"))
    b.append(text(36, 637, "Mode filtres (SW0 = 0) : SD → ARM → BRAM source (image ×2) → "
                  "filter_engine (filtre) → zone image ; nom du filtre → zone texte. Un seul "
                  "écrivain par zone.", 12, "start"))
    b.append(text(36, 657, "×2 : chaque pixel de l'image CIFAR 32×32 devient un carré 2×2 → "
                  "64×64 à gauche de l'écran 96×64 ; les 32 colonnes de droite servent au texte.",
                  12, "start", cls="muted"))
    # Animated data flows: CNN path (copper), filter and display path (green)
    b.append(path("M130 140 L188 152 M275 176 L275 206 M285 242 L285 268 M870 242 L870 268",
                  ACCENT, 3.5, False, dash="2 10", extra='class="flow" opacity="0.85"'))
    b.append(path("M560 242 L560 268 M560 332 L560 396 M740 398 L740 355 L830 355 L830 334 "
                  "M910 332 L910 396 M870 450 L870 476 M960 510 L998 510",
                  "#3a9d5d", 3.5, False, dash="2 10", extra='class="flow" opacity="0.85"'))
    css = ("@keyframes flow { to { stroke-dashoffset: -48; } }\n"
           "  .flow { animation: flow 1.2s linear infinite; }")
    # Legend
    yl = 694
    for i, (key, cap) in enumerate([("done", "existant, vérifié"),
                                    ("phase3", "à concevoir en phase 3"),
                                    ("future", "extension future")]):
        fill, stroke, label = STATUS[key]
        b.append(rect(150 + i * 250, yl - 12, 22, 16, fill, stroke, 1.4, 4))
        b.append(text(180 + i * 250, yl + 1, cap, 12, "start", cls="muted"))
    b.append(path(f"M{900} {yl - 4} L{940} {yl - 4}", ACCENT, 3.5, False, dash="2 10",
                  extra='class="flow"'))
    b.append(text(948, yl + 1, "flux de données", 12, "start", cls="muted"))
    return svg(W, H, "\n".join(b), "Plateforme de prototypage VELA", css)


# ─── CNN data structures (SEI slides 26-37) with the VELA network ────────────

LAYERS = [
    # name, channels, size, caption, op to next
    ("Image", 3, 24, "3×24×24", "conv1 3×3, 64 noyaux + ReLU"),
    ("conv1", 64, 24, "64×24×24", "maxpool 3×3 / 2"),
    ("pool1", 64, 12, "64×12×12", "conv2 3×3, 32 noyaux + ReLU"),
    ("conv2", 32, 12, "32×12×12", "maxpool 3×3 / 2"),
    ("pool2", 32, 6, "32×6×6", "conv3 3×3, 20 noyaux + ReLU"),
    ("conv3", 20, 6, "20×6×6", "maxpool 3×3 / 2"),
    ("pool3", 20, 3, "20×3×3", "reshape HWC"),
]


def box3d(x, y, depth, size, fill, stroke):
    """Isometric box: front face size x size, depth along a 30° diagonal."""
    dx, dy = depth * 0.8, -depth * 0.5
    front = f"M{x} {y} h{size} v{size} h{-size} z"
    top = f"M{x} {y} l{dx} {dy} h{size} l{-dx} {-dy} z"
    side = f"M{x + size} {y} l{dx} {dy} v{size} l{-dx} {-dy} z"
    return (f'<path d="{top}" fill="{fill}" stroke="{stroke}" stroke-width="1" opacity="0.75"/>'
            f'<path d="{side}" fill="{fill}" stroke="{stroke}" stroke-width="1" opacity="0.6"/>'
            f'<path d="{front}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')


def op_badge(x, y, letter):
    """Small accent circle with a letter, referring to the operation legend."""
    return (f'<circle cx="{x}" cy="{y}" r="9" fill="{PAPER}" stroke="{ACCENT}" stroke-width="1.4"/>'
            + text(x, y + 4, letter, 11, weight="bold", fill=ACCENT))


def cnn_inference_chain():
    W, H = 1180, 470
    ops = [op for *_, op in LAYERS] + ["FC 180→10 (M·v + b)", "softmax"]
    legend = [f"{chr(97 + i)} : {op}" for i, op in enumerate(ops)]
    b = [text(30, 36, "Chaîne d'inférence du CNN VELA : chaque couche transforme une feature map",
              19, "start", "bold"),
         text(30, 60, "Feature map = tableau 3D (canaux × hauteur × largeur), stocké en CHW. "
              "Dimensions réelles du réseau CIFAR-10 de VELA.", 13, "start", cls="muted"),
         text(30, 88, "   ·   ".join(legend[:4]), 12, "start", fill=ACCENT),
         text(30, 108, "   ·   ".join(legend[4:]), 12, "start", fill=ACCENT)]
    base_y, x = 270, 40
    items = []
    for k, (name, c, s, cap, op) in enumerate(LAYERS):
        size = s * 5.2
        depth = 10 + c * 0.9
        y = base_y - size / 2
        stored = name.startswith("pool") or name == "Image"
        fill = "#e8eef8" if name == "Image" else ("#e7f1ea" if stored else "#f4f5f6")
        stroke = "#3b6fd1" if name == "Image" else ("#2f6f4e" if stored else "#9aa4a8")
        g = [box3d(x, y, depth, size, fill, stroke),
             text(x + size / 2 + depth * 0.4, base_y + max(size / 2, 20) + 26, name, 13, weight="bold"),
             text(x + size / 2 + depth * 0.4, base_y + max(size / 2, 20) + 44, cap, 12, cls="mono")]
        if not stored:
            g.append(text(x + size / 2 + depth * 0.4, base_y + max(size / 2, 20) + 61,
                          "non stockée dans l'IP", 10, cls="muted"))
        items.append(g)
        nx = x + size + depth * 0.8 + 30
        items[-1].append(line(nx - 26, base_y, nx - 6, base_y, ACCENT, 1.6, "A"))
        items[-1].append(op_badge(nx - 16, base_y - 16, "abcdefg"[k]))
        x = nx
    # vector 180, FC, logits, softmax
    vx = x
    g = [rect(vx, base_y - 95, 14, 190, "#e7f1ea", "#2f6f4e", 1.2, 2)]
    for i in range(0, 190, 10):
        g.append(line(vx, base_y - 95 + i, vx + 14, base_y - 95 + i, "#2f6f4e", 0.4, False))
    g += [text(vx + 7, base_y + 120, "vecteur", 13, weight="bold"),
          text(vx + 7, base_y + 138, "180", 12, cls="mono"),
          line(vx + 20, base_y, vx + 50, base_y, ACCENT, 1.6, "A"),
          op_badge(vx + 35, base_y - 16, "h")]
    items.append(g)
    lx = vx + 58
    g = [rect(lx, base_y - 50, 14, 100, "#fdf3e1", "#b7791f", 1.2, 2)]
    for i in range(0, 100, 10):
        g.append(line(lx, base_y - 50 + i, lx + 14, base_y - 50 + i, "#b7791f", 0.5, False))
    g += [text(lx + 7, base_y + 75, "logits", 13, weight="bold"),
          text(lx + 7, base_y + 93, "10", 12, cls="mono"),
          line(lx + 20, base_y, lx + 50, base_y, ACCENT, 1.6, "A"),
          op_badge(lx + 35, base_y - 16, "i")]
    items.append(g)
    cx = lx + 58
    items.append([rect(cx, base_y - 26, 86, 52, ACCENT_SOFT, ACCENT, 1.6, 8),
                  text(cx + 43, base_y - 4, "classe", 12, cls="muted"),
                  text(cx + 43, base_y + 15, "« cat »", 14, weight="bold")])
    n = len(items)
    for k, g in enumerate(items):
        b.append(f'<g class="lay" style="animation-delay:{k * 1.3:.1f}s">' + "".join(g) + "</g>")
    b.append(text(30, 430, "MAC par image : conv1 1,0 M · conv2 2,65 M · conv3 0,21 M · FC 1 800  "
                  "(≈ 3,9 M).  Dans l'IP, conv + ReLU + maxpool sont fusionnés : seules les "
                  "fmaps « pool » sont mémorisées (en vert).", 12, "start", cls="muted"))
    b.append(text(30, 450, "La mise à plat suit l'ordre HWC (ordre d'apprentissage des poids "
                  "FC), voir TN001.", 12, "start", cls="muted"))
    css = sequence_css("lay", n, n * 1.3, 0.3)
    return svg(W, H, "\n".join(b), "Chaîne d'inférence du CNN VELA", css)


def cnn_convolution():
    """Two kernels K0, K1 slide over a 3-channel 6x6 input (zero padding) -> 2 output channels."""
    W, H, cell, n = 1100, 560, 30, 6
    ix, iy = 70, 170                # front (R) plane top-left
    off = 18                        # offset between stacked planes
    ox, oy = 760, 170               # output plane top-left
    period = 18.0                   # 36 positions per kernel, 2 kernels
    steps = n * n
    b = [text(30, 36, "Convolution multi-canal (3×3, padding « same »)", 19, "start", "bold"),
         text(30, 62, "S[c, i, j] = Σ_l Σ_m Σ_n  I[l, i+m−1, j+n−1] · K[c, l, m, n]  +  bias[c]",
              14, "start", cls="mono"),
         text(30, 84, "Chaque sortie combine les 3×3 voisins de TOUS les canaux d'entrée ; chaque "
              "noyau K_c produit un canal de sortie.", 13, "start", cls="muted")]
    # input planes B, G, R (back to front) with padding ring
    for k, c in enumerate("BGR"):
        px, py = ix + (2 - k) * off, iy - (2 - k) * off
        b.append(rect(px - cell, py - cell, (n + 2) * cell, (n + 2) * cell, "none", LINE, 1, 0,
                      'stroke-dasharray="4 3"'))
        for r in range(n):
            for q in range(n):
                b.append(rect(px + q * cell, py + r * cell, cell, cell, CH_SOFT[c], PAPER, 1))
    b.append(text(ix + n * cell / 2, iy + n * cell + 50, "I : entrée 3 canaux (R, G, B), 6×6",
                  13, weight="bold"))
    b.append(text(ix + n * cell / 2, iy + n * cell + 68, "pointillés : bord à 0 (padding)", 11.5,
                  cls="muted"))
    # sliding window: a 3x3 frame on each plane, moved by keyframes
    pos = [(r, q) for r in range(n) for q in range(n)]
    kf = []
    for i, (r, q) in enumerate(pos):
        p0, p1 = 100 * i / (2 * steps), 100 * (i + 1) / (2 * steps) - 0.01
        tf = f"translate({(q - 1) * cell}px, {(r - 1) * cell}px)"
        kf.append(f"{p0:.3f}%, {p1:.3f}% {{ transform: {tf}; }}")
    for i, (r, q) in enumerate(pos):
        p0, p1 = 50 + 100 * i / (2 * steps), 50 + 100 * (i + 1) / (2 * steps) - 0.01
        tf = f"translate({(q - 1) * cell}px, {(r - 1) * cell}px)"
        kf.append(f"{p0:.3f}%, {p1:.3f}% {{ transform: {tf}; }}")
    css = ["@keyframes slide { " + " ".join(kf) + " }",
           f".win {{ animation: slide {period}s infinite; animation-timing-function: step-end; }}"]
    for k, c in enumerate("BGR"):
        px, py = ix + (2 - k) * off, iy - (2 - k) * off
        b.append(f'<g transform="translate({px},{py})"><g class="win">'
                 + rect(0, 0, 3 * cell, 3 * cell, CH[c], INK, 2, 0, 'fill-opacity="0.55"')
                 + "</g></g>")
    # kernels
    kx, ky = 420, 150
    for kk in range(2):
        for k, c in enumerate("BGR"):
            px, py = kx + (2 - k) * 10, ky + kk * 150 - (2 - k) * 10
            for r in range(3):
                for q in range(3):
                    b.append(rect(px + q * 18, py + r * 18, 18, 18, CH_SOFT[c], CH[c], 0.8))
        b.append(text(kx + 30, ky + kk * 150 + 76, f"K{kk}", 15, weight="bold"))
        b.append(text(kx + 30, ky + kk * 150 + 94, "3 tranches 3×3", 11, cls="muted"))
    # sigma and bias
    for kk in range(2):
        sx, sy = 560, 200 + kk * 150
        b.append(f'<circle cx="{sx}" cy="{sy}" r="20" fill="{PAPER}" stroke="{ACCENT}" stroke-width="1.6"/>')
        b.append(text(sx, sy + 7, "Σ", 20, weight="bold", fill=ACCENT))
        b.append(text(sx + 52, sy - 26, f"+ bias[{kk}]", 12, fill=ACCENT))
        b.append(line(kx + 70, sy - 8, sx - 22, sy, ACCENT, 1.4, "A"))
        b.append(line(sx + 22, sy, ox - 8 + kk * 18, sy, ACCENT, 1.4, "A"))
    b.append(line(ix + 3 * cell, iy + cell, 536, 196, ACCENT, 1.2, "A", dash="5 4"))
    # output planes (S1 behind, S0 front), cells fill progressively
    for kk in (1, 0):
        px, py = ox + kk * 18, oy - kk * 18
        for r in range(n):
            for q in range(n):
                i = r * n + q
                b.append(rect(px + q * cell, py + r * cell, cell, cell, "#f1f3f4", PAPER, 1))
                start = (kk * steps + i) / (2 * steps) * 100
                b.append(rect(px + q * cell, py + r * cell, cell, cell,
                              "#c46a3a" if kk == 0 else "#8a4a2a", PAPER, 1, 0,
                              f'class="o{kk}_{i}" opacity="0"'))
                css.append(f"@keyframes f{kk}_{i} {{ 0%, {start:.3f}% {{ opacity: 0; }} "
                           f"{start + 0.01:.3f}%, 99.9% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}"
                           f" .o{kk}_{i} {{ animation: f{kk}_{i} {period}s infinite; }}")
    b.append(text(ox + n * cell / 2 + 9, iy + n * cell + 50, "S : 2 canaux de sortie, 6×6", 13,
                  weight="bold"))
    b.append(text(ox + n * cell / 2 + 9, iy + n * cell + 68, "S0 avec K0, puis S1 avec K1", 11.5,
                  cls="muted"))
    b.append(rect(30, 480, 1040, 56, ACCENT_SOFT, ACCENT, 1.2, 8))
    b.append(text(46, 503, "Dans l'IP VELA (TN005) : U = 8 noyaux traités en parallèle → chaque "
                  "valeur d'entrée lue une fois sert 8 MAC (8 DSP) ; conv1 : 64 noyaux sur 3 "
                  "canaux 24×24.", 12.5, "start"))
    b.append(text(46, 523, "Accumulateur exact ap_fixed<40,14>, résultat arrondi en 16 bits après "
                  "la somme (TN004).", 12.5, "start"))
    return svg(W, H, "\n".join(b), "Convolution multi-canal 3x3", "\n  ".join(css))


def cnn_maxpool():
    W, H, cell = 980, 580, 44
    rng = np.random.default_rng(7)
    vals = rng.integers(0, 10, size=(6, 6))
    ix, iy, ox, oy = 70, 120, 690, 164
    period = 13.5
    b = [text(30, 36, "MaxPool 3×3, pas 2, padding « SAME » (TensorFlow) : 6×6 → 3×3", 19,
              "start", "bold"),
         text(30, 62, "Chaque sortie = maximum d'une fenêtre 3×3 ; les fenêtres se recouvrent "
              "(pas 2 < taille 3).", 12.5, "start", cls="muted"),
         text(30, 81, "La ligne et la colonne manquantes sont ajoutées en bas / à droite "
              "et ignorées (–∞).", 12.5, "start", cls="muted")]
    # input with padding row/col
    for r in range(7):
        for q in range(7):
            pad = r == 6 or q == 6
            x, y = ix + q * cell, iy + r * cell
            if pad:
                b.append(rect(x, y, cell, cell, "#f7f8f8", LINE, 1, 0, 'stroke-dasharray="4 3"'))
                b.append(text(x + cell / 2, y + 27, "–∞", 13, cls="muted"))
            else:
                b.append(rect(x, y, cell, cell, "#eef3f8", PAPER, 1))
                b.append(text(x + cell / 2, y + 28, str(vals[r, q]), 16, cls="mono"))
    b.append(text(ix + 3 * cell, iy + 7 * cell + 30, "entrée (après ReLU), 6×6 + bordure SAME",
                  13, weight="bold"))
    # windows
    out = np.zeros((3, 3), dtype=int)
    kf, css = [], []
    for k in range(9):
        pi, pj = divmod(k, 3)
        win = vals[2 * pi:2 * pi + 3, 2 * pj:2 * pj + 3]
        out[pi, pj] = win.max()
        p0, p1 = 100 * k / 9, 100 * (k + 1) / 9 - 0.01
        kf.append(f"{p0:.3f}%, {p1:.3f}% {{ transform: translate({2 * pj * cell}px, {2 * pi * cell}px); }}")
    css.append("@keyframes pool { " + " ".join(kf) + " }")
    css.append(f".pwin {{ animation: pool {period}s infinite step-end; }}")
    b.append(f'<g transform="translate({ix},{iy})"><g class="pwin">'
             + rect(0, 0, 3 * cell, 3 * cell, ACCENT, ACCENT, 3, 4, 'fill-opacity="0.12"')
             + "</g></g>")
    # max markers per window (cell of the max), shown in turn
    for k in range(9):
        pi, pj = divmod(k, 3)
        win = vals[2 * pi:2 * pi + 3, 2 * pj:2 * pj + 3]
        r, q = np.unravel_index(win.argmax(), win.shape)
        x, y = ix + (2 * pj + q) * cell, iy + (2 * pi + r) * cell
        b.append(rect(x + 3, y + 3, cell - 6, cell - 6, "none", ACCENT, 2.5, 6,
                      f'class="mx{k}" opacity="0"'))
        p0, p1 = 100 * k / 9, 100 * (k + 1) / 9
        css.append(f"@keyframes mx{k} {{ 0%, {p0:.3f}% {{ opacity: 0; }} {p0 + 0.01:.3f}%, "
                   f"{p1 - 0.01:.3f}% {{ opacity: 1; }} {p1:.3f}%, 100% {{ opacity: 0; }} }} "
                   f".mx{k} {{ animation: mx{k} {period}s infinite; }}")
    # output
    for k in range(9):
        pi, pj = divmod(k, 3)
        x, y = ox + pj * cell * 1.2, oy + pi * cell * 1.2
        b.append(rect(x, y, cell * 1.2, cell * 1.2, "#f1f3f4", PAPER, 1))
        p0 = 100 * k / 9
        g = (rect(x, y, cell * 1.2, cell * 1.2, ACCENT_SOFT, ACCENT, 1.4)
             + text(x + cell * 0.6, y + cell * 0.75, str(out[pi, pj]), 18, weight="bold",
                    cls="mono"))
        b.append(f'<g class="po{k}">{g}</g>')
        css.append(f"@keyframes po{k} {{ 0%, {p0:.3f}% {{ opacity: 0.15; }} {p0 + 0.01:.3f}%, "
                   f"99.9% {{ opacity: 1; }} 100% {{ opacity: 0.15; }} }} "
                   f".po{k} {{ animation: po{k} {period}s infinite; }}")
    b.append(text(ox + 1.8 * cell, oy + 3.6 * cell + 34, "sortie 3×3", 13, weight="bold"))
    b.append(line(ix + 7 * cell + 20, iy + 3.5 * cell, ox - 16, iy + 3.5 * cell, ACCENT, 1.6, "A"))
    b.append(text((ix + 7 * cell + ox) / 2, iy + 3.5 * cell - 10, "max(fenêtre)", 12, fill=ACCENT))
    b.append(rect(30, 506, 920, 52, ACCENT_SOFT, ACCENT, 1.2, 8))
    b.append(text(46, 528, "Dans l'IP VELA : maxpool fusionné à la convolution — chaque sortie de "
                  "conv met à jour les 1, 2 ou 4 cases qui la contiennent ;", 12.5, "start"))
    b.append(text(46, 548, "le tampon est remis à 0 avant la couche, ce qui fait aussi la ReLU "
                  "(TN005 §2).", 12.5, "start"))
    return svg(W, H, "\n".join(b), "MaxPool 3x3 pas 2 padding SAME", "\n  ".join(css))


def softmax_example():
    """Real probabilities of CIFAR-10 test image 0 with the golden model, or a fallback."""
    try:
        sys.path.insert(0, str(ROOT / "python"))
        from cnn import cnn_forward
        from data_loader import CIFAR10_CLASSES, load_dataset
        from weights_loader import load_weights
        imgs, _ = load_dataset(str(ROOT / "data" / "cifar-10-batches-bin"), max_images=1)
        probs, _ = cnn_forward(imgs[0], load_weights(str(ROOT / "data" / "CNN_coeff_3x3.txt")))
        return list(CIFAR10_CLASSES), [float(p) for p in probs]
    except Exception:                                     # data not available
        classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
        return classes, [0.004, 0.003, 0.013, 0.738, 0.010, 0.116, 0.052, 0.008, 0.053, 0.003]


def cnn_fc_softmax():
    W, H = 1180, 520
    classes, probs = softmax_example()
    b = [text(30, 36, "Mise à plat (HWC), couche dense et softmax", 19, "start", "bold"),
         text(30, 62, "pool3 (20×3×3) → vecteur de 180 valeurs dans l'ordre (h, w, c) → "
              "S = M·v + b (10 logits) → softmax → probabilités", 13, "start", cls="muted")]
    # pool3: 9 positions (h,w), each a column of 20 channels -> show 3x3 grid of small stacks
    gx, gy, cell = 60, 120, 34
    for h in range(3):
        for w in range(3):
            x, y = gx + w * (cell + 10), gy + h * (cell + 10)
            k = h * 3 + w
            b.append(f'<g class="pos" style="animation-delay:{k * 0.6:.1f}s">'
                     + rect(x, y, cell, cell, "#e7f1ea", "#2f6f4e", 1.4, 3)
                     + text(x + cell / 2, y + 15, f"({h},{w})", 10, cls="mono")
                     + text(x + cell / 2, y + 28, "20 c", 9.5, cls="muted") + "</g>")
    b.append(text(gx + 66, gy + 150, "pool3 : 3×3 positions", 13, weight="bold"))
    b.append(text(gx + 66, gy + 168, "× 20 canaux", 12, cls="muted"))
    # flattened vector: 9 segments of 20
    vx, vy, seg_h = 300, 100, 30
    b.append(line(gx + 140, gy + 60, vx - 12, gy + 60, ACCENT, 1.6, "A"))
    b.append(text((gx + 140 + vx) / 2, gy + 50, "reshape", 12, fill=ACCENT))
    for k in range(9):
        h, w = divmod(k, 3)
        y = vy + k * seg_h
        b.append(f'<g class="pos" style="animation-delay:{k * 0.6:.1f}s">'
                 + rect(vx, y, 26, seg_h - 2, "#e7f1ea", "#2f6f4e", 1.2, 2)
                 + text(vx + 34, y + 19, f"({h},{w}) c=0..19  → v[{k * 20}..{k * 20 + 19}]", 11,
                        "start", cls="mono") + "</g>")
    b.append(text(vx + 13, vy + 9 * seg_h + 24, "v : 180", 13, weight="bold"))
    b.append(text(vx + 13, vy + 9 * seg_h + 42, "index = (h·3 + w)·20 + c", 11.5, cls="mono muted"))
    # FC
    fx = 620
    b.append(rect(fx, 160, 120, 130, ACCENT_SOFT, ACCENT, 1.6, 8))
    b.append(text(fx + 60, 200, "S = M·v + b", 14, weight="bold"))
    b.append(text(fx + 60, 224, "M : 10×180", 12, cls="mono"))
    b.append(text(fx + 60, 244, "1 800 MAC", 12, cls="muted"))
    b.append(text(fx + 60, 264, "(+ 10 biais)", 12, cls="muted"))
    b.append(line(540, 225, fx - 8, 225, ACCENT, 1.6, "A"))
    b.append(line(fx + 120, 225, 790, 225, ACCENT, 1.6, "A"))
    b.append(text(765, 214, "logits", 11, fill=ACCENT))
    # softmax bars
    bx, by, bw = 800, 112, 200
    pred = int(np.argmax(probs))
    b.append(text(bx, by - 16, "softmax(z)_i = exp(z_i) / Σ_j exp(z_j)", 12, "start", cls="mono"))
    css = ["@keyframes grow { 0%, 45% { transform: scaleX(0); } 70%, 100% { transform: scaleX(1); } }"]
    for i, (c, p) in enumerate(zip(classes, probs)):
        y = by + i * 26
        col = ACCENT if i == pred else "#9aa4a8"
        b.append(text(bx + 70, y + 15, c, 11.5, "end", weight="bold" if i == pred else "normal"))
        b.append(rect(bx + 78, y + 3, bw, 16, "#f1f3f4", "none", 0, 3))
        b.append(f'<rect x="{bx + 78}" y="{y + 3}" width="{max(p * bw, 1.5):.1f}" height="16" '
                 f'rx="3" fill="{col}" class="bar" style="transform-origin:{bx + 78}px 0"/>')
        b.append(text(bx + 84 + bw, y + 15, f"{100 * p:.1f} %", 11, "start",
                      weight="bold" if i == pred else "normal", cls="mono"))
    css.append(".bar { animation: grow 9s ease-out infinite; transform-box: view-box; }")
    css.append(sequence_css("pos", 9, 5.4 * 1.0, 0.35))
    b.append(rect(30, 430, 1120, 70, ACCENT_SOFT, ACCENT, 1.2, 8))
    b.append(text(46, 454, f"Exemple réel : image de test n°0 (golden model) → « {classes[pred]} » "
                  f"à {100 * probs[pred]:.1f} %.", 12.5, "start"))
    b.append(text(46, 474, "Dans VELA, l'IP calcule les logits et la classe (argmax, exact) ; "
                  "l'ARM calcule la probabilité affichée sur l'OLED (TN004 §6, TN005 §5).",
                  12.5, "start"))
    b.append(text(46, 492, "L'ordre HWC de la mise à plat est imposé par l'ordre d'apprentissage "
                  "des poids M (TN001).", 12.5, "start"))
    return svg(W, H, "\n".join(b), "Mise à plat HWC, couche dense et softmax", "\n  ".join(css))


DIAGRAMS = {
    "pixel_grid.svg": pixel_grid,
    "chw_memory.svg": chw_memory,
    "hwc_memory.svg": hwc_memory,
    "hw_methodology.svg": hw_methodology,
    "vela_platform.svg": vela_platform,
    "cnn_inference_chain.svg": cnn_inference_chain,
    "cnn_convolution.svg": cnn_convolution,
    "cnn_maxpool.svg": cnn_maxpool,
    "cnn_fc_softmax.svg": cnn_fc_softmax,
}


def main(out_dir=ROOT / "docs" / "diagrams"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in DIAGRAMS.items():
        (out_dir / name).write_text(fn())
        print(f"Wrote {out_dir / name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "diagrams")
