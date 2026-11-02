"""
test_cnn_ip_interface.py — Data contract between the PS software and the CNN IP

Ce que le logiciel de l'ARM écrit dans l'IP (fenêtre image à 0x1000) et relit
(fenêtre logits à 0x20), d'après la carte des registres générée par Vitis HLS
(VELA-TN-005). Ces fonctions de référence serviront à valider le code C de
la phase 3.
"""

import numpy as np
from vela_testing import requirement
from cnn_hw_model import (TN004_FORMATS, input_raw, pack_image_words, quantize,
                          unpack_logit_words)


@requirement("VELA-NFR-006")
def test_input_raw_matches_input_t():
    """Code brut / 2^11 = valeur input_t (arrondi au demi supérieur, saturation)."""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 3, 5000), [0.5 / 2048, -0.5 / 2048, 1.5 / 2048,
                                                 15.9999, 16.0, 20.0, -16.0, -20.0]])
    np.testing.assert_array_equal(input_raw(x) / 2048.0, quantize(x, 16, 5, "rnd"))


@requirement("VELA-NFR-006")
def test_input_raw_saturation_bounds():
    """Saturation : +16 et au-delà -> 32767 ; -16 et en deçà -> -32768."""
    assert input_raw(np.array([16.0, 100.0])).tolist() == [32767, 32767]
    assert input_raw(np.array([-16.0, -100.0])).tolist() == [-32768, -32768]


@requirement("VELA-NFR-006")
def test_pack_image_words_layout():
    """864 mots ; élément 2n dans les bits 15:0, 2n+1 dans les bits 31:16, ordre CHW."""
    img = np.zeros((3, 24, 24))
    img[0, 0, 0], img[0, 0, 1] = 1.0, -1.0          # elements 0 and 1
    img[2, 23, 23] = 0.5                              # last element (1727)
    words = pack_image_words(img)
    assert words.shape == (864,)
    assert words[0] == (0xF800 << 16) | 0x0800        # -1.0 -> 0xF800, 1.0 -> 0x0800
    assert words[863] == (0x0400 << 16)               # 0.5 -> 0x0400 in the high half


@requirement("VELA-NFR-006")
def test_unpack_logit_words():
    """5 mots -> 10 logits logit_t (2^-9 près), signe compris."""
    width, ibits = TN004_FORMATS["logit"]
    logits = np.array([1.5, -2.25, 0, 63.998046875, -64, 0.001953125, 3, -3, 10, -0.5])
    raw = np.round(logits * 2 ** (width - ibits)).astype(np.int16).view(np.uint16).astype(np.uint32)
    words = raw[0::2] | (raw[1::2] << 16)
    np.testing.assert_array_equal(unpack_logit_words(words), logits)
