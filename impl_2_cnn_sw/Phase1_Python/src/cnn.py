"""
cnn.py — Implémentation des couches CNN en NumPy

Toutes les fonctions travaillent en format CHW (Canaux, Hauteur, Largeur),
cohérent avec le format CIFAR-10 et avec l'implémentation C++ HLS à venir.

Architecture CNN basique CIFAR-10 :
  Image 24x24x3 (format CHW : 3x24x24)
    → Conv(3x3, 64) + RELU  → 24x24x64
    → MaxPool(3x3, stride=2) → 12x12x64
    → Conv(3x3, 32) + RELU  → 12x12x32
    → MaxPool(3x3, stride=2) → 6x6x32
    → Conv(3x3, 20) + RELU  → 6x6x20
    → MaxPool(3x3, stride=2) → 3x3x20
    → Reshape                → 180
    → FC(180→10)             → 10
    → Softmax                → probabilités
"""

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Couche 1 : Convolution 2D multi-canal
# ────────────────────────────────────────────────────────────────────────────

def conv2d(input_fmap, weights, biases, padding="same"):
    """
    Convolution 2D multi-canal.

    Args:
        input_fmap : np.ndarray (C_in, H, W)            — feature map d'entrée
        weights    : np.ndarray (C_out, C_in, kH, kW)   — kernels
        biases     : np.ndarray (C_out,)                — biais
        padding    : "same" conserve H et W, "valid" les réduit

    Retourne :
        output_fmap : np.ndarray (C_out, H_out, W_out) float32

    Formule :
        S[c_out, i, j] = Σ_{l,m,n} I[l, i+m, j+n] x K[c_out, l, m, n]
                         + bias[c_out]
    """
    c_in, h_in, w_in   = input_fmap.shape
    c_out, _, kh, kw   = weights.shape

    # Padding pour conserver les dimensions (mode "same")
    if padding == "same":
        pad_h = kh // 2
        pad_w = kw // 2
        input_padded = np.pad(
            input_fmap,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant", constant_values=0
        )
        h_out, w_out = h_in, w_in
    else:  # "valid" : pas de padding
        input_padded = input_fmap
        h_out = h_in - kh + 1
        w_out = w_in - kw + 1

    output = np.zeros((c_out, h_out, w_out), dtype=np.float32)

    # Boucles de convolution (lisibles, équivalentes au C++ HLS)
    for co in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                acc = biases[co]
                for ci in range(c_in):
                    for m in range(kh):
                        for n in range(kw):
                            acc += (input_padded[ci, i + m, j + n]
                                    * weights[co, ci, m, n])
                output[co, i, j] = acc

    return output


def conv2d_fast(input_fmap, weights, biases, padding="same"):
    """
    Version vectorisée de conv2d (même résultat, ~100x plus rapide).
    Utilisée en pratique ; conv2d() sert de référence pour le débogage.
    """
    c_in, h_in, w_in = input_fmap.shape
    c_out, _, kh, kw = weights.shape

    if padding == "same":
        pad_h, pad_w = kh // 2, kw // 2
        input_padded = np.pad(
            input_fmap,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant", constant_values=0
        )
        h_out, w_out = h_in, w_in
    else:
        input_padded = input_fmap
        h_out = h_in - kh + 1
        w_out = w_in - kw + 1

    # Extraction des patches via stride tricks
    # patches : (h_out, w_out, c_in, kh, kw)
    patches = np.lib.stride_tricks.as_strided(
        input_padded,
        shape=(h_out, w_out, c_in, kh, kw),
        strides=(
            input_padded.strides[1],
            input_padded.strides[2],
            input_padded.strides[0],
            input_padded.strides[1],
            input_padded.strides[2],
        )
    )
    # Aplatissement et multiplication matricielle
    # patches_flat : (h_out*w_out, c_in*kh*kw)
    patches_flat = patches.reshape(h_out * w_out, c_in * kh * kw)
    # weights_flat : (c_out, c_in*kh*kw)
    weights_flat = weights.reshape(c_out, c_in * kh * kw)
    # output_flat  : (h_out*w_out, c_out)
    output_flat  = patches_flat @ weights_flat.T + biases[np.newaxis, :]
    # Reshape → (c_out, h_out, w_out)
    output = output_flat.reshape(h_out, w_out, c_out).transpose(2, 0, 1)

    return output.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Couche 2 : RELU
# ────────────────────────────────────────────────────────────────────────────

def relu(fmap):
    """
    REctified Linear Unit : RELU(a) = max(0, a)
    Fonctionne sur n'importe quelle forme.
    """
    return np.maximum(0.0, fmap).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Couche 3 : MaxPool 2D
# ────────────────────────────────────────────────────────────────────────────

def maxpool2d(fmap, pool_size=3, stride=2):
    """
    MaxPool 2D multi-canal.

    Args:
        fmap      : np.ndarray (C, H, W)
        pool_size : taille de la fenêtre (carré, défaut=3)
        stride    : pas de déplacement (défaut=2)

    Retourne :
        np.ndarray (C, H_out, W_out)
        H_out = (H - pool_size) // stride + 1
    """
    c, h, w = fmap.shape
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1

    output = np.zeros((c, h_out, w_out), dtype=np.float32)

    for i in range(h_out):
        for j in range(w_out):
            i0, j0 = i * stride, j * stride
            window = fmap[:, i0:i0 + pool_size, j0:j0 + pool_size]
            output[:, i, j] = window.reshape(c, -1).max(axis=1)

    return output


# ────────────────────────────────────────────────────────────────────────────
# Couche 4 : Fully Connected (perceptron)
# ────────────────────────────────────────────────────────────────────────────

def fully_connected(fmap, weights, biases):
    """
    Couche entièrement connectée : S = W x I + b

    Args:
        fmap    : np.ndarray de forme quelconque — sera aplati en vecteur
        weights : np.ndarray (n_out, n_in)
        biases  : np.ndarray (n_out,)

    Retourne :
        np.ndarray (n_out,) float32
    """
    vec = fmap.flatten().astype(np.float32)
    return (weights @ vec + biases).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Couche 5 : SoftMax
# ────────────────────────────────────────────────────────────────────────────

def softmax(logits):
    """
    SoftMax numériquement stable :
        softmax(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
    La soustraction de max(z) évite les débordements float.
    """
    z = logits - logits.max()
    exp_z = np.exp(z)
    return (exp_z / exp_z.sum()).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Pipeline CNN complet
# ────────────────────────────────────────────────────────────────────────────

def cnn_forward(image, weights, verbose=False):
    """
    Passe avant complète du CNN CIFAR-10 basique.

    Args:
        image   : np.ndarray (3, 24, 24) float32 prétraitée
        weights : dict retourné par weights_loader.load_weights()
        verbose : affiche les formes intermédiaires si True

    Retourne :
        probs : np.ndarray (10,) — probabilités softmax
        pred  : int              — classe prédite (argmax)
    """
    def _log(name, fmap):
        if verbose:
            print(f"  {name:<25} {str(fmap.shape):<20} "
                  f"min={fmap.min():7.3f}  max={fmap.max():7.3f}")

    x = image.astype(np.float32)
    _log("Input", x)

    # Bloc 1 : Conv + RELU + MaxPool
    x = conv2d_fast(x, weights["conv1_w"], weights["conv1_b"])
    _log("Conv1 (avant RELU)", x)
    x = relu(x)
    _log("RELU1", x)
    x = maxpool2d(x, pool_size=3, stride=2)
    _log("MaxPool1", x)

    # Bloc 2 : Conv + RELU + MaxPool
    x = conv2d_fast(x, weights["conv2_w"], weights["conv2_b"])
    _log("Conv2 (avant RELU)", x)
    x = relu(x)
    _log("RELU2", x)
    x = maxpool2d(x, pool_size=3, stride=2)
    _log("MaxPool2", x)

    # Bloc 3 : Conv + RELU + MaxPool
    x = conv2d_fast(x, weights["conv3_w"], weights["conv3_b"])
    _log("Conv3 (avant RELU)", x)
    x = relu(x)
    _log("RELU3", x)
    x = maxpool2d(x, pool_size=3, stride=2)
    _log("MaxPool3", x)

    # Flatten : 3x3x20 = 180
    _log("Avant reshape", x)
    # Reshape en ordre canonique (C, H, W) → vecteur colonne-major
    # IMPORTANT : l'ordre doit correspondre aux poids FC appris
    # Le projet SEI utilise l'ordre HWC (Height-Width-Channel) pour le reshape
    x_hwc = x.transpose(1, 2, 0)   # (C,H,W) → (H,W,C) = (3,3,20)
    vec = x_hwc.flatten()           # 180 éléments
    _log("Vecteur (après reshape)", vec)

    # Fully Connected
    logits = fully_connected(vec, weights["fc_w"], weights["fc_b"])
    _log("Logits FC", logits)

    # SoftMax
    probs = softmax(logits)
    _log("Probabilités SoftMax", probs)

    pred = int(np.argmax(probs))
    return probs, pred


# ────────────────────────────────────────────────────────────────────────────
# Test avec des poids fictifs (valide la logique sans fichier de poids)
# ────────────────────────────────────────────────────────────────────────────

def _test_with_synthetic_weights():
    """
    Validation de la logique de convolution avec des poids contrôlés.
    Un kernel identité doit reproduire l'image d'entrée (1 canal).
    """
    print("\n[Test] Convolution avec kernel identité (1 canal)...")
    # Image test 1x5x5 remplie avec des valeurs connues
    img  = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
    # Kernel identité 1x1x3x3 (centre=1, reste=0)
    kern = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kern[0, 0, 1, 1] = 1.0
    bias = np.zeros(1, dtype=np.float32)

    out_fast = conv2d_fast(img, kern, bias, padding="same")
    out_ref  = conv2d(img, kern, bias, padding="same")

    assert out_fast.shape == (1, 5, 5), f"Forme incorrecte : {out_fast.shape}"
    assert np.allclose(out_fast, img, atol=1e-5), \
        "Conv identité : résultat différent de l'entrée"
    assert np.allclose(out_fast, out_ref, atol=1e-5), \
        "Divergence entre conv_fast et conv de référence"
    print("    [OK] Kernel identité : sortie = entrée")

    print("[Test] RELU (valeurs négatives zéro, positives inchangées)...")
    x = np.array([-2.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float32)
    r = relu(x)
    assert np.allclose(r, [0, 0, 0, 0.5, 3.0]), f"RELU incorrect : {r}"
    print("    [OK]")

    print("[Test] MaxPool 2x2 sur image 1x4x4...")
    img4 = np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]],
                    dtype=np.float32).reshape(1, 4, 4)
    mp = maxpool2d(img4, pool_size=2, stride=2)
    assert mp.shape == (1, 2, 2), f"Forme MaxPool incorrecte : {mp.shape}"
    expected = np.array([[[6, 8],[14, 16]]], dtype=np.float32)
    assert np.allclose(mp, expected), f"MaxPool incorrect : {mp}"
    print("    [OK]")

    print("[Test] SoftMax (somme = 1, valeur max = argmax)...")
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    probs  = softmax(logits)
    assert abs(probs.sum() - 1.0) < 1e-6, f"SoftMax somme={probs.sum()}"
    assert np.argmax(probs) == 4, "SoftMax argmax incorrect"
    print("    [OK]")

    print("\n[OK] Tous les tests unitaires passés.")


if __name__ == "__main__":
    _test_with_synthetic_weights()