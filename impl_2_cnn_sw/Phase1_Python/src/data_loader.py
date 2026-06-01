"""
data_loader.py — Loading and preprocessing data for the CNN model.

Binary format CIFAR-10:
    - [1 byte label][3072 bytes pixel data] x 10000 images per batch
    - 3072 = 3 channels (RGB) * 32 width * 32 height
    - Order: R(1024 bytes), G(1024 bytes), B(1024 bytes), row-major

Applied preprocessing steps:
    1. Read the raw 32x32x3 image
    2. Centered 24x24 crop
    3. Normalization: m' = (m - µ) / max(σ, 1/√N)
        µ and σ are calculated simultaneously across all 3 channels (not per separate channel)
"""

import numpy as np 
import os

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

IMAGE_SIZE_ORIGINAL = 32
IMAGE_SIZE_CROPPED  = 24
N_CHANNELS          = 3
RECORD_SIZE         = 1 + N_CHANNELS * IMAGE_SIZE_ORIGINAL * IMAGE_SIZE_ORIGINAL  # 3073 bytes per record


def load_batch(filepath):
    """
    Load a single CIFAR-10 batch file.
    
    Returns:
        images: np.ndarray float32 of shape (N, 3, 32, 32) - values in [0, 255]
        labels: np.ndarray int of shape (N,) - values in [0, 9]
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
        
    n_images = len(raw) // RECORD_SIZE
    raw = raw[:n_images * RECORD_SIZE].reshape(n_images, RECORD_SIZE)
    
    labels = raw[:, 0].astype(np.int32)  # First byte is the label
    pixels = raw[:, 1:].reshape(n_images, N_CHANNELS, IMAGE_SIZE_ORIGINAL, IMAGE_SIZE_ORIGINAL)  # Reshape to (N, 3, 32, 32)
    images = pixels.astype(np.float32)  # Convert to float32 for preprocessing
    
    return images, labels


def crop_center(image, crop_size=IMAGE_SIZE_CROPPED):
    """
    Centered crop 
    """
    _, h, w = image.shape
    assert h >= crop_size and w >= crop_size, f"Image ({h}, {w}) too short to crop {crop_size}"
    top     = (h - crop_size) // 2
    left    = (w - crop_size) // 2
    return image[:, top:top + crop_size, left:left + crop_size]


def normalize(image): 
    """
    Global normalization: 
        µ  = moyenne de tous les pixels de l'image (3 canaux ensemble)
        σ  = écart-type de tous les pixels
        m' = (m - µ) / max(σ, 1/√N)
    """
    n = image.size
    mu = image.mean()
    sigma = image.std()
    min_std = 1.0 / np.sqrt(n)
    normalized = (image - mu) / max(sigma, min_std)
    return normalized.astype(np.float32)
    
def preprocess(image):
    """
    Pipeline complet : crop centré + normalisation.
    Entrée  : (3, 32, 32) float32
    Sortie  : (3, 24, 24) float32 normalisée
    """
    cropped    = crop_center(image)
    normalized = normalize(cropped)
    return normalized
 
 
def load_dataset(data_dir, filename="test_batch.bin", max_images=None):
    """
    Charge et prétraite un batch CIFAR-10 complet.
 
    Args:
        data_dir   : répertoire contenant les fichiers .bin
        filename   : nom du fichier (défaut : test_batch.bin)
        max_images : si précisé, limite le nombre d'images chargées
 
    Retourne :
        images_pp : np.ndarray (N, 3, 24, 24) float32 prétraitées
        labels    : np.ndarray (N,) int32
    """
    filepath = os.path.join(data_dir, filename)
    images_raw, labels = load_batch(filepath)
 
    if max_images is not None:
        images_raw = images_raw[:max_images]
        labels     = labels[:max_images]
 
    images_pp = np.array([preprocess(img) for img in images_raw],
                         dtype=np.float32)
    return images_pp, labels
 
 
# ─── Test rapide (exécuter ce fichier directement) ──────────────────────────
if __name__ == "__main__":
    import sys
 
    # Adapter ce chemin à votre installation
    DATA_DIR = "./cifar10_data/cifar-10-batches-bin"
 
    print("=== Test data_loader.py ===\n")
 
    # 1. Chargement d'un seul batch
    try:
        imgs, labs = load_batch(os.path.join(DATA_DIR, "test_batch.bin"))
        print(f"[OK] Batch chargé : {imgs.shape} images, {labs.shape} labels")
        print(f"     Labels uniques présents : {sorted(set(labs.tolist()))}")
        print(f"     Exemple label[0] = {labs[0]} ({CIFAR10_CLASSES[labs[0]]})")
        print(f"     Pixel min={imgs[0].min():.0f}  max={imgs[0].max():.0f}")
    except FileNotFoundError as e:
        print(f"[!] {e}")
        print("    → Vérifiez le chemin DATA_DIR dans ce script.")
        sys.exit(1)
 
    # 2. Crop
    img0_crop = crop_center(imgs[0])
    print(f"\n[OK] Crop 24×24 : {imgs[0].shape} → {img0_crop.shape}")
 
    # 3. Normalisation
    img0_norm = normalize(img0_crop)
    print(f"[OK] Normalisation : µ={img0_norm.mean():.4f}  "
          f"σ={img0_norm.std():.4f}  "
          f"min={img0_norm.min():.3f}  max={img0_norm.max():.3f}")
 
    # 4. Pipeline complet sur 100 images
    imgs_pp, labs_pp = load_dataset(DATA_DIR, max_images=100)
    print(f"\n[OK] Pipeline complet (100 images) : {imgs_pp.shape}")
    print(f"     µ moyen sur le batch = {imgs_pp.mean():.6f}  "
          f"(attendu ≈ 0)")
    print("\n=== Tous les tests passés ===")