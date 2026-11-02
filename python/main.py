"""
main.py — Point d'entrée : validation du CNN CIFAR-10

Usage :
    python main.py                             # 100 images, chemins par défaut
    python main.py --images 1000               # 1000 images
    python main.py --images 0                  # toutes les images (10000)
    python main.py --verbose                   # affiche les détails de chaque image
    python main.py --single 42                 # inférence sur l'image n°42 uniquement
    python main.py --data /chemin/vers/cifar10 # chemin personnalisé
"""

import argparse
import os
import sys
import time
import numpy as np

# Modules du projet
from data_loader    import load_dataset, load_batch, preprocess, CIFAR10_CLASSES
from weights_loader import load_weights, print_weights_summary
from cnn            import cnn_forward


# ─── Chemins par défaut ──────────────────────────────────────────────────────
DEFAULT_DATA_DIR    = "../data/cifar-10-batches-bin"
DEFAULT_WEIGHTS_FILE = "../data/CNN_coeff_3x3.txt"
DEFAULT_N_IMAGES    = 100


# ─── Inférence sur un batch ──────────────────────────────────────────────────

def run_validation(images, labels, weights, verbose=False):
    """
    Exécute le CNN sur toutes les images et calcule les métriques.

    Retourne :
        n_correct : int   — nombre de bonnes prédictions
        accuracy  : float — taux de reconnaissance [0, 1]
        errors    : list  — liste des indices d'images mal classées
    """
    n = len(images)
    n_correct = 0
    errors = []

    for i, (img, label) in enumerate(zip(images, labels)):
        probs, pred = cnn_forward(img, weights, verbose=False)

        correct = (pred == label)
        if correct:
            n_correct += 1
        else:
            errors.append(i)

        if verbose or (i % 100 == 0 and not verbose):
            conf = float(probs[pred]) * 100
            mark = "✓" if correct else "✗"
            print(f"  [{i:5d}] {mark}  "
                  f"prédit={CIFAR10_CLASSES[pred]:<12} "
                  f"réel={CIFAR10_CLASSES[label]:<12} "
                  f"confiance={conf:5.1f}%")

    accuracy = n_correct / n if n > 0 else 0.0
    return n_correct, accuracy, errors


def print_confusion_summary(images, labels, weights, n_classes=10):
    """Affiche une matrice de confusion simplifiée (vrais vs prédits)."""
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for img, label in zip(images, labels):
        _, pred = cnn_forward(img, weights)
        confusion[label, pred] += 1

    print("\nMatrice de confusion (lignes=réel, colonnes=prédit) :")
    header = " " * 14 + "  ".join(f"{c[:6]:>6}" for c in CIFAR10_CLASSES)
    print(header)
    for i, row in enumerate(confusion):
        label_name = f"{CIFAR10_CLASSES[i]:<12}"
        values = "  ".join(f"{v:>6}" for v in row)
        print(f"  {label_name}  {values}")

    # Précision par classe
    print("\nPrécision par classe :")
    for i in range(n_classes):
        total = confusion[i].sum()
        if total > 0:
            acc = confusion[i, i] / total * 100
            print(f"  {CIFAR10_CLASSES[i]:<12} : {acc:5.1f}%  "
                  f"({confusion[i,i]}/{total})")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CNN CIFAR-10 — validation Python (référence algorithmique)"
    )
    parser.add_argument("--data",    default=DEFAULT_DATA_DIR,
                        help="Répertoire contenant les fichiers .bin CIFAR-10")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS_FILE,
                        help="Fichier de poids CNN_coeff_3x3.txt")
    parser.add_argument("--images",  type=int, default=DEFAULT_N_IMAGES,
                        help="Nombre d'images à traiter (0 = toutes)")
    parser.add_argument("--verbose", action="store_true",
                        help="Affiche le résultat pour chaque image")
    parser.add_argument("--single",  type=int, default=-1,
                        help="Inférence sur une seule image (numéro)")
    parser.add_argument("--confusion", action="store_true",
                        help="Affiche la matrice de confusion")
    parser.add_argument("--weights-summary", action="store_true",
                        help="Affiche le résumé des poids chargés")
    args = parser.parse_args()

    print("=" * 60)
    print("CNN CIFAR-10 — Référence Python")
    print("=" * 60)

    # ── 1. Chargement des poids ──────────────────────────────────
    print(f"\n[1] Chargement des poids : {args.weights}")
    try:
        weights = load_weights(args.weights)
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERREUR : {e}")
        sys.exit(1)
    print(f"  [OK] {sum(a.size for a in weights.values())} paramètres chargés")

    if args.weights_summary:
        print()
        print_weights_summary(weights)

    # ── 2. Mode image unique ─────────────────────────────────────
    if args.single >= 0:
        print(f"\n[2] Inférence sur l'image n°{args.single}")
        try:
            from data_loader import load_batch
            imgs_raw, labs = load_batch(
                os.path.join(args.data, "test_batch.bin")
            )
        except FileNotFoundError as e:
            print(f"  ERREUR : {e}")
            sys.exit(1)

        img_raw = imgs_raw[args.single]
        label   = labs[args.single]
        img_pp  = preprocess(img_raw)

        print(f"  Label réel : {CIFAR10_CLASSES[label]} ({label})")
        print(f"\n  Passes forward (verbose) :")
        probs, pred = cnn_forward(img_pp, weights, verbose=True)

        print(f"\n  ─── Résultat ───")
        print(f"  Prédit : {CIFAR10_CLASSES[pred]} ({pred})  "
              f"{'✓ CORRECT' if pred == label else '✗ INCORRECT'}")
        print(f"\n  Probabilités :")
        for i, (cls, p) in enumerate(zip(CIFAR10_CLASSES, probs)):
            bar = "█" * int(p * 40)
            mark = " ←" if i == pred else ""
            print(f"    {cls:<12} {p:.4f}  {bar}{mark}")
        return

    # ── 3. Chargement du dataset ─────────────────────────────────
    max_images = args.images if args.images > 0 else None
    n_label    = f"{max_images}" if max_images else "toutes"
    print(f"\n[2] Chargement de {n_label} images depuis : {args.data}")
    try:
        images, labels = load_dataset(
            args.data,
            filename="test_batch.bin",
            max_images=max_images
        )
    except FileNotFoundError as e:
        print(f"  ERREUR : {e}")
        print("\n  Téléchargez CIFAR-10 (version binaire) depuis :")
        print("  https://www.cs.toronto.edu/~kriz/cifar.html")
        print(f"  Puis extrayez dans : {args.data}")
        sys.exit(1)
    print(f"  [OK] {len(images)} images chargées  (format {images.shape})")

    # ── 4. Validation ────────────────────────────────────────────
    print(f"\n[3] Inférence sur {len(images)} images...")
    if not args.verbose:
        print("  (une ligne toutes les 100 images ; utilisez --verbose pour tout voir)\n")

    t_start = time.time()
    n_correct, accuracy, errors = run_validation(
        images, labels, weights, verbose=args.verbose
    )
    t_elapsed = time.time() - t_start

    # ── 5. Résultats ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(f"  Images testées    : {len(images)}")
    print(f"  Correctes         : {n_correct}")
    print(f"  Taux de détection : {accuracy * 100:.2f}%")
    print(f"  Temps total       : {t_elapsed:.1f}s  "
          f"({t_elapsed / len(images) * 1000:.1f} ms/image)")

    if accuracy >= 0.60:
        print(f"\n  [OK] Objectif atteint (≥ 60%)")
    else:
        print(f"\n  [!] Objectif non atteint (< 60%)")
        print(f"      → Vérifiez l'ordre du reshape (CHW vs HWC)")
        print(f"      → Vérifiez le padding de la convolution")

    # ── 6. Matrice de confusion (optionnel) ──────────────────────
    if args.confusion:
        print_confusion_summary(images, labels, weights)


if __name__ == "__main__":
    main()