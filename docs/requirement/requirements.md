## Requirements

| ID          | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| VELA-FR-001 | Pipeline CNN complet (toutes couches intégrées)               |
| VELA-FR-002 | Chargement des poids TF avec transposition CHW                |
| VELA-FR-003 | Prétraitement CIFAR-10 (crop + normalisation)                 |
| VELA-FR-004 | cnn_forward() retourne (probs shape 10, pred int 0-9)         |
| VELA-FR-005 | Accuracy ≥ 60% sur 1000 images test                           |
| VELA-FR-006 | Aucune classe à 0% de précision                               |
| VELA-FR-007 | Filtres image (grayscale, sobel, gaussian, median, threshold) |