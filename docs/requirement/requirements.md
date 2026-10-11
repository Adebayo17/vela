## Requirements

| ID          | Description                                                                  |
| ----------- | ---------------------------------------------------------------------------- |
| VELA-FR-001 | Pipeline CNN complet (toutes couches intégrées)                              |
| VELA-FR-002 | Chargement des poids TF avec transposition CHW                               |
| VELA-FR-003 | Prétraitement CIFAR-10 (crop + normalisation)                                |
| VELA-FR-004 | cnn_forward() retourne (probs shape 10, pred int 0-9)                        |
| VELA-FR-005 | Accuracy ≥ 60% sur 1000 images test                                          |
| VELA-FR-006 | Aucune classe à 0% de précision                                              |
| VELA-FR-007 | Filtres image (grayscale, sobel, gaussian, median, threshold)                |
| VELA-FR-008 | RTL des filtres bit-exact vs modèle entier, < 1 LSB vs golden model flottant |
| VELA-FR-009 | Affichage OLED : init SSD1331 puis pixels RGB565 (octet haut d'abord) en SPI mode 0 à 6,25 MHz |

## Non-functional requirements

Cible matérielle : Zynq XC7Z020 (`xc7z020clg400-1`, Digilent Arty Z7-20).
Ressources disponibles : 53 200 LUT, 106 400 FF, 140 BRAM 36 Kb, 220 DSP.

| ID           | Description                                                                                                                     | Vérification                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| VELA-NFR-001 | Toute la logique PL fonctionne sur une horloge unique de 100 MHz, fournie par `FCLK_CLK0` du PS                                 | Contrainte d'horloge de synthèse ; block design (phase 3)     |
| VELA-NFR-002 | Chaque module RTL respecte le timing à 100 MHz après placement-routage hors contexte : WNS ≥ 0 et WHS ≥ 0                       | `make synth` (rapport de timing par module)                   |
| VELA-NFR-003 | Les filtres acceptent 1 pixel par cycle en régime établi ; les filtres 3x3 ont au plus `IMG_WIDTH + 1` cycles d'arrêt par image | Simulation : test de débit (nombre de cycles par image)       |
| VELA-NFR-004 | Les filtres n'utilisent aucun DSP (réservés au CNN) et occupent au total ≤ 10 % des LUT et des FF (≤ 5 320 LUT, ≤ 10 640 FF)    | `make synth` (rapport d'utilisation par module)               |
| VELA-NFR-005 | Le RTL de `rtl/` est en Verilog-2001 synthétisable, sans syntaxe SystemVerilog ni primitive Xilinx, et passe le lint Verilator `-Wall` sans avertissement | `make rtl-lint` + compilation `iverilog -g2005`               |

Les seuils de VELA-NFR-004 sont un budget initial : ils réservent l'essentiel
du FPGA au CNN (phase 2b) et seront révisés après la première synthèse.
