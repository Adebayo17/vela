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
| VELA-FR-010 | Deux modes choisis par SW0 : classification CNN des images de la carte SD (classe et probabilité sur l'OLED) et filtres d'image (filtre choisi par les boutons) |
| VELA-FR-011 | `filter_engine` écrit dans la zone image de la mémoire d'affichage, pour chaque mode, la conversion RGB565 de la sortie du modèle entier (gaussian et median en couleur, canal par canal), bit-exact |
| VELA-FR-012 | `display_reader` envoie la mémoire d'affichage à l'OLED en ordre raster, en continu, sans perdre ni dupliquer de pixel |
| VELA-FR-013 | La zone image (colonnes 0-63) de la mémoire d'affichage n'est écrite que par `filter_engine`, la zone texte (colonnes 64-95) que par l'ARM (VELA-TN-007 §5.1) |

## Non-functional requirements

Cible matérielle : Zynq XC7Z020 (`xc7z020clg400-1`, Digilent Arty Z7-20).
Ressources disponibles : 53 200 LUT, 106 400 FF, 140 BRAM 36 Kb, 220 DSP.

| ID           | Description                                                                                                                                               | Vérification                                              |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| VELA-NFR-001 | Toute la logique PL fonctionne sur une horloge unique de 100 MHz, fournie par `FCLK_CLK0` du PS                                                           | Contrainte d'horloge de synthèse ; block design (phase 3) |
| VELA-NFR-002 | Chaque module RTL respecte le timing à 100 MHz après placement-routage hors contexte : WNS ≥ 0 et WHS ≥ 0                                                 | `make synth` (rapport de timing par module)               |
| VELA-NFR-003 | Les filtres acceptent 1 pixel par cycle en régime établi ; les filtres 3x3 ont au plus `IMG_WIDTH + 1` cycles d'arrêt par image                           | Simulation : test de débit (nombre de cycles par image)   |
| VELA-NFR-004 | Les filtres n'utilisent aucun DSP (réservés au CNN) et occupent au total ≤ 10 % des LUT et des FF (≤ 5 320 LUT, ≤ 10 640 FF)                              | `make synth` (rapport d'utilisation par module)           |
| VELA-NFR-005 | Le RTL de `rtl/` est en Verilog-2001 synthétisable, sans syntaxe SystemVerilog ni primitive Xilinx, et passe le lint Verilator `-Wall` sans avertissement | `make rtl-lint` + compilation `iverilog -g2005`           |

Les seuils de VELA-NFR-004 sont un budget initial : ils réservent l'essentiel
du FPGA au CNN (phase 2b) et seront révisés après la première synthèse.

### CNN accelerator (Phase 2b, Vitis HLS 2024.2)

| ID           | Description                                                                                                                                                      | Vérification                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| VELA-NFR-006 | Le CNN en virgule fixe (`ap_fixed`) donne la même classe que le golden model flottant sur ≥ 99 % des 1000 images de test, et respecte VELA-FR-005 et VELA-FR-006 | C-simulation Vitis HLS (testbench C++) sur 1000 images       |
| VELA-NFR-007 | Le RTL généré par Vitis HLS donne des sorties identiques au bit près à celles du C++ `ap_fixed`                                                                  | Co-simulation C/RTL Vitis HLS                                |
| VELA-NFR-008 | Latence de l'inférence ≤ 10 ms par image à 100 MHz (≤ 1 000 000 cycles), de l'image chargée au résultat disponible                                               | Rapport de synthèse HLS, confirmé par la co-simulation C/RTL |
| VELA-NFR-009 | L'IP CNN occupe ≤ 50 % des DSP (≤ 110), ≤ 50 % des BRAM 36 Kb (≤ 70) et ≤ 50 % des LUT (≤ 26 600) de la XC7Z020                                                  | Rapport d'utilisation après implémentation hors contexte     |
| VELA-NFR-010 | L'IP CNN respecte le timing à 100 MHz après placement-routage hors contexte : WNS ≥ 0 et WHS ≥ 0                                                                 | Implémentation Vivado hors contexte de l'IP exportée         |

Justification des seuils (budget initial, révisable après la première synthèse HLS) :

- **NFR-006** : la virgule fixe introduit des erreurs d'arrondi ; 99 % d'accord
  avec le golden model borne leur effet tout en tolérant les images dont les
  deux premières probabilités sont presque égales.
- **NFR-008** : le CNN demande ≈ 3,9 M multiplications-accumulations par image ;
  10 ms à 100 MHz impose au moins 4 MAC par cycle en moyenne. L'inférence est
  déclenchée par un appui sur un bouton : 10 ms reste imperceptible.
- **NFR-009** : 27 846 paramètres ≈ 54 Ko en 16 bits, soit ≈ 13 BRAM 36 Kb ; la
  moitié du FPGA laisse la place aux filtres (≈ 1,4 % des LUT), au chemin
  d'affichage, à une mémoire d'image et à l'interconnexion AXI avec le PS.

### FPGA integration (Phase 3, VELA-TN-007)

| ID           | Description                                                                                       | Vérification                          |
| ------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------- |
| VELA-NFR-011 | Le design complet (block design) respecte le timing à 100 MHz après implémentation : WNS ≥ 0, WHS ≥ 0 | Rapport de timing Vivado du design complet |
| VELA-NFR-012 | Sur la carte, la classe du CNN est celle du golden model sur ≥ 9 des 10 premières images de test  | Mode de validation au démarrage (console) |

Depuis la phase 3, VELA-NFR-004 est mesurée sur `filter_engine`, qui contient
tous les filtres réellement instanciés (gaussian et median ×3 pour la couleur).
