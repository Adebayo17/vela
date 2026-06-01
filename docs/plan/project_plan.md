# Plan de projet — CNN sur FPGA (Arty Z7-20)

## Vue d'ensemble des 3 implémentations

| # | Nom | Entrée | Traitement | Sortie | Statut |
|---|-----|--------|-----------|--------|--------|
| 1 | `impl_1_sobel` | Pattern généré | Sobel PL Verilog | OLED live | ✅ Terminé |
| 2 | `impl_2_cnn_sw` | SD card / fichiers | CNN ARM C++ | Debug UART + OLED | 🔨 En cours |
| 3 | `impl_3_cnn_hw` | SD card / caméra | CNN accéléré HLS+PL | OLED live | 📋 Planifié |

**Réutilisation entre implémentations :**
- `oled_controller.v` → impl_1 → impl_3 (inchangé)
- `spi_master.v`      → impl_1 → impl_3 (inchangé)
- `constraints.xdc`  → impl_1 → impl_3 (étendu)
- `cnn_hw.cpp`        → impl_2 → impl_3 (exporté comme IP)

---

## Labels GitHub

| Label | Couleur | Description |
|-------|---------|-------------|
| `setup` | `#6c757d` | Configuration initiale du projet |
| `documentation` | `#0075ca` | README, rapports, analyses |
| `impl-1-sobel` | `#e4b429` | Implémentation 1 — Filtre Sobel |
| `impl-2-cnn-sw` | `#28a745` | Implémentation 2 — CNN logiciel |
| `impl-3-cnn-hw` | `#d73a4a` | Implémentation 3 — CNN accéléré HLS |
| `python` | `#3572A5` | Code Python Phase 1 |
| `cpp` | `#f34b7d` | Code C++ Phase 2 |
| `hls` | `#b60205` | Vitis HLS spécifique |
| `verilog` | `#7b68ee` | Code Verilog RTL |
| `vivado` | `#e10098` | Projets et scripts Vivado |
| `embedded-c` | `#795548` | Application ARM bare-metal |
| `phase-1` | `#bfd4f2` | Phase 1 Python |
| `phase-2` | `#bfe5bf` | Phase 2 HLS C++ |
| `phase-3` | `#ffddbf` | Phase 3 Intégration FPGA |
| `validation` | `#00b4d8` | Tests et vérification |
| `analysis` | `#9c36b5` | Analyse et dimensionnement |
| `optimization` | `#ff6b35` | Optimisations performance |
| `enhancement` | `#a2eeef` | Améliorations optionnelles |

---

## Milestones (Sprints)

### Sprint 0 — Setup (5 tâches)
**Objectif :** dépôt propre, 3 implémentations structurées, impl_1 archivée.

- [ ] Initialiser la structure du dépôt
- [ ] Configurer .gitignore pour Vivado et Vitis
- [ ] Écrire le README principal du projet
- [ ] Archiver et documenter le projet Sobel
- [ ] Ajouter un script TCL de reconstruction Vivado (impl_1)

---

### Sprint 1 — Python CNN (5 tâches)
**Objectif :** référence algorithmique validée ≥ 60% sur CIFAR-10.

- [ ] Implémenter data_loader.py
- [ ] Implémenter weights_loader.py
- [ ] Implémenter les couches CNN dans cnn.py
- [ ] Implémenter top.py et valider sur 100 images
- [ ] Valider le taux de détection sur 1000 images

**Critère de sortie de sprint :** `python top.py --images 1000` affiche ≥ 60%.

---

### Sprint 2 — HLS C++ (6 tâches)
**Objectif :** IP HLS synthétisée, exportée, métriques documentées.

- [ ] Porter le CNN en C++ double précision (référence)
- [ ] Analyser les plages de valeurs pour le dimensionnement ap_fixed
- [ ] Porter le CNN en C++ ap_fixed pour Vitis HLS
- [ ] Synthèse Vitis HLS et analyse des métriques
- [ ] Optimiser la HLS avec directives pipeline/unroll
- [ ] Exporter l'IP HLS vers Vivado

**Critère de sortie de sprint :** IP exportée dans `impl_3_cnn_hw/ip/`, rapport HLS committé.

---

### Sprint 3 — Intégration FPGA (7 tâches)
**Objectif :** CNN tourne sur la carte, résultat affiché sur l'OLED.

- [ ] Créer le block design Vivado pour impl_3
- [ ] Écrire le wrapper Verilog top_cnn_oled.v
- [ ] Écrire result_display.v
- [ ] Écrire l'application ARM bare-metal (main.c)
- [ ] Mettre à jour le fichier de contraintes XDC
- [ ] Générer et flasher le bitstream
- [ ] Validation finale : 10 images sur carte

**Critère de sortie de sprint :** ≥ 9/10 images : résultat FPGA = résultat Python.

---

### Sprint 4 — Améliorations (4 tâches, optionnel)
**Objectif :** fonctionnalités bonus et documentation finale.

- [ ] Ajouter la navigation entre images (BTN1)
- [ ] Ajouter l'affichage du top-3 des classes
- [ ] Tester CNN_coeff_5x5.txt (kernels 5×5)
- [ ] Exporter le script TCL de reconstruction impl_3

---

## Instructions d'import GitHub

### Créer les labels
Dans `Settings → Labels`, créer chaque label du tableau ci-dessus.

### Créer les milestones
Dans `Issues → Milestones` :
1. Sprint 0 — Setup
2. Sprint 1 — Python CNN
3. Sprint 2 — HLS C++
4. Sprint 3 — Intégration FPGA
5. Sprint 4 — Améliorations

### Importer les tâches (via GitHub CLI)
```bash
# Installer gh si nécessaire
gh auth login

# Créer les issues depuis le CSV (script manuel ou via gh issue create)
while IFS=',' read -r title body labels milestone; do
  gh issue create --title "$title" --body "$body" --label "$labels" --milestone "$milestone"
done < github_tasks.csv
```

### Ou manuellement
Créer chaque issue dans `Issues → New Issue`, en assignant les labels et milestones correspondants, puis les ajouter au Project Board en vue Kanban avec les colonnes :
- **Backlog** → tâches Sprint 4
- **À faire** → tâches du sprint courant
- **En cours** → tâches actives (limit WIP : 2)
- **En revue** → PR ouverte
- **Terminé** → validé