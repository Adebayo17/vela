# bash, not /bin/sh (dash): Vitis' settings64.sh uses `source`
SHELL := /bin/bash

# Heavy EDA runs (Vitis, Vivado) in their own systemd scope with a memory cap:
# on a memory shortage only that run is killed, not the terminal (and the
# Claude Code session) it was started from — systemd-oomd kills a whole
# terminal scope under memory pressure (incident of 2026-11-01).
# MEM_LIMIT= (empty) disables the wrapper.
MEM_LIMIT ?= 4G
LIMITED = $(if $(MEM_LIMIT),systemd-run --user --scope --quiet -p MemoryMax=$(MEM_LIMIT) -p MemorySwapMax=1G,)

# Use the project venv (created by `make setup`) when it exists, else system python3.
# Absolute path: some targets `cd python` first.
PYTHON := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,python3)

# ─── Setup ───────────────────────────────────────────────────────────────────

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "Venv ready. Run: source .venv/bin/activate"
	@echo "If tkinter is missing: sudo apt install python3-tk"


# ─── Tests principaux ────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest -v --tb=short -m "not slow" --disable-warnings

test-all:
	$(PYTHON) -m pytest -v --tb=short --disable-warnings


# ─── Tests par module ────────────────────────────────────────────────────────

test-cnn:
	$(PYTHON) -m pytest python/tests/test_cnn.py -v --tb=short -m "not slow"

test-filters:
	$(PYTHON) -m pytest python/tests/test_filters.py -v --tb=short


# ─── Tests filtre par filtre ─────────────────────────────────────────────────

test-grayscale:
	$(PYTHON) -m pytest python/tests/test_filters.py::TestGrayscale -v --tb=short

test-sobel:
	$(PYTHON) -m pytest python/tests/test_filters.py::TestSobel -v --tb=short

test-gaussian:
	$(PYTHON) -m pytest python/tests/test_filters.py::TestGaussian -v --tb=short

test-median:
	$(PYTHON) -m pytest python/tests/test_filters.py::TestMedian -v --tb=short

test-threshold:
	$(PYTHON) -m pytest python/tests/test_filters.py::TestThreshold -v --tb=short


# ─── RTL (Phase 2a) ──────────────────────────────────────────────────────────

RTL_SRC := $(wildcard rtl/*.v)
TRACE_REPORT := docs/requirement/traceability_report.md

rtl-test:
	$(PYTHON) -m pytest verif/tests -v --tb=short

# Same tests + VCD traces for GTKWave in build/waves/<test>/ (K = pytest -k filter)
#   make rtl-waves K=grayscale
#   gtkwave build/waves/<test>/wave.gtkw
rtl-waves:
	VELA_WAVES=1 $(PYTHON) -m pytest verif/tests -v --tb=short $(if $(K),-k "$(K)")
	@echo ""
	@echo "Traces : $$(ls -d build/waves/*/ 2>/dev/null | wc -l) dossiers dans build/waves/"
	@echo "Ouvrir : gtkwave build/waves/<test>/wave.gtkw"

rtl-lint:
	@for f in $(RTL_SRC); do \
		echo "lint $$f"; verilator --lint-only -Wall -y rtl $$f || exit 1; \
	done


# ─── HLS (phase 2b) ──────────────────────────────────────────────────────────
# En-tête C++ des poids (généré, non versionné) + tests de la référence C++

hls-weights:
	$(PYTHON) tools/gen_weights_header.py

# Référence C++ double + CNN ap_fixed (g++ et en-têtes Vitis, comme une C-simulation)
hls-test:
	$(PYTHON) -m pytest verif/tests/test_hls_cnn_ref.py verif/tests/test_hls_cnn_hw.py -v --tb=short

# Vitis HLS (étape 4) : projet dans build/hls/cnn_proj, testbench hls/testbench.cpp
# comparé au modèle bit-exact (données : tools/gen_hls_tb_data.py).
#   make hls-csim     C-simulation (~1 min)
#   make hls-synth    C-simulation + synthèse, rapport de latence / ressources (~3 min)
#   make hls-cosim    synthèse + co-simulation C/RTL sur HLS_N images (long)
#   make hls-export   synthèse + export de l'IP (build/hls/cnn_hw_ip.zip)
#   make hls-impl     export + implémentation Vivado hors contexte (NFR-009/010)
VITIS_SETTINGS ?= $(HOME)/tools/Xilinx/Vitis/2024.2/settings64.sh
HLS_N ?= 20
HLS_REPORT := build/hls/cnn_proj/sol1/syn/report/cnn_hw_csynth.rpt

hls-data: hls-weights
	$(PYTHON) tools/gen_hls_tb_data.py $(HLS_N) build/hls

define run_hls
	cd build && . $(VITIS_SETTINGS) >/dev/null && \
		VELA_HLS_STEPS="$(1)" VELA_HLS_N=$(HLS_N) $(LIMITED) vitis-run --mode hls --tcl ../tools/hls.tcl \
		> hls_$(2).log 2>&1 || (tail -30 hls_$(2).log; exit 1)
	@grep -E "PASS:|FAIL:|MISMATCH" build/hls_$(2).log || true
endef

hls-csim: hls-data
	$(call run_hls,csim,csim)

hls-synth: hls-data
	$(call run_hls,csim synth,synth)
	@sed -n '/+ Latency:/,/Detail/p' $(HLS_REPORT) | sed -n '5,7p'
	@grep -E "^\|Total  |^\|Utilization" $(HLS_REPORT) | head -2

hls-cosim: HLS_N = 3
hls-cosim: hls-data
	$(call run_hls,cosim,cosim)
	@grep -E "Verilog\|" build/hls/cnn_proj/sol1/sim/report/cnn_hw_cosim.rpt 2>/dev/null || true

hls-export: hls-data
	$(call run_hls,synth export,export)
	@ls -la build/hls/cnn_hw_ip.zip

# Implémentation hors contexte du RTL généré, dans un Vivado séparé de Vitis
hls-impl: hls-export
	cd build && $(LIMITED) vivado -mode batch -nojournal -nolog \
		-source ../tools/hls_impl.tcl -tclargs hls/cnn_proj/sol1/syn/verilog hls/impl \
		> hls_impl.log 2>&1 || (tail -30 hls_impl.log; exit 1)
	@grep "^SUMMARY" build/hls_impl.log
	$(PYTHON) tools/hls_impl_summary.py build/hls/impl

# Plages de valeurs + types ap_fixed candidats sur les 10 000 images (TN004, ~3 min)
cnn-fixed-point:
	$(PYTHON) tools/cnn_fixed_point_analysis.py --images 10000 \
		--configs 16:16:rnd,14:14:rnd,12:12:rnd,16:16:trn --report build/cnn_fixed_point_10k.md


# ─── Synthèse Vivado (hors contexte) ─────────────────────────────────────────
# Synthèse + placement-routage de chaque module à 100 MHz sur xc7z020,
# rapports dans build/synth/<module>/, bilan + vérif NFR-002/004.
#   make synth                          (tous les modules, quelques minutes)
#   make synth MODULES="sobel_filter"   (un sous-ensemble)

synth:
	cd build 2>/dev/null || mkdir -p build
	cd build && $(LIMITED) vivado -mode batch -nojournal -nolog \
		-source ../tools/synth.tcl -tclargs synth $(MODULES) > synth.log 2>&1 \
		|| (tail -30 synth.log; exit 1)
	@grep "^SUMMARY" build/synth.log
	$(PYTHON) tools/synth_summary.py build/synth


# ─── FPGA (phase 3) ──────────────────────────────────────────────────────────
# Projet Vivado, block design, bitstream et XSA recréés par tools/fpga.tcl
# (VELA-TN-007). L'IP CNN vient de make hls-export (build/hls/cnn_hw_ip.zip).
#   make fpga-bd   block design validé + docs/diagrams/vela_bd.svg (~2 min)
#   make fpga      + synthèse, implémentation, bitstream, XSA (build/fpga/)

fpga-ip:
	@test -f build/hls/cnn_hw_ip.zip || (echo "build/hls/cnn_hw_ip.zip manquant : make hls-export"; exit 1)
	rm -rf build/ip/cnn_hw && mkdir -p build/ip/cnn_hw
	cd build/ip/cnn_hw && unzip -q ../../hls/cnn_hw_ip.zip

fpga-bd: fpga-ip
	cd build && $(LIMITED) vivado -mode batch -nojournal -nolog \
		-source ../tools/fpga.tcl -tclargs fpga bd > fpga_bd.log 2>&1 \
		|| (tail -40 fpga_bd.log; exit 1)
	@grep "^CHECK\|^STAGE" build/fpga_bd.log
	cd build && $(LIMITED) xvfb-run -a vivado -mode gui -nojournal -nolog \
		-source ../tools/fpga_bd_svg.tcl -tclargs fpga > fpga_bd_svg.log 2>&1 \
		|| (tail -20 fpga_bd_svg.log; exit 1)
	@grep "^STAGE" build/fpga_bd_svg.log

fpga: fpga-ip
	cd build && $(LIMITED) vivado -mode batch -nojournal -nolog \
		-source ../tools/fpga.tcl -tclargs fpga all > fpga.log 2>&1 \
		|| (tail -40 fpga.log; exit 1)
	@grep "^CHECK\|^SUMMARY\|^STAGE" build/fpga.log


# ─── Logiciel ARM (phase 3) ──────────────────────────────────────────────────
# Plateforme Vitis + application bare-metal + BOOT.BIN (tools/sw_build.py)
# depuis le XSA de make fpga. Table de validation : make sw-golden.
#   make sw      build/sw/vela_app.elf, build/sw/BOOT.BIN (~5 min)

.PHONY: sw sw-golden fpga fpga-bd fpga-ip diagrams

sw-golden:
	$(PYTHON) tools/gen_sw_golden.py

sw:
	@test -f build/fpga/vela.xsa || (echo "build/fpga/vela.xsa manquant : make fpga"; exit 1)
	$(LIMITED) vitis -s tools/sw_build.py build/fpga/vela.xsa build/vitis build/sw \
		> build/sw.log 2>&1 || (tail -40 build/sw.log; exit 1)
	@grep "^SWCHECK" build/sw.log
	@ls -la build/sw/vela_app.elf build/sw/BOOT.BIN


# ─── Documentation ───────────────────────────────────────────────────────────
# Schémas SVG de docs/diagrams (certains animés), générés par script

diagrams:
	$(PYTHON) tools/gen_diagrams.py


# ─── Traçabilité ─────────────────────────────────────────────────────────────

trace:
	$(PYTHON) tools/traceability_scanner.py --srs docs/requirement/requirements.md \
		--tests python/tests verif/tests --source python verif/models \
		--output $(TRACE_REPORT)


# ─── Exécution ───────────────────────────────────────────────────────────────

run:
	cd python && $(PYTHON) main.py --single 0


# ─── GUI ─────────────────────────────────────────────────────────────────────

gui:
	cd python && $(PYTHON) gui_simulator.py

gui-small:
	cd python && $(PYTHON) gui_simulator.py --images 20


# ─── Couverture ──────────────────────────────────────────────────────────────

coverage:
	$(PYTHON) -m pytest -v --disable-warnings --cov=python --cov=verif/models --cov-report=term-missing


# ─── Nettoyage ───────────────────────────────────────────────────────────────

clean:
	rm -rf .pytest_cache
	rm -rf python/__pycache__
	rm -rf python/.pytest_cache
	rm -rf python/htmlcov
	rm -rf python/coverage.xml
	rm -rf .coverage htmlcov coverage.xml python/.coverage
	rm -rf python/tests/__pycache__
	rm -rf verif/tests/__pycache__ verif/models/__pycache__ tools/__pycache__
	rm -rf build hls/generated