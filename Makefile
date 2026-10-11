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

RTL_SRC := $(filter-out rtl/oled_controller.v rtl/spi_master.v,$(wildcard rtl/*.v))
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
		grep -q "TODO: Phase 2a" $$f && continue; \
		echo "lint $$f"; verilator --lint-only -Wall --quiet -y rtl $$f || exit 1; \
	done


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
	rm -rf build