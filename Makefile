PYTHON := python3

# ─── Setup ───────────────────────────────────────────────────────────────────

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "Venv ready. Run: source .venv/bin/activate"
	@echo "If tkinter is missing: sudo apt install python3-tk"


# ─── Tests principaux ────────────────────────────────────────────────────────

test:
	cd python && $(PYTHON) -m pytest tests/ -v --tb=short -m "not slow" --disable-warnings

test-all:
	cd python && $(PYTHON) -m pytest tests/ -v --tb=short --disable-warnings


# ─── Tests par module ────────────────────────────────────────────────────────

test-cnn:
	cd python && $(PYTHON) -m pytest tests/test_cnn.py -v --tb=short -m "not slow"

test-filters:
	cd python && $(PYTHON) -m pytest tests/test_filters.py -v --tb=short


# ─── Tests filtre par filtre ─────────────────────────────────────────────────

test-grayscale:
	cd python && $(PYTHON) -m pytest tests/test_filters.py::TestGrayscale -v --tb=short

test-sobel:
	cd python && $(PYTHON) -m pytest tests/test_filters.py::TestSobel -v --tb=short

test-gaussian:
	cd python && $(PYTHON) -m pytest tests/test_filters.py::TestGaussian -v --tb=short

test-median:
	cd python && $(PYTHON) -m pytest tests/test_filters.py::TestMedian -v --tb=short

test-threshold:
	cd python && $(PYTHON) -m pytest tests/test_filters.py::TestThreshold -v --tb=short


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
	cd python && $(PYTHON) -m pytest tests/ -v --disable-warnings --cov=./ --cov-report=term-missing


# ─── Nettoyage ───────────────────────────────────────────────────────────────

clean:
	rm -rf .pytest_cache
	rm -rf python/__pycache__
	rm -rf python/.pytest_cache
	rm -rf python/htmlcov
	rm -rf python/coverage.xml
	rm -rf python/tests/__pycache__