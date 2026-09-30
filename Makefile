PYTHON := python3

test:
	cd python && $(PYTHON) -m pytest tests/ -v --tb=short -m "not slow" --disable-warnings

test-all:
	cd python && $(PYTHON) -m pytest tests/ -v --tb=short --disable-warnings

run:
	cd python && $(PYTHON) main.py --single 0

coverage:
	cd python && $(PYTHON) -m pytest tests/ -v --disable-warnings --cov=./ --cov-report=term-missing

clean:
	rm -rf python/__pycache__
	rm -rf python/tests/__pycache__
	rm -rf python/.pytest_cache
	rm -rf python/htmlcov
	rm -rf python/coverage.xml

