"""
conftest.py — Shared pytest fixtures and requirement decorator.

This file is automatically loaded by pytest before any test module.
It provides:
    - @requirement decorator    : links a test to one or more SRS requirements IDs.
    - Shared fixtures           : paths, loaded weights, sample images. 
"""

import os
import numpy as np
import pytest

## ———— Requirement Decorator —————————————————————————————————————————————————————— ##

def requirement(*req_ids):
    """
    Decorator that links a test function to one or more SRS requirement IDs.
    
    Usage:
        @requirement("VELA-FR-001")
        def test_conv2d_identity():
            ...
        
        @requirement("VELA-FR-003", "VELA-FR-004")
        def test_preprocess_statistics():
            ...
        
    The decorator adds a pytest mark that is picked up by the 
    traceability scanner (tools/traceability_scanner.py) to generate the traceability matrix.
    """
    ids_label = ",".join(req_ids)
    mark = pytest.mark.requirement(ids=list(req_ids), reason=f"Covers: {ids_label}")
    
    def decorator(func):
        func._requirement_ids = list(req_ids)  # Attach requirement IDs to the function for traceability
        return mark(func)
    
    return decorator

## ———— pytest marks registration —————————————————————————————————————————————————— ##

def pytest_configure(config):
    """
    Register custom pytest marks for requirements.
    This allows us to use @pytest.mark.requirement in tests without warnings.
    """
    config.addinivalue_line(
        "markers", 
        "requirement(ids, reason): marks test as covering specific SRS requirements"
    )
    
    config.addinivalue_line(
        "markers",
        "slow: marks test as slow (excluded from fast CI run with -m 'not slow')"
    )

## ———— Path Fixtures —————————————————————————————————————————————————————————————— ##

@pytest.fixture(scope="session")
def data_dir():
    """
    Path to CIFAR-10 binary data directory.
    """
    path = os.environ.get(
        "VELA_DATA_DIR",
        os.path.join(os.path.dirname(__file__), "../../../data/cifar-10-batches-bin")
    )
    return os.path.abspath(path)


@pytest.fixture(scope="session")
def data_available(data_dir):
    """
    Returns True if CIFAR-10 data files are accessible.
    """
    return os.path.exists(os.path.join(data_dir, "test_batch.bin"))


@pytest.fixture(scope="session")
def weights_file():
    """
    Path to CNN_coeff_3x3.txt.
    """
    path = os.environ.get(
        "VELA_WEIGHTS_FILE",
        os.path.join(os.path.dirname(__file__), "../../../data/CNN_coeff_3x3.txt")
    )
    return os.path.abspath(path)


@pytest.fixture(scope="session")
def weights_available(weights_file):
    """
    Returns True if the CNN weights file is accessible.
    """
    return os.path.exists(weights_file)


## ———— Data Fixtures —————————————————————————————————————————————————————————————— ##

@pytest.fixture(scope="session")
def loaded_weights(weights_file, weights_available):
    """
    Session-scoped fixture: loads CNN weights once for all tests.
    Skips if weight file not found.
    """
    if not weights_available:
        pytest.skip(f"Weight file not found: {weights_file}")
    from weights_loader import load_weights
    return load_weights(weights_file)


@pytest.fixture(scope="session")
def sample_images_10(data_dir, data_available):
    """
    Session-scoped fixture: loads first 10 preprocessed CIFAR-10 images.
    Skip if data not found.
    """
    if not data_available:
        pytest.skip(f"CIFAR-10 data not found in: {data_dir}")
    from data_loader import load_dataset
    images, labels = load_dataset(data_dir, max_images=10)
    return images, labels
    
@pytest.fixture(scope="session")
def sample_images_100(data_dir, data_available):
    """
    Session-scoped fixture: loads first 100 preprocessed CIFAR-10 images.
    Skip if data not found.
    """
    if not data_available:
        pytest.skip(f"CIFAR-10 data not found in: {data_dir}")
    from data_loader import load_dataset
    images, labels = load_dataset(data_dir, max_images=100)
    return images, labels


## ———— Synthetic Data Fixtures (no files needed) —————————————————————————————————— ##

@pytest.fixture
def synthetic_image_1channel():
    """
    Synthetic 1x5x5 feature map filled with arange(25).
    """
    return np.arange(25, dtype=np.float32).reshape(1, 5, 5)

@pytest.fixture
def identity_kernel_1to1():
    """
    Synthetic 1x1x3x3 identity kernel (center=1, rest=0).
    """
    kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kernel[0, 0, 1, 1] = 1.0
    return kernel

@pytest.fixture
def zero_bias_1():
    """
    Synthetic single zero bias.
    """
    return np.zeros(1, dtype=np.float32)

@pytest.fixture
def synthetic_rgb_image_32x32():
    """
    Synthetic 3x32x32 raw CIFAR-10-like image, values in [0, 255].
    """
    rng = np.random.default_rng(seed=42)
    img = rng.integers(0, 256, size=(3, 32, 32), dtype=np.uint8).astype(np.float32)
    return img
