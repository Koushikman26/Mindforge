"""
Pytest configuration and shared fixtures

This module provides common fixtures and configuration for all tests.
"""

import sys
import os
from pathlib import Path
import pytest
import asyncio
from typing import Generator, AsyncGenerator
import numpy as np
from unittest.mock import MagicMock, AsyncMock

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

# Import after path is set
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import Settings, get_settings
from app.ai_engine.utils.logging_config import setup_logging


# Configure test logging
setup_logging(log_level="DEBUG")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings for testing"""
    return Settings(
        environment="testing",
        debug=True,
        log_level="DEBUG",
        ai_engine_url="http://localhost:8001",
        secret_key="test-secret-key",
        database_url=None,  # Use in-memory database for tests
    )


@pytest.fixture
def override_settings(test_settings):
    """Override the get_settings dependency"""
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield test_settings
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_settings) -> TestClient:
    """Create a test client for the FastAPI app"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(override_settings):
    """Create an async test client"""
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a sample test image"""
    # Create a simple test image (640x480 RGB)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    """Convert sample image to bytes"""
    import cv2
    _, buffer = cv2.imencode('.jpg', sample_image)
    return buffer.tobytes()


@pytest.fixture
def mock_ai_service():
    """Mock AI service for testing"""
    mock = AsyncMock()
    mock.check_health = AsyncMock(return_value={"status": "healthy"})
    mock.analyze_stock_image = AsyncMock(return_value={
        "success": True,
        "products": {"product1": {"count": 10, "confidence": 0.95}},
        "summary": {"total_items": 10},
        "processing_time": 0.5,
        "error": None
    })
    return mock


@pytest.fixture
def mock_image_pipeline():
    """Mock image processing pipeline"""
    mock = MagicMock()
    mock.process_single_image = MagicMock(return_value={
        "success": True,
        "processed_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "processing_time": 0.2,
        "improvements": {"brightness": 10, "contrast": 5},
        "processing_steps": ["noise_reduction", "lighting_correction"]
    })
    mock.get_processing_statistics = MagicMock(return_value={
        "total_processed": 100,
        "average_time": 0.3
    })
    return mock


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for test files"""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def mock_redis():
    """Mock Redis client for caching tests"""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    return mock


# Markers for test categorization
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


# Test data fixtures
@pytest.fixture
def valid_product_data():
    """Valid product data for testing"""
    return {
        "product_name": "Test Product",
        "count": 10,
        "confidence": 0.95,
        "location": {"x": 100, "y": 100, "width": 50, "height": 50}
    }


@pytest.fixture
def invalid_product_data():
    """Invalid product data for testing"""
    return {
        "product_name": "",  # Invalid: empty name
        "count": -1,  # Invalid: negative count
        "confidence": 1.5,  # Invalid: confidence > 1
    }


@pytest.fixture
def batch_images(sample_image_bytes):
    """Create multiple test images for batch processing"""
    return [sample_image_bytes for _ in range(3)]


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Add any cleanup code here if needed


# Performance monitoring
@pytest.fixture
def performance_monitor():
    """Monitor test performance"""
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return PerformanceMonitor()


# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    yield
    # Cleanup is automatic when tests end