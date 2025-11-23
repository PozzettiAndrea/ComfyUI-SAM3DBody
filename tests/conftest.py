"""
Pytest configuration for ComfyUI-SAM3DBody tests.

This module sets up mocks for ComfyUI dependencies to allow testing
without a full ComfyUI installation.
"""

import sys
from unittest.mock import MagicMock
from pathlib import Path


def pytest_configure(config):
    """Configure pytest and set up mocks before test collection."""
    # Mock ComfyUI modules that aren't available during testing
    mock_modules = [
        'folder_paths',
        'comfy',
        'comfy.utils',
        'comfy.model_management',
        'server',
        'aiohttp',
    ]

    for module_name in mock_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MagicMock()

    # Configure test markers
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no model loading)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with real models"
    )


def pytest_ignore_collect(collection_path, path, config):
    """Ignore __init__.py files during test collection."""
    if collection_path.name == "__init__.py":
        return True
    return False
