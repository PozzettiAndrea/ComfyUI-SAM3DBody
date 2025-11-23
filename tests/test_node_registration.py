"""
Unit tests for ComfyUI-SAM3DBody node registration.

These tests verify that nodes are properly registered without loading models.
"""

import pytest
import sys
from pathlib import Path


@pytest.mark.unit
def test_pytorch_available():
    """Test that PyTorch is available."""
    import torch
    assert torch.__version__ is not None


@pytest.mark.unit
def test_node_imports():
    """Test that all node classes can be imported."""
    # Import directly from node modules to bypass __init__.py
    from nodes.processing.load_model import LoadSAM3DBodyModel
    from nodes.processing.process import SAM3DBodyProcess, SAM3DBodyProcessAdvanced
    from nodes.processing.visualize import SAM3DBodyVisualize, SAM3DBodyExportMesh, SAM3DBodyGetVertices

    assert LoadSAM3DBodyModel is not None
    assert SAM3DBodyProcess is not None
    assert SAM3DBodyProcessAdvanced is not None
    assert SAM3DBodyVisualize is not None
    assert SAM3DBodyExportMesh is not None
    assert SAM3DBodyGetVertices is not None


@pytest.mark.unit
def test_node_class_mappings():
    """Test that NODE_CLASS_MAPPINGS is properly structured."""
    from nodes.processing.load_model import NODE_CLASS_MAPPINGS as load_mappings
    from nodes.processing.process import NODE_CLASS_MAPPINGS as process_mappings
    from nodes.processing.visualize import NODE_CLASS_MAPPINGS as viz_mappings

    # All mappings should be dictionaries
    assert isinstance(load_mappings, dict)
    assert isinstance(process_mappings, dict)
    assert isinstance(viz_mappings, dict)

    # Check that mappings are not empty
    assert len(load_mappings) > 0
    assert len(process_mappings) > 0
    assert len(viz_mappings) > 0


@pytest.mark.unit
def test_node_display_name_mappings():
    """Test that NODE_DISPLAY_NAME_MAPPINGS is properly structured."""
    from nodes.processing.load_model import NODE_DISPLAY_NAME_MAPPINGS as load_names
    from nodes.processing.process import NODE_DISPLAY_NAME_MAPPINGS as process_names
    from nodes.processing.visualize import NODE_DISPLAY_NAME_MAPPINGS as viz_names

    # All mappings should be dictionaries
    assert isinstance(load_names, dict)
    assert isinstance(process_names, dict)
    assert isinstance(viz_names, dict)

    # Check that mappings are not empty
    assert len(load_names) > 0
    assert len(process_names) > 0
    assert len(viz_names) > 0


@pytest.mark.unit
def test_package_structure():
    """Test that the package structure is correct."""
    from pathlib import Path

    # Get the root directory
    root = Path(__file__).parent.parent

    # Check key directories exist
    assert (root / "nodes").exists()
    assert (root / "nodes" / "processing").exists()
    assert (root / "sam_3d_body").exists()
    assert (root / "tests").exists()

    # Check key files exist
    assert (root / "__init__.py").exists()
    assert (root / "nodes" / "__init__.py").exists()
    assert (root / "nodes" / "processing" / "__init__.py").exists()
    assert (root / "install.py").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "pyproject.toml").exists()


@pytest.mark.unit
def test_node_categories():
    """Test that nodes have appropriate categories."""
    from nodes.processing.load_model import LoadSAM3DBodyModel
    from nodes.processing.process import SAM3DBodyProcess
    from nodes.processing.visualize import SAM3DBodyExportMesh

    # All nodes should have CATEGORY attribute
    assert hasattr(LoadSAM3DBodyModel, 'CATEGORY')
    assert hasattr(SAM3DBodyProcess, 'CATEGORY')
    assert hasattr(SAM3DBodyExportMesh, 'CATEGORY')

    # Categories should be strings
    assert isinstance(LoadSAM3DBodyModel.CATEGORY, str)
    assert isinstance(SAM3DBodyProcess.CATEGORY, str)
    assert isinstance(SAM3DBodyExportMesh.CATEGORY, str)


@pytest.mark.unit
def test_node_functions_exist():
    """Test that node classes have required methods."""
    from nodes.processing.load_model import LoadSAM3DBodyModel
    from nodes.processing.process import SAM3DBodyProcess

    # Check LoadSAM3DBodyModel has required attributes
    assert hasattr(LoadSAM3DBodyModel, 'INPUT_TYPES')
    assert hasattr(LoadSAM3DBodyModel, 'RETURN_TYPES')
    assert hasattr(LoadSAM3DBodyModel, 'FUNCTION')
    assert hasattr(LoadSAM3DBodyModel, 'CATEGORY')

    # Check function methods exist
    assert hasattr(LoadSAM3DBodyModel, LoadSAM3DBodyModel.FUNCTION)
    assert hasattr(SAM3DBodyProcess, SAM3DBodyProcess.FUNCTION)
