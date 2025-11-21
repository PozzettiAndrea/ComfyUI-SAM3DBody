"""
Integration tests for ComfyUI-SAM3DBody inference.

These tests verify that the model can actually load and run inference.
Requires HF_TOKEN environment variable for downloading models.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import os


@pytest.fixture
def test_image():
    """Create a simple test image tensor."""
    # Create a 512x512 RGB image (ComfyUI format: BHWC)
    image = np.random.rand(1, 512, 512, 3).astype(np.float32)
    return torch.from_numpy(image)


@pytest.fixture
def test_image_from_file():
    """Load test image from assets directory if available."""
    assets_dir = Path(__file__).parent.parent / "assets"
    test_images = list(assets_dir.glob("*.jpg")) + list(assets_dir.glob("*.png"))

    if test_images:
        from PIL import Image
        img = Image.open(test_images[0]).convert('RGB')
        # Resize to reasonable size for testing
        img = img.resize((512, 512))
        # Convert to ComfyUI format (BHWC, normalized to 0-1)
        img_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array[None, ...])
    else:
        # Fall back to random image
        image = np.random.rand(1, 512, 512, 3).astype(np.float32)
        return torch.from_numpy(image)


@pytest.mark.integration
def test_load_model_cpu():
    """Test loading model on CPU."""
    from nodes.processing.load_model import LoadSAM3DBodyModel

    loader = LoadSAM3DBodyModel()

    # Check if HF_TOKEN is available
    hf_token = os.environ.get('HF_TOKEN', '')
    if not hf_token:
        pytest.skip("HF_TOKEN not set, skipping model download test")

    # Test loading from HuggingFace
    result = loader.load_model(
        model_source="huggingface",
        model_path="facebook/sam-3d-body-vith",  # Use smaller model for testing
        device="cpu",
        hf_token=hf_token,
        mhr_path=None
    )

    # Should return a tuple with model
    assert isinstance(result, tuple)
    assert len(result) == 1
    model = result[0]
    assert model is not None


@pytest.mark.integration
def test_model_caching():
    """Test that model caching works correctly."""
    from nodes.processing.load_model import LoadSAM3DBodyModel, _MODEL_CACHE

    # Clear cache
    _MODEL_CACHE.clear()

    loader = LoadSAM3DBodyModel()

    hf_token = os.environ.get('HF_TOKEN', '')
    if not hf_token:
        pytest.skip("HF_TOKEN not set, skipping cache test")

    # Load model twice
    result1 = loader.load_model(
        model_source="huggingface",
        model_path="facebook/sam-3d-body-vith",
        device="cpu",
        hf_token=hf_token,
        mhr_path=None
    )

    cache_size_after_first = len(_MODEL_CACHE)

    result2 = loader.load_model(
        model_source="huggingface",
        model_path="facebook/sam-3d-body-vith",
        device="cpu",
        hf_token=hf_token,
        mhr_path=None
    )

    cache_size_after_second = len(_MODEL_CACHE)

    # Cache should have one entry
    assert cache_size_after_first == 1
    # Cache size should not increase on second load
    assert cache_size_after_second == 1
    # Both results should be the same object (cached)
    assert result1[0] is result2[0]


@pytest.mark.integration
def test_process_image_cpu(test_image):
    """Test processing an image on CPU."""
    from nodes.processing.load_model import LoadSAM3DBodyModel
    from nodes.processing.process import SAM3DBodyProcess

    hf_token = os.environ.get('HF_TOKEN', '')
    if not hf_token:
        pytest.skip("HF_TOKEN not set, skipping inference test")

    # Load model
    loader = LoadSAM3DBodyModel()
    model_result = loader.load_model(
        model_source="huggingface",
        model_path="facebook/sam-3d-body-vith",
        device="cpu",
        hf_token=hf_token,
        mhr_path=None
    )
    model = model_result[0]

    # Process image
    processor = SAM3DBodyProcess()
    result = processor.process(
        model=model,
        image=test_image,
        bbox_threshold=0.8,
        inference_type="body",  # Use body-only for faster testing
        mask=None
    )

    # Should return tuple with mesh data
    assert isinstance(result, tuple)
    assert len(result) >= 1

    mesh_data = result[0]

    # Validate mesh_data structure
    assert isinstance(mesh_data, dict)
    # Check that mesh_data has some content
    assert len(mesh_data) > 0


@pytest.mark.integration
def test_export_mesh(test_image, tmp_path):
    """Test exporting mesh to file."""
    from nodes.processing.load_model import LoadSAM3DBodyModel
    from nodes.processing.process import SAM3DBodyProcess
    from nodes.processing.visualize import SAM3DBodyExportMesh

    hf_token = os.environ.get('HF_TOKEN', '')
    if not hf_token:
        pytest.skip("HF_TOKEN not set, skipping export test")

    # Load model
    loader = LoadSAM3DBodyModel()
    model_result = loader.load_model(
        model_source="huggingface",
        model_path="facebook/sam-3d-body-vith",
        device="cpu",
        hf_token=hf_token,
        mhr_path=None
    )
    model = model_result[0]

    # Process image
    processor = SAM3DBodyProcess()
    result = processor.process(
        model=model,
        image=test_image,
        bbox_threshold=0.8,
        inference_type="body",
        mask=None
    )
    mesh_data = result[0]

    # Export mesh
    exporter = SAM3DBodyExportMesh()
    output_dir = str(tmp_path)
    result = exporter.export_mesh(
        mesh_data=mesh_data,
        filename="test_mesh.obj",
        output_dir=output_dir
    )

    # Should return file path
    assert isinstance(result, tuple)
    assert len(result) == 1
    file_path = result[0]

    # File should exist
    assert Path(file_path).exists()
    assert file_path.endswith(".obj")


@pytest.mark.integration
def test_get_vertices(test_image):
    """Test getting mesh vertices."""
    from nodes.processing.load_model import LoadSAM3DBodyModel
    from nodes.processing.process import SAM3DBodyProcess
    from nodes.processing.visualize import SAM3DBodyGetVertices

    hf_token = os.environ.get('HF_TOKEN', '')
    if not hf_token:
        pytest.skip("HF_TOKEN not set, skipping vertices test")

    # Load model
    loader = LoadSAM3DBodyModel()
    model_result = loader.load_model(
        model_source="huggingface",
        model_path="facebook/sam-3d-body-vith",
        device="cpu",
        hf_token=hf_token,
        mhr_path=None
    )
    model = model_result[0]

    # Process image
    processor = SAM3DBodyProcess()
    result = processor.process(
        model=model,
        image=test_image,
        bbox_threshold=0.8,
        inference_type="body",
        mask=None
    )
    mesh_data = result[0]

    # Get vertices
    vertices_node = SAM3DBodyGetVertices()
    result = vertices_node.get_vertices(mesh_data=mesh_data)

    # Should return vertices
    assert isinstance(result, tuple)
    assert len(result) >= 1
    # Check that we got some output
    assert result[0] is not None
