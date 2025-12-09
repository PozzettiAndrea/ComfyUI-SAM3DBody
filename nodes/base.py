# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
#
# Blender auto-detection logic adapted from:
# ComfyUI-UniRig by Tsinghua University and Tripo AI
# https://github.com/VAST-AI-Research/UniRig
# Licensed under MIT License
"""
Base utilities for ComfyUI SAM 3D Body nodes.

Provides conversion functions between ComfyUI tensor formats and other formats
(PIL, numpy, OpenCV) commonly used in computer vision tasks.
Also handles Blender auto-detection and installation for FBX export.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import folder_paths


def comfy_image_to_pil(image):
    """
    Convert ComfyUI image tensor to PIL Image.

    Args:
        image: ComfyUI image tensor [B, H, W, C] in range [0, 1]

    Returns:
        PIL Image in RGB format
    """
    # Take first image from batch
    img_np = image[0].cpu().numpy()
    # Convert from [0, 1] to [0, 255]
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def pil_to_comfy_image(pil_image):
    """
    Convert PIL Image to ComfyUI image tensor.

    Args:
        pil_image: PIL Image in RGB format

    Returns:
        ComfyUI image tensor [1, H, W, C] in range [0, 1]
    """
    # Convert to numpy and normalize
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    # Add batch dimension: [H, W, C] -> [1, H, W, C]
    return torch.from_numpy(img_np).unsqueeze(0)


def comfy_image_to_numpy(image):
    """
    Convert ComfyUI image tensor to numpy array in BGR format (OpenCV format).

    Args:
        image: ComfyUI image tensor [B, H, W, C] in range [0, 1]

    Returns:
        numpy array in BGR format [H, W, C] in range [0, 255]
    """
    # Take first image from batch
    img_np = image[0].cpu().numpy()
    # Convert from [0, 1] to [0, 255]
    img_np = (img_np * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    img_bgr = img_np[..., ::-1].copy()
    return img_bgr


def numpy_to_comfy_image(np_image):
    """
    Convert numpy array in BGR format to ComfyUI image tensor.

    Args:
        np_image: numpy array in BGR format [H, W, C] in range [0, 255]

    Returns:
        ComfyUI image tensor [1, H, W, C] in range [0, 1]
    """
    # Convert BGR to RGB
    img_rgb = np_image[..., ::-1].copy()
    # Normalize to [0, 1]
    img_rgb = img_rgb.astype(np.float32) / 255.0
    # Add batch dimension
    return torch.from_numpy(img_rgb).unsqueeze(0)


def comfy_mask_to_numpy(mask):
    """
    Convert ComfyUI mask to numpy array.

    Args:
        mask: ComfyUI mask tensor [N, H, W] in range [0, 1]

    Returns:
        numpy array [N, H, W] in range [0, 1]
    """
    return mask.cpu().numpy()


def numpy_to_comfy_mask(np_mask):
    """
    Convert numpy array to ComfyUI mask.

    Args:
        np_mask: numpy array [H, W] or [N, H, W] in range [0, 1]

    Returns:
        ComfyUI mask tensor [N, H, W] in range [0, 1]
    """
    # Add batch dimension if needed
    if np_mask.ndim == 2:
        np_mask = np.expand_dims(np_mask, axis=0)
    return torch.from_numpy(np_mask.astype(np.float32))


def vertices_to_point_cloud(vertices):
    """
    Convert mesh vertices to point cloud format for visualization.

    Args:
        vertices: numpy array of shape [N, 3] representing mesh vertices

    Returns:
        Dictionary containing point cloud data
    """
    return {
        "points": vertices,
        "colors": None,  # Can be added if vertex colors are available
    }


# ============================================================================
# Blender Auto-Detection and Installation
# ============================================================================

# Get paths relative to this file
NODE_DIR = Path(__file__).parent.parent.absolute()  # Go up from nodes/ to ComfyUI-SAM3DBody/
LIB_DIR = NODE_DIR / "lib"
BLENDER_SCRIPT = str(LIB_DIR / "blender_export_sam3d_fbx.py")
BLENDER_MULTI_SCRIPT = str(LIB_DIR / "blender_export_multi.py")

# Find Blender executable
# Priority: 1) SAM3DBODY_BLENDER_PATH env var, 2) system blender, 3) bundled blender, 4) auto-install
BLENDER_DIR = LIB_DIR / "blender"
BLENDER_EXE = None


def find_system_blender():
    """Find Blender installed on the system."""
    import shutil

    # Check PATH first
    blender_on_path = shutil.which("blender")
    if blender_on_path:
        return blender_on_path

    # Common install locations by platform
    plat = sys.platform
    candidates = []

    if plat == "darwin":  # macOS
        candidates = [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
        ]
    elif plat == "win32":  # Windows
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        candidates = [
            Path(program_files) / "Blender Foundation" / "Blender 4.2" / "blender.exe",
            Path(program_files) / "Blender Foundation" / "Blender 4.1" / "blender.exe",
            Path(program_files) / "Blender Foundation" / "Blender 4.0" / "blender.exe",
            Path(program_files_x86) / "Blender Foundation" / "Blender 4.2" / "blender.exe",
        ]
    else:  # Linux
        candidates = [
            Path("/usr/bin/blender"),
            Path("/usr/local/bin/blender"),
            Path("/snap/bin/blender"),
            Path.home() / ".local/bin/blender",
        ]

    for candidate in candidates:
        if Path(candidate).is_file():
            return str(candidate)

    return None


# Check for user-specified Blender path first
custom_blender = os.environ.get('SAM3DBODY_BLENDER_PATH', '')
if custom_blender and os.path.isfile(custom_blender):
    BLENDER_EXE = custom_blender
    print(f"[SAM3DBody] Using custom Blender: {BLENDER_EXE}")
else:
    # Try to find system-installed Blender
    system_blender = find_system_blender()
    if system_blender:
        BLENDER_EXE = system_blender
        print(f"[SAM3DBody] Found system Blender: {BLENDER_EXE}")

if not BLENDER_EXE and BLENDER_DIR.exists():
    # Support both relative imports (ComfyUI) and absolute imports (testing)
    try:
        from ..install import find_blender_executable
    except ImportError:
        from install import find_blender_executable
    blender_bin = find_blender_executable(str(BLENDER_DIR))
    if blender_bin:
        BLENDER_EXE = str(blender_bin)
        os.environ['BLENDER_EXE'] = BLENDER_EXE
        print(f"[SAM3DBody] Found Blender: {BLENDER_EXE}")

# Install Blender if not found (unless disabled via env var)
SKIP_BLENDER_INSTALL = os.environ.get('SAM3DBODY_SKIP_BLENDER_INSTALL', '0') == '1'

if not BLENDER_EXE and not SKIP_BLENDER_INSTALL:
    print("[SAM3DBody] Blender not found, installing...")
    try:
        # Import from parent package
        sys.path.insert(0, str(NODE_DIR))
        try:
            from ..install import install_blender
        except ImportError:
            from install import install_blender
        BLENDER_EXE = install_blender(target_dir=BLENDER_DIR)
        if BLENDER_EXE:
            os.environ['BLENDER_EXE'] = BLENDER_EXE
            print(f"[SAM3DBody] Blender installed: {BLENDER_EXE}")
        else:
            print("[SAM3DBody] Warning: Blender installation failed")
    except Exception as e:
        print(f"[SAM3DBody] Warning: Could not install Blender: {e}")
elif not BLENDER_EXE and SKIP_BLENDER_INSTALL:
    print("[SAM3DBody] Blender not found and installation is disabled (SAM3DBODY_SKIP_BLENDER_INSTALL=1)")
    print("[SAM3DBody] FBX export will not be available")
