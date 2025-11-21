# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
ComfyUI SAM 3D Body Custom Nodes

A ComfyUI wrapper for Meta's SAM 3D Body - Robust Full-Body Human Mesh Recovery.
Provides nodes for loading models, processing images, and visualizing 3D mesh reconstructions.
"""

import sys
from pathlib import Path

# Add this directory to sys.path so vendored sam_3d_body package is importable
_custom_node_dir = Path(__file__).parent
if str(_custom_node_dir) not in sys.path:
    sys.path.insert(0, str(_custom_node_dir))

# Pytest guard - only load nodes when running in ComfyUI
if 'pytest' not in sys.modules:
    try:
        # Import node mappings
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        print("[SAM3DBody] ComfyUI-SAM3DBody nodes loaded successfully")

    except Exception as e:
        print(f"[SAM3DBody] [ERROR] Failed to load nodes: {e}")
        import traceback
        traceback.print_exc()
        # Provide empty mappings so ComfyUI can still load
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    # Testing mode - provide dummy mappings
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Module metadata
__version__ = "1.0.0"
__author__ = "SAM 3D Body Team - Meta AI"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
