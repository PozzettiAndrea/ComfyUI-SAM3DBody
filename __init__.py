# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
ComfyUI SAM 3D Body Custom Nodes

A ComfyUI wrapper for Meta's SAM 3D Body - Robust Full-Body Human Mesh Recovery.
Provides nodes for loading models, processing images, and visualizing 3D mesh reconstructions.
"""

import sys
import os
from pathlib import Path

# Module metadata
__version__ = "1.0.0"
__author__ = "SAM 3D Body Team - Meta AI"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Add nodes directory to sys.path so vendored sam_3d_body package is importable
_custom_node_dir = Path(__file__).resolve().parent
_nodes_dir = _custom_node_dir / "nodes"
if str(_nodes_dir) not in sys.path:
    sys.path.insert(0, str(_nodes_dir))

# Only run initialization when loaded by ComfyUI, not during pytest
if 'PYTEST_CURRENT_TEST' not in os.environ:
    # Check if isolated environment exists
    _env_path = _custom_node_dir / "_env_sam3dbody"
    if _env_path.exists():
        print("[SAM3DBody] Isolated environment found")
    else:
        print("[SAM3DBody] WARNING: Isolated environment not found at _env_sam3dbody")
        print("[SAM3DBody] Run 'comfy-env install' to create the isolated environment")

    # Setup import stubs BEFORE importing nodes
    try:
        from comfy_env import setup_isolated_imports
        setup_isolated_imports(__file__)
    except ImportError:
        print("[SAM3DBody] comfy-env not installed, import stubbing disabled")
    except Exception as e:
        print(f"[SAM3DBody] Failed to setup import stubs: {e}")

    # Import node classes
    print("[SAM3DBody] Initializing ComfyUI-SAM3DBody custom nodes...")
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print(f"[SAM3DBody] [OK] Loaded {len(NODE_CLASS_MAPPINGS)} node(s)")

        for node_name in sorted(NODE_CLASS_MAPPINGS.keys()):
            print(f"[SAM3DBody]   - {node_name}")

    except Exception as e:
        print(f"[SAM3DBody] [ERROR] Failed to import nodes: {e}")
        import traceback
        traceback.print_exc()
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    # Enable process isolation - all nodes run in pixi env (Python 3.11)
    try:
        from comfy_env import enable_isolation
        enable_isolation(NODE_CLASS_MAPPINGS)
    except ImportError:
        pass
    except Exception as e:
        print(f"[SAM3DBody] Failed to enable isolation: {e}")

    # Import server routes
    try:
        from .nodes import server
        print("[SAM3DBody] [OK] Server routes registered")
    except Exception as e:
        print(f"[SAM3DBody] [WARNING] Failed to register server routes: {e}")

    print("[SAM3DBody] Initialization complete")
else:
    print("[SAM3DBody] Running in pytest mode, skipping node initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for UI components
WEB_DIRECTORY = "./web"
