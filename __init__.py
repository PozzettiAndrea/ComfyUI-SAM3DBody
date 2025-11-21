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

# Add this directory to sys.path so vendored sam_3d_body package is importable
_custom_node_dir = Path(__file__).parent
if str(_custom_node_dir) not in sys.path:
    sys.path.insert(0, str(_custom_node_dir))

# Initialization tracking
INIT_SUCCESS = False
INIT_ERRORS = []

# Robust pytest detection with override capability
force_init = os.environ.get('SAM3DB_FORCE_INIT') == '1'
is_pytest = (
    'PYTEST_CURRENT_TEST' in os.environ or
    '_pytest' in sys.modules or
    'pytest' in sys.modules
)
skip_init = is_pytest and not force_init

if skip_init:
    print("[SAM3DBody] Running in pytest mode, skipping node initialization")
    print("[SAM3DBody] Set SAM3DB_FORCE_INIT=1 to force initialization in pytest")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    # Step 1: Import node classes
    print("[SAM3DBody] Initializing ComfyUI-SAM3DBody custom nodes...")
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print(f"[SAM3DBody] [OK] Loaded {len(NODE_CLASS_MAPPINGS)} node(s)")

        # List available nodes
        for node_name in sorted(NODE_CLASS_MAPPINGS.keys()):
            print(f"[SAM3DBody]   - {node_name}")

        INIT_SUCCESS = True

    except Exception as e:
        print(f"[SAM3DBody] [ERROR] Failed to import nodes: {e}")
        INIT_ERRORS.append(f"Node import failed: {e}")
        import traceback
        traceback.print_exc()
        # Provide empty mappings so ComfyUI can still load
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    # Step 2: Additional initialization (if needed)
    if INIT_SUCCESS:
        print("[SAM3DBody] Initialization complete")
    else:
        print("[SAM3DBody] [WARNING] Initialization completed with errors")
        print("[SAM3DBody] Check the error messages above for details")
        print("[SAM3DBody] If you see import errors, verify the installation:")
        print("[SAM3DBody]   1. Run install.py if not already done")
        print("[SAM3DBody]   2. Check that sam_3d_body/ directory has all required files")
        print("[SAM3DBody]   3. Verify requirements.txt dependencies are installed")

# Web directory (if web UI components are added in future)
WEB_DIRECTORY = "./web"
