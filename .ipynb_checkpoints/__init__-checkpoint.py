# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""ComfyUI SAM 3D Body - Robust Full-Body Human Mesh Recovery."""

import sys
print("[geompack] loading...", file=sys.stderr, flush=True)
from comfy_env import register_nodes
print("[geompack] calling register_nodes", file=sys.stderr, flush=True
      
# Server routes for API endpoints
try:
    from .nodes import server
except Exception:
    pass

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
