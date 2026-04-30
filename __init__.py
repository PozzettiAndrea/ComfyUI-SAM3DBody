# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""ComfyUI SAM 3D Body - Robust Full-Body Human Mesh Recovery."""

from comfy_env import register_nodes

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
