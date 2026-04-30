# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Preview node for SAM 3D Body rigged meshes.

Displays rigged FBX files with interactive skeleton manipulation.
"""

import logging
import os
import folder_paths
from comfy_api.latest import io

log = logging.getLogger("sam3dbody")


class SAM3DBodyPreviewRiggedMesh(io.ComfyNode):
    """
    Preview rigged mesh with interactive FBX viewer.

    Displays the rigged FBX in a Three.js viewer with skeleton visualization
    and interactive bone manipulation controls.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBodyPreviewRiggedMesh",
            display_name="SAM 3D Body: Preview Rigged Mesh",
            category="SAM3DBody/visualization",
            is_output_node=True,
            inputs=[
                io.String.Input("fbx_output_path",
                                tooltip="FBX filename from output directory (from SAM3D export node)"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, fbx_output_path):
        """Preview the rigged mesh in an interactive FBX viewer."""
        log.info(f" Preparing preview...")

        fbx_path = fbx_output_path
        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found: {fbx_path}")

        log.info(f" FBX path: {fbx_path}")

        # FBX is already in output, so viewer can access it directly
        # Assume all FBX files have skinning and skeleton
        has_skinning = True
        has_skeleton = True

        log.info(f" Has skinning: {has_skinning}")
        log.info(f" Has skeleton: {has_skeleton}")

        return io.NodeOutput(
            ui={
                "fbx_file": [fbx_output_path],
                "has_skinning": [bool(has_skinning)],
                "has_skeleton": [bool(has_skeleton)],
            }
        )


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPreviewRiggedMesh": SAM3DBodyPreviewRiggedMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPreviewRiggedMesh": "SAM 3D Body: Preview Rigged Mesh",
}
