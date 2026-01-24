# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Preview node for SAM 3D Body rigged meshes.

Displays rigged FBX files with interactive skeleton manipulation.
"""

import os
import folder_paths


class SAM3DBodyPreviewRiggedMesh:
    """
    Preview rigged mesh with interactive FBX viewer.

    Displays the rigged FBX in a Three.js viewer with skeleton visualization
    and interactive bone manipulation controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_output_path": ("STRING", {
                    "tooltip": "FBX filename from output directory (from SAM3D export node)"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SAM3DBody/visualization"

    def preview(self, fbx_output_path):
        """Preview the rigged mesh in an interactive FBX viewer."""
        print(f"[SAM3DBodyPreviewRiggedMesh] Preparing preview...")

        fbx_path = fbx_output_path
        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found: {fbx_path}")

        print(f"[SAM3DBodyPreviewRiggedMesh] FBX path: {fbx_path}")

        # FBX is already in output, so viewer can access it directly
        # Assume all FBX files have skinning and skeleton
        has_skinning = True
        has_skeleton = True

        print(f"[SAM3DBodyPreviewRiggedMesh] Has skinning: {has_skinning}")
        print(f"[SAM3DBodyPreviewRiggedMesh] Has skeleton: {has_skeleton}")

        return {
            "ui": {
                "fbx_file": [fbx_output_path],
                "has_skinning": [bool(has_skinning)],
                "has_skeleton": [bool(has_skeleton)],
            }
        }


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPreviewRiggedMesh": SAM3DBodyPreviewRiggedMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPreviewRiggedMesh": "SAM 3D Body: Preview Rigged Mesh",
}
