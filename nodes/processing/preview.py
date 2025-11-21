# Copyright (c) Meta Platforms, Inc. and affiliates.
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

        # FBX should already be in output directory
        output_dir = folder_paths.get_output_directory()
        fbx_path = os.path.join(output_dir, fbx_output_path)

        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found in output directory: {fbx_output_path}")

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


class SAM3DBodyPreviewSkeleton:
    """
    Preview skeleton (bones only, no mesh) with interactive 3D viewer.

    Displays skeleton joints and bones in a Three.js viewer with
    rotation, zoom, and pan controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON", {
                    "tooltip": "Skeleton data from SAM3D Body"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SAM3DBody/visualization"

    def preview(self, skeleton):
        """Preview skeleton in interactive 3D viewer."""
        import json
        import numpy as np
        import torch

        print(f"[SAM3DBodyPreviewSkeleton] Preparing skeleton preview...")

        # Get joint positions
        joint_positions = skeleton.get("joint_positions")

        if joint_positions is None:
            raise RuntimeError("Skeleton has no joint_positions data")

        # Convert to numpy if needed
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.cpu().numpy()

        # Convert to list for JSON serialization
        joint_positions_list = joint_positions.tolist()

        # Get joint rotations if available
        joint_rotations = skeleton.get("joint_rotations")
        joint_rotations_list = None

        if joint_rotations is not None:
            if isinstance(joint_rotations, torch.Tensor):
                joint_rotations = joint_rotations.cpu().numpy()
            joint_rotations_list = joint_rotations.tolist()

        # Save skeleton data to temporary JSON file for viewer
        output_dir = folder_paths.get_output_directory()
        skeleton_json_path = os.path.join(output_dir, "_temp_skeleton_preview.json")

        skeleton_data = {
            "joint_positions": joint_positions_list,
            "joint_rotations": joint_rotations_list,
            "num_joints": len(joint_positions_list),
        }

        with open(skeleton_json_path, 'w') as f:
            json.dump(skeleton_data, f)

        print(f"[SAM3DBodyPreviewSkeleton] Prepared skeleton with {len(joint_positions_list)} joints")

        return {
            "ui": {
                "skeleton_file": ["_temp_skeleton_preview.json"],
                "num_joints": [len(joint_positions_list)],
            }
        }


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPreviewRiggedMesh": SAM3DBodyPreviewRiggedMesh,
    "SAM3DBodyPreviewSkeleton": SAM3DBodyPreviewSkeleton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPreviewRiggedMesh": "SAM 3D Body: Preview Rigged Mesh",
    "SAM3DBodyPreviewSkeleton": "SAM 3D Body: Preview Skeleton",
}
