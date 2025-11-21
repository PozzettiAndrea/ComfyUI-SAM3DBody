# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Export nodes for SAM 3D Body meshes.

Exports meshes with rigging data to various formats.
"""

import os
import json
import time
import tempfile
import subprocess
import numpy as np
import torch
import folder_paths

from ..base import BLENDER_EXE, BLENDER_SCRIPT
from ...constants import BLENDER_TIMEOUT


class SAM3DBodyExportFBX:
    """
    Export SAM3D Body mesh with skeleton to FBX format.

    Takes mesh data from SAM3D and exports it as a rigged FBX file
    using Blender for format conversion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3D Body Process node"
                }),
                "output_filename": ("STRING", {
                    "default": "sam3d_rigged.fbx",
                    "tooltip": "Output filename for the FBX file"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody/export"
    OUTPUT_NODE = True

    def export_fbx(self, mesh_data, output_filename):
        """Export mesh with skeleton to FBX format."""
        print(f"[SAM3DBodyExportFBX] Exporting to FBX...")

        # Extract mesh data
        vertices = mesh_data.get("vertices")
        faces = mesh_data.get("faces")
        joint_coords = mesh_data.get("joint_coords")  # 127 joints

        if vertices is None or faces is None:
            raise RuntimeError("Mesh vertices or faces not found in mesh_data")

        # Convert tensors to numpy if needed
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if joint_coords is not None and isinstance(joint_coords, torch.Tensor):
            joint_coords = joint_coords.cpu().numpy()

        print(f"[SAM3DBodyExportFBX] Mesh: {len(vertices)} vertices, {len(faces)} faces")
        if joint_coords is not None:
            print(f"[SAM3DBodyExportFBX] Skeleton: {len(joint_coords)} joints")

        # Prepare output path
        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'
        output_fbx_path = os.path.join(output_dir, output_filename)

        # Create a simple OBJ file first (Blender can import this easily)
        temp_dir = folder_paths.get_temp_directory()
        temp_obj_path = os.path.join(temp_dir, f"temp_mesh_{int(time.time())}.obj")

        # Write OBJ file
        self._write_obj_file(temp_obj_path, vertices, faces)
        print(f"[SAM3DBodyExportFBX] Wrote temporary OBJ: {temp_obj_path}")

        # Save skeleton data if available
        skeleton_json_path = None
        if joint_coords is not None:
            skeleton_json_path = os.path.join(temp_dir, f"skeleton_{int(time.time())}.json")
            skeleton_data = {
                "joint_positions": joint_coords.tolist(),
                "num_joints": len(joint_coords),
            }
            with open(skeleton_json_path, 'w') as f:
                json.dump(skeleton_data, f)
            print(f"[SAM3DBodyExportFBX] Saved skeleton data: {skeleton_json_path}")

        try:
            # Use Blender to convert OBJ to FBX (if available)
            if BLENDER_EXE and os.path.exists(BLENDER_EXE):
                print(f"[SAM3DBodyExportFBX] Using Blender: {BLENDER_EXE}")

                if not os.path.exists(BLENDER_SCRIPT):
                    raise RuntimeError(f"Blender export script not found: {BLENDER_SCRIPT}")

                # Run Blender with export script
                cmd = [
                    BLENDER_EXE,
                    '--background',
                    '--python', BLENDER_SCRIPT,
                    '--',
                    temp_obj_path,
                    output_fbx_path,
                ]

                if skeleton_json_path:
                    cmd.append(skeleton_json_path)

                print(f"[SAM3DBodyExportFBX] Running Blender export...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)

                if result.returncode != 0:
                    print(f"[SAM3DBodyExportFBX] Blender stderr: {result.stderr}")
                    print(f"[SAM3DBodyExportFBX] Blender stdout: {result.stdout}")
                    raise RuntimeError(f"Blender export failed with return code {result.returncode}")

            else:
                # Fallback: just copy the OBJ to output with .obj extension
                print("[SAM3DBodyExportFBX] [WARNING] Blender not found, exporting as OBJ instead")
                fallback_path = output_fbx_path.replace('.fbx', '.obj')
                import shutil
                shutil.copy(temp_obj_path, fallback_path)
                output_fbx_path = fallback_path
                print(f"[SAM3DBodyExportFBX] Saved as OBJ: {fallback_path}")

            if not os.path.exists(output_fbx_path):
                raise RuntimeError(f"Export completed but output file not found: {output_fbx_path}")

            print(f"[SAM3DBodyExportFBX] ✓ Successfully exported to: {output_fbx_path}")

            return (os.path.basename(output_fbx_path),)

        finally:
            # Clean up temporary files
            if os.path.exists(temp_obj_path):
                os.unlink(temp_obj_path)
            if skeleton_json_path and os.path.exists(skeleton_json_path):
                os.unlink(skeleton_json_path)

    def _write_obj_file(self, filepath, vertices, faces):
        """Write mesh to OBJ file format."""
        with open(filepath, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyExportFBX": SAM3DBodyExportFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyExportFBX": "SAM 3D Body: Export FBX",
}
