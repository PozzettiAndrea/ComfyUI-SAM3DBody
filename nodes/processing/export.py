# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
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

            # Convert mesh bounds to plain Python types (with coordinate transform applied)
            mesh_min = vertices.min(axis=0)
            mesh_max = vertices.max(axis=0)
            if isinstance(mesh_min, np.ndarray):
                mesh_min = [float(x) for x in mesh_min]
            if isinstance(mesh_max, np.ndarray):
                mesh_max = [float(x) for x in mesh_max]
            # Apply same transform as mesh: flip all axes
            mesh_min = [-mesh_min[0], -mesh_min[1], -mesh_min[2]]
            mesh_max = [-mesh_max[0], -mesh_max[1], -mesh_max[2]]
            # Ensure min < max after flipping (signs reverse order)
            mesh_min, mesh_max = [min(mesh_min[i], mesh_max[i]) for i in range(3)], [max(mesh_min[i], mesh_max[i]) for i in range(3)]

            # Apply coordinate transform to joint positions to match mesh (flip X, Y, Z)
            joint_coords_flipped = joint_coords.copy()
            joint_coords_flipped[:, 0] = -joint_coords_flipped[:, 0]  # Flip X (demirror)
            joint_coords_flipped[:, 1] = -joint_coords_flipped[:, 1]  # Flip Y (point up)
            joint_coords_flipped[:, 2] = -joint_coords_flipped[:, 2]  # Flip Z
            print(f"[SAM3DBodyExportFBX] Applied coordinate transform to mesh and skeleton (flipped X, Y, Z)")

            skeleton_data = {
                "joint_positions": joint_coords_flipped.tolist(),
                "num_joints": len(joint_coords),
                # Add mesh vertices bounds for debugging coordinate spaces
                "mesh_vertices_bounds_min": mesh_min,
                "mesh_vertices_bounds_max": mesh_max,
            }

            # Extract skinning weights from MHR model
            print(f"[SAM3DBodyExportFBX] Extracting skinning weights from MHR model...")
            try:
                mhr_model_path = os.path.expanduser('~/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/snapshots/3a55aef9c322e36127d57755573c161baf7c1785/assets/mhr_model.pt')

                if os.path.exists(mhr_model_path):
                    mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                    lbs = mhr_model.character_torch.linear_blend_skinning

                    # Get the flattened skinning data
                    vert_indices = lbs.vert_indices_flattened.cpu().numpy().astype(int)
                    skin_indices = lbs.skin_indices_flattened.cpu().numpy().astype(int)
                    skin_weights = lbs.skin_weights_flattened.cpu().numpy().astype(float)

                    # Convert from flattened format to per-vertex format
                    # Create a dictionary mapping vertex index to list of (bone_index, weight) tuples
                    vertex_weights = {}
                    for i in range(len(vert_indices)):
                        vert_idx = int(vert_indices[i])
                        bone_idx = int(skin_indices[i])
                        weight = float(skin_weights[i])

                        if vert_idx not in vertex_weights:
                            vertex_weights[vert_idx] = []
                        vertex_weights[vert_idx].append([bone_idx, weight])

                    # Convert to list format for JSON serialization
                    # Format: list of lists, where each inner list is [bone_idx, weight] pairs for that vertex
                    skinning_data = []
                    num_vertices = len(vertices)
                    for vert_idx in range(num_vertices):
                        if vert_idx in vertex_weights:
                            skinning_data.append(vertex_weights[vert_idx])
                        else:
                            skinning_data.append([])  # No influences for this vertex

                    skeleton_data["skinning_weights"] = skinning_data

                    print(f"[SAM3DBodyExportFBX] ✓ Extracted skinning weights for {num_vertices} vertices")
                    print(f"[SAM3DBodyExportFBX] Total bone influences: {len(vert_indices)}")

                    # Show statistics
                    num_influenced = len([v for v in skinning_data if len(v) > 0])
                    print(f"[SAM3DBodyExportFBX] Vertices with bone influences: {num_influenced}/{num_vertices}")
                else:
                    print(f"[SAM3DBodyExportFBX] [WARNING] MHR model not found, skipping skinning weights")
            except Exception as e:
                print(f"[SAM3DBodyExportFBX] [WARNING] Failed to extract skinning weights: {e}")
                import traceback
                traceback.print_exc()

            # Get joint parent hierarchy from mesh_data (added by Process node)
            # The skeleton output now includes joint_parents from the MHR model
            joint_parents = None
            joint_rotations = mesh_data.get("joint_rotations")

            # Check if joint_parents is in joint_rotations (they're bundled together sometimes)
            if isinstance(joint_rotations, dict) and "joint_parents" in joint_rotations:
                joint_parents_data = joint_rotations["joint_parents"]
            else:
                joint_parents_data = mesh_data.get("joint_parents")

            if joint_parents_data is not None:
                if isinstance(joint_parents_data, np.ndarray):
                    joint_parents = joint_parents_data.astype(int).tolist()
                elif isinstance(joint_parents_data, torch.Tensor):
                    joint_parents = joint_parents_data.cpu().numpy().astype(int).tolist()
                else:
                    joint_parents = [int(p) for p in joint_parents_data]

                print(f"[SAM3DBodyExportFBX] ✓ Found {len(joint_parents)} joint parents in mesh data")
                print(f"[SAM3DBodyExportFBX] First 10 parents: {joint_parents[:10]}")
                skeleton_data["joint_parents"] = joint_parents
            else:
                # Load joint parents from MHR model if we have 127 joints
                if len(joint_coords) == 127:
                    print(f"[SAM3DBodyExportFBX] Loading MHR skeleton hierarchy from model...")
                    try:
                        # Load MHR model to extract joint parents
                        mhr_model_path = os.path.expanduser('~/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/snapshots/3a55aef9c322e36127d57755573c161baf7c1785/assets/mhr_model.pt')

                        if os.path.exists(mhr_model_path):
                            print(f"[SAM3DBodyExportFBX] Loading from: {mhr_model_path}")
                            mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                            joint_parents_tensor = mhr_model.character_torch.skeleton.joint_parents
                            joint_parents = joint_parents_tensor.cpu().numpy().astype(int).tolist()
                            skeleton_data["joint_parents"] = joint_parents
                            print(f"[SAM3DBodyExportFBX] ✓ Loaded MHR hierarchy with {len(joint_parents)} joints")
                            print(f"[SAM3DBodyExportFBX] Root joint (parent=-1) at index: {joint_parents.index(-1)}")
                        else:
                            print(f"[SAM3DBodyExportFBX] [WARNING] MHR model not found at {mhr_model_path}")
                    except Exception as e:
                        print(f"[SAM3DBodyExportFBX] [WARNING] Failed to load MHR hierarchy: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[SAM3DBodyExportFBX] No joint parent hierarchy available")

            # Add camera and other transform data if available
            camera = mesh_data.get("camera")
            focal_length = mesh_data.get("focal_length")
            if camera is not None:
                if isinstance(camera, torch.Tensor):
                    camera = camera.cpu().numpy()
                skeleton_data["camera"] = [float(x) for x in camera.flatten()] if isinstance(camera, np.ndarray) else camera
            if focal_length is not None:
                if isinstance(focal_length, (torch.Tensor, np.ndarray)):
                    focal_length = float(focal_length.item() if hasattr(focal_length, 'item') else focal_length)
                skeleton_data["focal_length"] = float(focal_length)

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

                # Always print stdout to see debug info
                if result.stdout:
                    print(f"[SAM3DBodyExportFBX] Blender stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[SAM3DBodyExportFBX] Blender stderr:\n{result.stderr}")

                if result.returncode != 0:
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
            # Write vertices (with coordinate transform to match skeleton: flip X and Z, negate Y)
            for v in vertices:
                f.write(f"v {-v[0]:.6f} {-v[1]:.6f} {-v[2]:.6f}\n")

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
