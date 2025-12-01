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
import glob

from ..base import BLENDER_EXE, BLENDER_SCRIPT, BLENDER_MULTI_SCRIPT
from ...constants import BLENDER_TIMEOUT


def find_mhr_model_path(mesh_data=None):
    """
    Find the MHR model path using multiple fallback strategies.

    Args:
        mesh_data: Optional mesh_data dict that may contain mhr_path

    Returns:
        str: Path to mhr_model.pt or None if not found
    """
    # Strategy 1: Check mesh_data for explicitly provided path
    if mesh_data and mesh_data.get("mhr_path"):
        mhr_path = mesh_data["mhr_path"]
        if os.path.exists(mhr_path):
            return mhr_path

    # Strategy 2: Check environment variable
    env_path = os.environ.get("SAM3D_MHR_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path

    # Strategy 3: Search ComfyUI models/sam3dbody/ folder
    sam3dbody_dir = os.path.join(folder_paths.models_dir, "sam3dbody", "assets", "mhr_model.pt")
    if os.path.exists(sam3dbody_dir):
        return sam3dbody_dir

    # Strategy 4 (legacy): Search HuggingFace cache for backwards compatibility
    hf_cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3")
    if os.path.exists(hf_cache_base):
        pattern = os.path.join(hf_cache_base, "snapshots", "*", "assets", "mhr_model.pt")
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]

    return None


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
            joint_coords_flipped[:, 0] = -joint_coords_flipped[:, 0]
            joint_coords_flipped[:, 1] = -joint_coords_flipped[:, 1]
            joint_coords_flipped[:, 2] = -joint_coords_flipped[:, 2]

            skeleton_data = {
                "joint_positions": joint_coords_flipped.tolist(),
                "num_joints": len(joint_coords),
                "mesh_vertices_bounds_min": mesh_min,
                "mesh_vertices_bounds_max": mesh_max,
            }

            # Extract skinning weights from MHR model
            try:
                mhr_model_path = find_mhr_model_path(mesh_data)

                if mhr_model_path and os.path.exists(mhr_model_path):
                    mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                    lbs = mhr_model.character_torch.linear_blend_skinning

                    vert_indices = lbs.vert_indices_flattened.cpu().numpy().astype(int)
                    skin_indices = lbs.skin_indices_flattened.cpu().numpy().astype(int)
                    skin_weights = lbs.skin_weights_flattened.cpu().numpy().astype(float)

                    vertex_weights = {}
                    for i in range(len(vert_indices)):
                        vert_idx = int(vert_indices[i])
                        bone_idx = int(skin_indices[i])
                        weight = float(skin_weights[i])

                        if vert_idx not in vertex_weights:
                            vertex_weights[vert_idx] = []
                        vertex_weights[vert_idx].append([bone_idx, weight])

                    skinning_data = []
                    num_vertices = len(vertices)
                    for vert_idx in range(num_vertices):
                        if vert_idx in vertex_weights:
                            skinning_data.append(vertex_weights[vert_idx])
                        else:
                            skinning_data.append([])

                    skeleton_data["skinning_weights"] = skinning_data
            except Exception:
                pass  # Skip skinning weights if extraction fails

            # Get joint parent hierarchy from mesh_data
            joint_parents = None
            joint_rotations = mesh_data.get("joint_rotations")

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
                skeleton_data["joint_parents"] = joint_parents
            else:
                # Load joint parents from MHR model if we have 127 joints
                if len(joint_coords) == 127:
                    try:
                        mhr_model_path = find_mhr_model_path(mesh_data)
                        if mhr_model_path and os.path.exists(mhr_model_path):
                            mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                            joint_parents_tensor = mhr_model.character_torch.skeleton.joint_parents
                            joint_parents = joint_parents_tensor.cpu().numpy().astype(int).tolist()
                            skeleton_data["joint_parents"] = joint_parents
                    except Exception:
                        pass

            # Add camera and focal length if available
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

        try:
            # Use Blender to convert OBJ to FBX
            if BLENDER_EXE and os.path.exists(BLENDER_EXE):
                if not os.path.exists(BLENDER_SCRIPT):
                    raise RuntimeError(f"Blender export script not found: {BLENDER_SCRIPT}")

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

                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=BLENDER_TIMEOUT)

                if result.returncode != 0:
                    raise RuntimeError(f"Blender export failed with return code {result.returncode}")

            else:
                # Fallback: just copy the OBJ to output with .obj extension
                fallback_path = output_fbx_path.replace('.fbx', '.obj')
                import shutil
                shutil.copy(temp_obj_path, fallback_path)
                output_fbx_path = fallback_path

            if not os.path.exists(output_fbx_path):
                raise RuntimeError(f"Export completed but output file not found: {output_fbx_path}")

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
            for v in vertices:
                f.write(f"v {-v[0]:.6f} {-v[1]:.6f} {-v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


class SAM3DBodyExportMultipleFBX:
    """
    Export multiple SAM3D Body meshes with skeletons to a single FBX file.

    Takes multi-person mesh data and exports all meshes with their armatures
    into a single combined FBX file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multi_mesh_data": ("SAM3D_MULTI_OUTPUT", {
                    "tooltip": "Multi-person mesh data from SAM3D Body Process Multiple node"
                }),
                "output_filename": ("STRING", {
                    "default": "sam3d_multi_rigged.fbx",
                    "tooltip": "Output filename for the combined FBX file"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "export_multi_fbx"
    CATEGORY = "SAM3DBody/export"
    OUTPUT_NODE = True

    def export_multi_fbx(self, multi_mesh_data, output_filename):
        """Export all meshes with skeletons to a single combined FBX file."""

        num_people = multi_mesh_data.get("num_people", 0)
        people = multi_mesh_data.get("people", [])
        faces = multi_mesh_data.get("faces")

        print(f"[SAM3D Export] num_people from data: {num_people}")
        print(f"[SAM3D Export] actual people list length: {len(people)}")

        if num_people == 0 or len(people) == 0:
            raise RuntimeError("No mesh data to export")

        # Setup output path
        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'
        output_fbx_path = os.path.join(output_dir, output_filename)

        # Find MHR model path for skinning weights
        mhr_model_path = find_mhr_model_path(multi_mesh_data)

        # Load skinning data once (same for all people)
        skinning_data = None
        joint_parents = None
        if mhr_model_path and os.path.exists(mhr_model_path):
            try:
                mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                lbs = mhr_model.character_torch.linear_blend_skinning

                vert_indices = lbs.vert_indices_flattened.cpu().numpy().astype(int)
                skin_indices = lbs.skin_indices_flattened.cpu().numpy().astype(int)
                skin_weights = lbs.skin_weights_flattened.cpu().numpy().astype(float)

                vertex_weights = {}
                for j in range(len(vert_indices)):
                    vert_idx = int(vert_indices[j])
                    bone_idx = int(skin_indices[j])
                    weight = float(skin_weights[j])
                    if vert_idx not in vertex_weights:
                        vertex_weights[vert_idx] = []
                    vertex_weights[vert_idx].append([bone_idx, weight])

                # Get joint parents
                joint_parents = mhr_model.character_torch.skeleton.joint_parents.cpu().numpy().astype(int).tolist()
            except Exception:
                pass

        # Build combined data structure for all people
        temp_files = []
        combined_data = {
            "output_path": output_fbx_path,
            "people": [],
        }

        try:
            for i, person in enumerate(people):
                vertices = person.get("pred_vertices")
                joint_coords = person.get("pred_joint_coords")
                cam_t = person.get("pred_cam_t")  # Camera translation for world positioning

                if vertices is None:
                    continue

                # Convert to numpy
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                if joint_coords is not None and isinstance(joint_coords, torch.Tensor):
                    joint_coords = joint_coords.cpu().numpy()
                if cam_t is not None and isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()

                # Apply world position offset from camera translation
                if cam_t is not None:
                    vertices = vertices + cam_t  # Broadcast adds cam_t to each vertex
                    if joint_coords is not None:
                        joint_coords = joint_coords + cam_t

                # Write OBJ file for this person
                temp_obj = tempfile.NamedTemporaryFile(suffix=f'_person{i}.obj', delete=False)
                temp_files.append(temp_obj.name)
                self._write_obj_file(temp_obj.name, vertices, faces)

                # Prepare skeleton data
                skeleton_info = {}
                if joint_coords is not None:
                    # Apply coordinate transform to joint positions
                    joint_coords_flipped = joint_coords.copy()
                    joint_coords_flipped[:, 0] = -joint_coords_flipped[:, 0]
                    joint_coords_flipped[:, 1] = -joint_coords_flipped[:, 1]
                    joint_coords_flipped[:, 2] = -joint_coords_flipped[:, 2]

                    skeleton_info = {
                        "joint_positions": joint_coords_flipped.tolist(),
                        "num_joints": len(joint_coords),
                    }

                    # Add skinning weights (build per-vertex data)
                    if vertex_weights:
                        num_vertices = len(vertices)
                        skinning_list = []
                        for vert_idx in range(num_vertices):
                            if vert_idx in vertex_weights:
                                skinning_list.append(vertex_weights[vert_idx])
                            else:
                                skinning_list.append([])
                        skeleton_info["skinning_weights"] = skinning_list

                    # Add joint parents
                    if joint_parents:
                        skeleton_info["joint_parents"] = joint_parents

                # Add person to combined data
                combined_data["people"].append({
                    "obj_path": temp_obj.name,
                    "skeleton": skeleton_info,
                    "index": i,
                })

            print(f"[SAM3D Export] people added to combined_data: {len(combined_data['people'])}")

            if not combined_data["people"]:
                raise RuntimeError("No valid mesh data to export")

            # Write combined JSON
            combined_json = tempfile.NamedTemporaryFile(suffix='_combined.json', delete=False, mode='w')
            temp_files.append(combined_json.name)
            json.dump(combined_data, combined_json)
            combined_json.close()

            # Export using Blender with combined script
            if BLENDER_EXE and os.path.exists(BLENDER_EXE):
                if not os.path.exists(BLENDER_MULTI_SCRIPT):
                    raise RuntimeError(f"Blender multi-export script not found: {BLENDER_MULTI_SCRIPT}")

                cmd = [
                    BLENDER_EXE,
                    "--background",
                    "--python", BLENDER_MULTI_SCRIPT,
                    "--",
                    combined_json.name,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)

                if result.returncode != 0:
                    raise RuntimeError(f"Blender export failed: {result.stderr}")

                if not os.path.exists(output_fbx_path):
                    raise RuntimeError(f"Export completed but output file not found: {output_fbx_path}")

            else:
                # Fallback: export individual OBJs
                for person_data in combined_data["people"]:
                    obj_path = person_data["obj_path"]
                    idx = person_data["index"]
                    fallback_path = output_fbx_path.replace('.fbx', f'_person{idx}.obj')
                    import shutil
                    shutil.copy(obj_path, fallback_path)
                output_fbx_path = output_fbx_path.replace('.fbx', '_person0.obj')

            return (os.path.basename(output_fbx_path),)

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass

    def _write_obj_file(self, filepath, vertices, faces):
        """Write mesh to OBJ file format."""
        with open(filepath, 'w') as f:
            for v in vertices:
                f.write(f"v {-v[0]:.6f} {-v[1]:.6f} {-v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyExportFBX": SAM3DBodyExportFBX,
    "SAM3DBodyExportMultipleFBX": SAM3DBodyExportMultipleFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyExportFBX": "SAM 3D Body: Export FBX",
    "SAM3DBodyExportMultipleFBX": "SAM 3D Body: Export Multiple FBX",
}
