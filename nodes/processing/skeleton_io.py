# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Skeleton I/O nodes for SAM 3D Body.

Save, load, and manipulate skeleton data.
"""

import os
import json
import time
import subprocess
import numpy as np
import torch
import folder_paths


class SAM3DBodySaveSkeleton:
    """
    Save skeleton data to file in multiple formats.

    Exports skeleton with joint positions, rotations, and MHR parameters
    to JSON, BVH, or FBX format.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON", {
                    "tooltip": "Skeleton data from SAM3D Body Process node"
                }),
                "output_filename": ("STRING", {
                    "default": "skeleton",
                    "tooltip": "Output filename (extension will be added based on format)"
                }),
                "format": (["json", "bvh", "fbx"], {
                    "default": "json",
                    "tooltip": "Export format: JSON (full data), BVH (animation), or FBX (armature)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_skeleton"
    CATEGORY = "SAM3DBody/skeleton"
    OUTPUT_NODE = True

    def save_skeleton(self, skeleton, output_filename, format="json"):
        """Save skeleton to file in specified format."""
        print(f"[SAM3DBodySaveSkeleton] Saving skeleton as {format.upper()}...")

        # Prepare output path
        output_dir = folder_paths.get_output_directory()

        # Add extension if not present
        if not output_filename.endswith(f'.{format}'):
            output_filename = f"{output_filename}.{format}"

        output_path = os.path.join(output_dir, output_filename)

        # Save based on format
        if format == "json":
            self._save_json(skeleton, output_path)
        elif format == "bvh":
            self._save_bvh(skeleton, output_path)
        elif format == "fbx":
            self._save_fbx(skeleton, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"[SAM3DBodySaveSkeleton] ✓ Saved to: {output_path}")
        return (os.path.basename(output_path),)

    def _save_json(self, skeleton, output_path):
        """Save skeleton to JSON format (full data)."""
        # Convert tensors to numpy/lists
        json_data = {}

        for key, value in skeleton.items():
            if value is None:
                json_data[key] = None
            elif isinstance(value, torch.Tensor):
                json_data[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value

        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"[SAM3DBodySaveSkeleton] Saved JSON with {len(json_data)} fields")

    def _save_bvh(self, skeleton, output_path):
        """Save skeleton to BVH format (animation format)."""
        joint_positions = skeleton.get("joint_positions")
        joint_rotations = skeleton.get("joint_rotations")

        if joint_positions is None:
            raise RuntimeError("Skeleton has no joint_positions data")

        # Convert to numpy if needed
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.cpu().numpy()
        if joint_rotations is not None and isinstance(joint_rotations, torch.Tensor):
            joint_rotations = joint_rotations.cpu().numpy()

        # Create BVH file
        with open(output_path, 'w') as f:
            # Write header
            f.write("HIERARCHY\n")
            f.write("ROOT Hips\n")
            f.write("{\n")

            # For simplicity, create a flat hierarchy (all joints as children of root)
            # In a full implementation, you'd use the proper MHR hierarchy
            root_pos = joint_positions[0] if len(joint_positions) > 0 else [0, 0, 0]
            f.write(f"  OFFSET {root_pos[0]:.6f} {root_pos[1]:.6f} {root_pos[2]:.6f}\n")
            f.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")

            # Write joints (simplified - first 20 joints)
            num_joints = min(len(joint_positions), 20)
            for i in range(1, num_joints):
                pos = joint_positions[i]
                f.write(f"  JOINT Joint_{i:03d}\n")
                f.write("  {\n")
                f.write(f"    OFFSET {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
                f.write("    CHANNELS 3 Zrotation Xrotation Yrotation\n")

                # End site for leaf joints
                if i == num_joints - 1:
                    f.write("    End Site\n")
                    f.write("    {\n")
                    f.write("      OFFSET 0.0 0.0 0.0\n")
                    f.write("    }\n")

                f.write("  }\n")

            f.write("}\n")

            # Write motion data (single frame)
            f.write("MOTION\n")
            f.write("Frames: 1\n")
            f.write("Frame Time: 0.033333\n")

            # Write frame data (positions + rotations)
            # Root position
            f.write(f"{root_pos[0]:.6f} {root_pos[1]:.6f} {root_pos[2]:.6f} ")
            f.write("0.0 0.0 0.0 ")  # Root rotation (simplified)

            # Joint rotations (simplified - zeros for now)
            for i in range(1, num_joints):
                f.write("0.0 0.0 0.0 ")

            f.write("\n")

        print(f"[SAM3DBodySaveSkeleton] Saved BVH with {num_joints} joints")

    def _save_fbx(self, skeleton, output_path):
        """Save skeleton to FBX format (armature only) using Blender."""
        joint_positions = skeleton.get("joint_positions")

        if joint_positions is None:
            raise RuntimeError("Skeleton has no joint_positions data")

        # Convert to numpy if needed
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.cpu().numpy()

        # Save skeleton data to temporary JSON
        temp_dir = folder_paths.get_temp_directory()
        skeleton_json_path = os.path.join(temp_dir, f"skeleton_{int(time.time())}.json")

        skeleton_data = {
            "joint_positions": joint_positions.tolist(),
            "num_joints": len(joint_positions),
        }

        with open(skeleton_json_path, 'w') as f:
            json.dump(skeleton_data, f)

        try:
            # Find Blender
            blender_exe = self._find_blender()

            if not blender_exe or not os.path.exists(blender_exe):
                raise RuntimeError("Blender not found. Set BLENDER_EXE environment variable or install Blender.")

            # Create Blender script
            blender_script = self._create_blender_skeleton_export_script()
            script_path = os.path.join(temp_dir, f"export_skeleton_{int(time.time())}.py")

            with open(script_path, 'w') as f:
                f.write(blender_script)

            # Run Blender
            cmd = [
                blender_exe,
                '--background',
                '--python', script_path,
                '--',
                skeleton_json_path,
                output_path,
            ]

            print(f"[SAM3DBodySaveSkeleton] Running Blender to export FBX...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"[SAM3DBodySaveSkeleton] Blender stderr: {result.stderr}")
                print(f"[SAM3DBodySaveSkeleton] Blender stdout: {result.stdout}")
                raise RuntimeError(f"Blender export failed with return code {result.returncode}")

            if not os.path.exists(output_path):
                raise RuntimeError(f"Export completed but output file not found: {output_path}")

            print(f"[SAM3DBodySaveSkeleton] Exported skeleton as FBX")

        finally:
            # Clean up temporary files
            if os.path.exists(skeleton_json_path):
                os.unlink(skeleton_json_path)
            if 'script_path' in locals() and os.path.exists(script_path):
                os.unlink(script_path)

    def _find_blender(self):
        """Try to find Blender executable."""
        possible_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",
        ]

        env_path = os.environ.get("BLENDER_EXE")
        if env_path:
            possible_paths.insert(0, env_path)

        for path in possible_paths:
            if os.path.exists(path):
                return path

        try:
            result = subprocess.run(['which', 'blender'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        return None

    def _create_blender_skeleton_export_script(self):
        """Create Blender script for exporting skeleton to FBX."""
        return """
import bpy
import sys
import json

# Get arguments
args = sys.argv[sys.argv.index("--") + 1:]
skeleton_json = args[0]
fbx_path = args[1]

# Load skeleton data
with open(skeleton_json, 'r') as f:
    skeleton_data = json.load(f)

joint_positions = skeleton_data['joint_positions']

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create armature
bpy.ops.object.armature_add()
armature_obj = bpy.context.active_object
armature_obj.name = "SAM3D_Skeleton"
armature = armature_obj.data

# Enter edit mode
bpy.ops.object.mode_set(mode='EDIT')

# Clear default bone
armature.edit_bones.clear()

# Create bones for each joint
for i, pos in enumerate(joint_positions):
    bone = armature.edit_bones.new(f"Joint_{i:03d}")
    bone.head = (pos[0], pos[1], pos[2])
    bone.tail = (pos[0], pos[1] + 0.1, pos[2])  # Small offset for visualization

# Exit edit mode
bpy.ops.object.mode_set(mode='OBJECT')

# Export to FBX
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=False,
    object_types={'ARMATURE'},
    add_leaf_bones=False,
)

print(f"Successfully exported skeleton to {fbx_path}")
"""


class SAM3DBodyLoadSkeleton:
    """
    Load skeleton data from file.

    Supports JSON, BVH, and FBX formats.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", {
                    "default": "",
                    "tooltip": "Path to skeleton file (JSON, BVH, or FBX)"
                }),
            },
        }

    RETURN_TYPES = ("SKELETON",)
    RETURN_NAMES = ("skeleton",)
    FUNCTION = "load_skeleton"
    CATEGORY = "SAM3DBody/skeleton"

    def load_skeleton(self, filepath):
        """Load skeleton from file."""
        print(f"[SAM3DBodyLoadSkeleton] Loading skeleton from: {filepath}")

        if not os.path.exists(filepath):
            # Try in output directory
            output_dir = folder_paths.get_output_directory()
            filepath = os.path.join(output_dir, filepath)

            if not os.path.exists(filepath):
                raise RuntimeError(f"Skeleton file not found: {filepath}")

        # Determine format from extension
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.json':
            skeleton = self._load_json(filepath)
        elif ext == '.bvh':
            skeleton = self._load_bvh(filepath)
        elif ext == '.fbx':
            skeleton = self._load_fbx(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        print(f"[SAM3DBodyLoadSkeleton] ✓ Loaded skeleton")
        return (skeleton,)

    def _load_json(self, filepath):
        """Load skeleton from JSON format."""
        with open(filepath, 'r') as f:
            json_data = json.load(f)

        # Convert lists back to numpy arrays
        skeleton = {}
        for key, value in json_data.items():
            if value is None:
                skeleton[key] = None
            elif isinstance(value, list):
                skeleton[key] = np.array(value)
            else:
                skeleton[key] = value

        print(f"[SAM3DBodyLoadSkeleton] Loaded JSON with {len(skeleton)} fields")
        return skeleton

    def _load_bvh(self, filepath):
        """Load skeleton from BVH format."""
        # Simplified BVH parser - extract joint positions from hierarchy
        joint_positions = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse joint positions from OFFSET lines
        for line in lines:
            if 'OFFSET' in line:
                parts = line.strip().split()
                if len(parts) == 4:  # OFFSET x y z
                    try:
                        pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                        joint_positions.append(pos)
                    except ValueError:
                        continue

        if not joint_positions:
            raise RuntimeError("No valid joint positions found in BVH file")

        # Create skeleton dictionary
        skeleton = {
            "joint_positions": np.array(joint_positions),
            "joint_rotations": None,
            "pose_params": None,
            "shape_params": None,
            "scale_params": None,
            "hand_pose": None,
            "global_rot": None,
            "expr_params": None,
            "camera": None,
            "focal_length": None,
        }

        print(f"[SAM3DBodyLoadSkeleton] Loaded BVH with {len(joint_positions)} joints")
        return skeleton

    def _load_fbx(self, filepath):
        """Load skeleton from FBX format using Blender."""
        # This would require Blender to extract skeleton data from FBX
        # For now, raise not implemented
        raise NotImplementedError("Loading from FBX not yet implemented. Use JSON format for full compatibility.")


class SAM3DBodyAddMeshToSkeleton:
    """
    Generate mesh from skeleton using MHR model.

    Takes skeleton data (pose, shape, scale parameters) and uses the
    MHR parametric model to generate the corresponding mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON", {
                    "tooltip": "Skeleton data with pose/shape/scale parameters"
                }),
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM3D Body model (needed for MHR)"
                }),
            },
        }

    RETURN_TYPES = ("SAM3D_OUTPUT",)
    RETURN_NAMES = ("mesh_data",)
    FUNCTION = "add_mesh"
    CATEGORY = "SAM3DBody/skeleton"

    def add_mesh(self, skeleton, model):
        """Generate mesh from skeleton parameters using MHR model."""
        print(f"[SAM3DBodyAddMeshToSkeleton] Generating mesh from skeleton...")

        try:
            # Extract MHR model from loaded model
            sam_3d_model = model["model"]

            # Get skeleton parameters
            pose_params = skeleton.get("pose_params")
            shape_params = skeleton.get("shape_params")
            scale_params = skeleton.get("scale_params")
            hand_pose = skeleton.get("hand_pose")
            global_rot = skeleton.get("global_rot")
            expr_params = skeleton.get("expr_params")

            # Check if we have the necessary parameters
            if pose_params is None or shape_params is None or scale_params is None:
                raise RuntimeError("Skeleton missing required parameters (pose_params, shape_params, scale_params)")

            # Convert numpy arrays to tensors if needed
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params).float()
            if isinstance(shape_params, np.ndarray):
                shape_params = torch.from_numpy(shape_params).float()
            if isinstance(scale_params, np.ndarray):
                scale_params = torch.from_numpy(scale_params).float()
            if hand_pose is not None and isinstance(hand_pose, np.ndarray):
                hand_pose = torch.from_numpy(hand_pose).float()
            if global_rot is not None and isinstance(global_rot, np.ndarray):
                global_rot = torch.from_numpy(global_rot).float()
            if expr_params is not None and isinstance(expr_params, np.ndarray):
                expr_params = torch.from_numpy(expr_params).float()

            # Add batch dimension if needed
            if pose_params.dim() == 1:
                pose_params = pose_params.unsqueeze(0)
            if shape_params.dim() == 1:
                shape_params = shape_params.unsqueeze(0)
            if scale_params.dim() == 1:
                scale_params = scale_params.unsqueeze(0)
            if hand_pose is not None and hand_pose.dim() == 1:
                hand_pose = hand_pose.unsqueeze(0)
            if global_rot is not None and global_rot.dim() == 1:
                global_rot = global_rot.unsqueeze(0)
            if expr_params is not None and expr_params.dim() == 1:
                expr_params = expr_params.unsqueeze(0)

            # Move to same device as model
            device = next(sam_3d_model.parameters()).device
            pose_params = pose_params.to(device)
            shape_params = shape_params.to(device)
            scale_params = scale_params.to(device)
            if hand_pose is not None:
                hand_pose = hand_pose.to(device)
            if global_rot is not None:
                global_rot = global_rot.to(device)
            if expr_params is not None:
                expr_params = expr_params.to(device)

            # Use MHR model to generate mesh
            # Access MHR from the model
            mhr = sam_3d_model.mhr_head.mhr if hasattr(sam_3d_model, 'mhr_head') else None

            if mhr is None:
                raise RuntimeError("MHR model not found in SAM3D model. Cannot generate mesh from skeleton.")

            # Build model parameters for MHR
            # Combine pose parameters: global_rot + pose_params + hand_pose
            # This is based on the MHR input format
            with torch.no_grad():
                # Generate mesh using MHR
                # Note: This is a simplified version - may need to adjust based on actual MHR interface
                vertices, joint_coords = mhr(
                    shape_params,
                    pose_params,
                    expr_params if expr_params is not None else torch.zeros(1, 72, device=device),
                )

            # Get faces from model
            faces = sam_3d_model.mhr_head.mhr.faces.cpu().numpy() if hasattr(sam_3d_model.mhr_head.mhr, 'faces') else None

            # Create mesh_data dictionary
            mesh_data = {
                "vertices": vertices,
                "faces": faces,
                "joints": None,  # 70 keypoints not directly available
                "joint_coords": joint_coords,  # 127 joints
                "joint_rotations": skeleton.get("joint_rotations"),
                "camera": skeleton.get("camera"),
                "focal_length": skeleton.get("focal_length"),
                "bbox": None,
                "pose_params": {
                    "body_pose": pose_params,
                    "hand_pose": hand_pose,
                    "global_rot": global_rot,
                    "shape": shape_params,
                    "scale": scale_params,
                    "expr": expr_params,
                },
                "raw_output": {},
                "all_people": [],
            }

            print(f"[SAM3DBodyAddMeshToSkeleton] ✓ Generated mesh with {len(vertices[0])} vertices")
            return (mesh_data,)

        except Exception as e:
            print(f"[SAM3DBodyAddMeshToSkeleton] [ERROR] Failed to generate mesh: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodySaveSkeleton": SAM3DBodySaveSkeleton,
    "SAM3DBodyLoadSkeleton": SAM3DBodyLoadSkeleton,
    "SAM3DBodyAddMeshToSkeleton": SAM3DBodyAddMeshToSkeleton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodySaveSkeleton": "SAM 3D Body: Save Skeleton",
    "SAM3DBodyLoadSkeleton": "SAM 3D Body: Load Skeleton",
    "SAM3DBodyAddMeshToSkeleton": "SAM 3D Body: Add Mesh to Skeleton",
}
