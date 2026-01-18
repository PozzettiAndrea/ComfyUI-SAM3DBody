# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Export nodes for SAM 3D Body meshes.

Exports meshes with rigging data to various formats using bpy in isolated venv.
"""

import os
import json
import time
import tempfile
import numpy as np
import torch
import folder_paths
import glob

from comfy_env import isolated


@isolated(env="sam3dbody", import_paths=[".", "..", "../.."])
class BpyFBXExporter:
    """Isolated bpy-based FBX exporter that runs in the sam3dbody venv."""

    FUNCTION = "export"

    def export(self, input_obj_path, output_fbx_path, skeleton_json_path=None, combined_json_path=None):
        """Export OBJ mesh to FBX using bpy.

        If combined_json_path is provided, exports multiple people into a single FBX.
        Otherwise, exports a single person.
        """
        import bpy
        from mathutils import Vector
        import numpy as np
        import json
        import os

        # Handle combined export mode
        if combined_json_path and os.path.exists(combined_json_path):
            return self._export_combined(combined_json_path, output_fbx_path)

        # Single person export mode - load skeleton data from JSON if provided
        joints = None
        num_joints = 0
        joint_parents_list = None
        skinning_weights = None
        global_rotations = None

        if skeleton_json_path and os.path.exists(skeleton_json_path):
            with open(skeleton_json_path, 'r') as f:
                skeleton_data = json.load(f)

            joint_positions = skeleton_data.get('joint_positions', [])
            num_joints = skeleton_data.get('num_joints', len(joint_positions))
            joint_parents_list = skeleton_data.get('joint_parents')
            skinning_weights = skeleton_data.get('skinning_weights')
            global_rotations_data = skeleton_data.get('global_rotations')

            if joint_positions:
                joints = np.array(joint_positions, dtype=np.float32)

            if global_rotations_data:
                global_rotations = np.array(global_rotations_data, dtype=np.float32)
                print(f"[BpyFBXExporter] Loaded global_rotations: shape {global_rotations.shape}")

        # Clean default scene
        for c in bpy.data.actions:
            bpy.data.actions.remove(c)
        for c in bpy.data.armatures:
            bpy.data.armatures.remove(c)
        for c in bpy.data.cameras:
            bpy.data.cameras.remove(c)
        for c in bpy.data.collections:
            bpy.data.collections.remove(c)
        for c in bpy.data.images:
            bpy.data.images.remove(c)
        for c in bpy.data.materials:
            bpy.data.materials.remove(c)
        for c in bpy.data.meshes:
            bpy.data.meshes.remove(c)
        for c in bpy.data.objects:
            bpy.data.objects.remove(c)
        for c in bpy.data.textures:
            bpy.data.textures.remove(c)

        # Create collection
        collection = bpy.data.collections.new('SAM3D_Export')
        bpy.context.scene.collection.children.link(collection)

        # Import OBJ mesh
        bpy.ops.wm.obj_import(filepath=input_obj_path)

        imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        if not imported_objects:
            raise RuntimeError("No mesh found after OBJ import")

        mesh_obj = imported_objects[0]
        mesh_obj.name = 'SAM3D_Character'

        # Move to our collection
        if mesh_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(mesh_obj)
        collection.objects.link(mesh_obj)

        # Create armature from skeleton if provided
        if joints is not None and num_joints > 0:
            # Create armature in edit mode
            bpy.ops.object.armature_add(enter_editmode=True)
            armature = bpy.data.armatures.get('Armature')
            armature.name = 'SAM3D_Skeleton'
            armature_obj = bpy.context.active_object
            armature_obj.name = 'SAM3D_Skeleton'

            # Move to our collection
            if armature_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(armature_obj)
            collection.objects.link(armature_obj)

            edit_bones = armature.edit_bones
            extrude_size = 0.05

            # Remove default bone
            default_bone = edit_bones.get('Bone')
            if default_bone:
                edit_bones.remove(default_bone)

            # Calculate skeleton center for root bone placement
            skeleton_center = joints.mean(axis=0)

            # Make positions relative to skeleton center
            rel_joints = joints - skeleton_center

            # Apply coordinate system correction to match mesh orientation
            rel_joints_corrected = np.zeros_like(rel_joints)
            rel_joints_corrected[:, 0] = rel_joints[:, 0]
            rel_joints_corrected[:, 1] = -rel_joints[:, 2]
            rel_joints_corrected[:, 2] = rel_joints[:, 1]

            # Create all bones
            bones_dict = {}
            for i in range(num_joints):
                bone_name = f'Joint_{i:03d}'
                bone = edit_bones.new(bone_name)
                bone.head = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2]))
                bone.tail = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2] + extrude_size))
                bones_dict[bone_name] = bone

            # Build hierarchical structure using joint parents if available
            if joint_parents_list and len(joint_parents_list) == num_joints:
                for i in range(num_joints):
                    parent_idx = joint_parents_list[i]
                    if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                        bone_name = f'Joint_{i:03d}'
                        parent_bone_name = f'Joint_{parent_idx:03d}'
                        bones_dict[bone_name].parent = bones_dict[parent_bone_name]
                        bones_dict[bone_name].use_connect = False
            else:
                # Fallback: create flat hierarchy with Joint_000 as root
                root_bone_name = 'Joint_000'
                for i in range(1, num_joints):
                    bone_name = f'Joint_{i:03d}'
                    bones_dict[bone_name].parent = bones_dict[root_bone_name]
                    bones_dict[bone_name].use_connect = False

            # Switch to object mode
            bpy.ops.object.mode_set(mode='OBJECT')

            # Position armature at skeleton center
            skeleton_center_corrected = np.zeros(3)
            skeleton_center_corrected[0] = skeleton_center[0]
            skeleton_center_corrected[1] = -skeleton_center[2]
            skeleton_center_corrected[2] = skeleton_center[1]
            armature_obj.location = Vector((skeleton_center_corrected[0], skeleton_center_corrected[1], skeleton_center_corrected[2]))

            # Apply skinning weights if available
            if skinning_weights:
                # Create vertex groups for each bone
                for i in range(num_joints):
                    bone_name = f'Joint_{i:03d}'
                    mesh_obj.vertex_groups.new(name=bone_name)

                # Assign weights to vertices
                num_vertices = len(mesh_obj.data.vertices)
                for vert_idx in range(min(num_vertices, len(skinning_weights))):
                    influences = skinning_weights[vert_idx]
                    if influences and len(influences) > 0:
                        for bone_idx, weight in influences:
                            if 0 <= bone_idx < num_joints and weight > 0.0001:
                                bone_name = f'Joint_{bone_idx:03d}'
                                vertex_group = mesh_obj.vertex_groups.get(bone_name)
                                if vertex_group:
                                    vertex_group.add([vert_idx], weight, 'REPLACE')

            # Deselect all
            for obj in bpy.context.selected_objects:
                obj.select_set(False)

            # Parent mesh to armature
            mesh_obj.select_set(True)
            armature_obj.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj

            if skinning_weights:
                bpy.ops.object.parent_set(type='ARMATURE')
            else:
                bpy.ops.object.parent_set(type='ARMATURE_NAME')

        # Make mesh double-sided AFTER skinning (so duplicated vertices inherit weights)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.duplicate()
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode='OBJECT')

        # Export to FBX
        os.makedirs(os.path.dirname(output_fbx_path) if os.path.dirname(output_fbx_path) else '.', exist_ok=True)

        # Select all objects in our collection
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in collection.objects:
            obj.select_set(True)

        # Export FBX
        bpy.ops.export_scene.fbx(
            filepath=output_fbx_path,
            check_existing=False,
            use_selection=True,
            add_leaf_bones=False,
            path_mode='COPY',
            embed_textures=True,
        )

        return {"success": True, "output_path": output_fbx_path}

    def _export_combined(self, combined_json_path, output_fbx_path):
        """
        Export multiple people into a single combined FBX file.

        Args:
            combined_json_path: Path to JSON file containing list of people data
            output_fbx_path: Output path for the combined FBX file
        """
        import bpy
        from mathutils import Vector
        import numpy as np
        import json
        import os

        # Load the combined data from JSON
        with open(combined_json_path, 'r') as f:
            people_data = json.load(f)

        # Clean default scene
        for c in bpy.data.actions:
            bpy.data.actions.remove(c)
        for c in bpy.data.armatures:
            bpy.data.armatures.remove(c)
        for c in bpy.data.cameras:
            bpy.data.cameras.remove(c)
        for c in bpy.data.collections:
            bpy.data.collections.remove(c)
        for c in bpy.data.images:
            bpy.data.images.remove(c)
        for c in bpy.data.materials:
            bpy.data.materials.remove(c)
        for c in bpy.data.meshes:
            bpy.data.meshes.remove(c)
        for c in bpy.data.objects:
            bpy.data.objects.remove(c)
        for c in bpy.data.textures:
            bpy.data.textures.remove(c)

        # Create collection for all exported objects
        collection = bpy.data.collections.new('SAM3D_Export')
        bpy.context.scene.collection.children.link(collection)

        # Process each person
        for person in people_data:
            obj_path = person["obj_path"]
            skeleton_json_path = person.get("skeleton_json_path")
            idx = person["index"]

            # Load skeleton data from JSON if provided
            joints = None
            num_joints = 0
            joint_parents_list = None
            skinning_weights = None
            global_rotations = None

            if skeleton_json_path and os.path.exists(skeleton_json_path):
                with open(skeleton_json_path, 'r') as f:
                    skeleton_data = json.load(f)

                joint_positions = skeleton_data.get('joint_positions', [])
                num_joints = skeleton_data.get('num_joints', len(joint_positions))
                joint_parents_list = skeleton_data.get('joint_parents')
                skinning_weights = skeleton_data.get('skinning_weights')
                global_rotations_data = skeleton_data.get('global_rotations')

                if joint_positions:
                    joints = np.array(joint_positions, dtype=np.float32)

                if global_rotations_data:
                    global_rotations = np.array(global_rotations_data, dtype=np.float32)
                    print(f"[BpyFBXExporter] Person {idx}: Loaded global_rotations shape {global_rotations.shape}")

            # Import OBJ mesh
            bpy.ops.wm.obj_import(filepath=obj_path)

            imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not imported_objects:
                continue

            mesh_obj = imported_objects[0]
            mesh_obj.name = f'SAM3D_Character_{idx}'

            # Move to our collection
            if mesh_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(mesh_obj)
            collection.objects.link(mesh_obj)

            # Create armature from skeleton if provided
            if joints is not None and num_joints > 0:
                # Create armature in edit mode
                bpy.ops.object.armature_add(enter_editmode=True)
                armature = bpy.context.active_object.data
                armature.name = f'SAM3D_Skeleton_{idx}'
                armature_obj = bpy.context.active_object
                armature_obj.name = f'SAM3D_Skeleton_{idx}'

                # Move to our collection
                if armature_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(armature_obj)
                collection.objects.link(armature_obj)

                edit_bones = armature.edit_bones
                extrude_size = 0.05

                # Remove default bone
                default_bone = edit_bones.get('Bone')
                if default_bone:
                    edit_bones.remove(default_bone)

                # Calculate skeleton center for root bone placement
                skeleton_center = joints.mean(axis=0)

                # Make positions relative to skeleton center
                rel_joints = joints - skeleton_center

                # Apply coordinate system correction to match mesh orientation
                rel_joints_corrected = np.zeros_like(rel_joints)
                rel_joints_corrected[:, 0] = rel_joints[:, 0]
                rel_joints_corrected[:, 1] = -rel_joints[:, 2]
                rel_joints_corrected[:, 2] = rel_joints[:, 1]

                # Create all bones with unique names for this person
                bones_dict = {}
                for i in range(num_joints):
                    bone_name = f'P{idx}_Joint_{i:03d}'
                    bone = edit_bones.new(bone_name)
                    bone.head = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2]))
                    bone.tail = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2] + extrude_size))
                    bones_dict[bone_name] = bone

                # Build hierarchical structure using joint parents if available
                if joint_parents_list and len(joint_parents_list) == num_joints:
                    for i in range(num_joints):
                        parent_idx = joint_parents_list[i]
                        if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                            bone_name = f'P{idx}_Joint_{i:03d}'
                            parent_bone_name = f'P{idx}_Joint_{parent_idx:03d}'
                            bones_dict[bone_name].parent = bones_dict[parent_bone_name]
                            bones_dict[bone_name].use_connect = False
                else:
                    # Fallback: create flat hierarchy with Joint_000 as root
                    root_bone_name = f'P{idx}_Joint_000'
                    for i in range(1, num_joints):
                        bone_name = f'P{idx}_Joint_{i:03d}'
                        bones_dict[bone_name].parent = bones_dict[root_bone_name]
                        bones_dict[bone_name].use_connect = False

                # Switch to object mode
                bpy.ops.object.mode_set(mode='OBJECT')

                # Position armature at skeleton center
                skeleton_center_corrected = np.zeros(3)
                skeleton_center_corrected[0] = skeleton_center[0]
                skeleton_center_corrected[1] = -skeleton_center[2]
                skeleton_center_corrected[2] = skeleton_center[1]
                armature_obj.location = Vector((skeleton_center_corrected[0], skeleton_center_corrected[1], skeleton_center_corrected[2]))

                # Apply skinning weights if available
                if skinning_weights:
                    # Create vertex groups for each bone
                    for i in range(num_joints):
                        bone_name = f'P{idx}_Joint_{i:03d}'
                        mesh_obj.vertex_groups.new(name=bone_name)

                    # Assign weights to vertices
                    num_vertices = len(mesh_obj.data.vertices)
                    for vert_idx in range(min(num_vertices, len(skinning_weights))):
                        influences = skinning_weights[vert_idx]
                        if influences and len(influences) > 0:
                            for bone_idx, weight in influences:
                                if 0 <= bone_idx < num_joints and weight > 0.0001:
                                    bone_name = f'P{idx}_Joint_{bone_idx:03d}'
                                    vertex_group = mesh_obj.vertex_groups.get(bone_name)
                                    if vertex_group:
                                        vertex_group.add([vert_idx], weight, 'REPLACE')

                # Deselect all
                for obj in bpy.context.selected_objects:
                    obj.select_set(False)

                # Parent mesh to armature
                mesh_obj.select_set(True)
                armature_obj.select_set(True)
                bpy.context.view_layer.objects.active = armature_obj

                if skinning_weights:
                    bpy.ops.object.parent_set(type='ARMATURE')
                else:
                    bpy.ops.object.parent_set(type='ARMATURE_NAME')

            # Make mesh double-sided AFTER skinning (so duplicated vertices inherit weights)
            bpy.context.view_layer.objects.active = mesh_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.duplicate()
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode='OBJECT')

        # Export to FBX
        os.makedirs(os.path.dirname(output_fbx_path) if os.path.dirname(output_fbx_path) else '.', exist_ok=True)

        # Select all objects in our collection
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in collection.objects:
            obj.select_set(True)

        # Export FBX with all objects
        bpy.ops.export_scene.fbx(
            filepath=output_fbx_path,
            check_existing=False,
            use_selection=True,
            add_leaf_bones=False,
            path_mode='COPY',
            embed_textures=True,
        )

        return {"success": True, "output_path": output_fbx_path}


@isolated(env="sam3dbody", import_paths=[".", "..", "../.."])
class BpyPoseApplier:
    """Isolated bpy-based pose applier that runs in the sam3dbody venv."""

    FUNCTION = "apply_pose"

    def apply_pose(self, input_fbx_path, output_fbx_path, transforms_json_path):
        """
        Load an FBX, apply bone transforms, and export to new FBX.

        Args:
            input_fbx_path: Path to input FBX file
            output_fbx_path: Path to output FBX file
            transforms_json_path: Path to JSON file containing bone transforms
        """
        import bpy
        import mathutils
        import json
        import os

        print(f"[BpyPoseApplier] Loading FBX: {input_fbx_path}")

        # Clear the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Import the FBX
        bpy.ops.import_scene.fbx(filepath=input_fbx_path)

        # Load bone transforms from JSON
        with open(transforms_json_path, 'r') as f:
            bone_transforms = json.load(f)

        print(f"[BpyPoseApplier] Loaded {len(bone_transforms)} bone transforms")

        # Find the armature
        armature = None
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break

        if not armature:
            return {"success": False, "error": "No armature found in FBX"}

        print(f"[BpyPoseApplier] Found armature: {armature.name}")
        print(f"[BpyPoseApplier] Armature has {len(armature.pose.bones)} pose bones")

        # Apply transforms to pose bones
        applied_count = 0
        for bone_name, transform in bone_transforms.items():
            if bone_name not in armature.pose.bones:
                print(f"[BpyPoseApplier] WARNING: Bone '{bone_name}' not found in armature")
                continue

            pose_bone = armature.pose.bones[bone_name]

            # Apply position delta (offset from rest pose)
            pos_delta = transform.get('position', {})
            if pos_delta:
                pose_bone.location.x += pos_delta.get('x', 0)
                pose_bone.location.y += pos_delta.get('y', 0)
                pose_bone.location.z += pos_delta.get('z', 0)

            # Apply rotation delta (quaternion multiply)
            quat_delta = transform.get('quaternion', {})
            if quat_delta:
                delta_quat = mathutils.Quaternion((
                    quat_delta.get('w', 1.0),
                    quat_delta.get('x', 0.0),
                    quat_delta.get('y', 0.0),
                    quat_delta.get('z', 0.0)
                ))
                # Multiply current rotation by delta
                pose_bone.rotation_quaternion = pose_bone.rotation_quaternion @ delta_quat

            # Apply scale delta (multiply)
            scale_delta = transform.get('scale', {})
            if scale_delta:
                pose_bone.scale.x *= scale_delta.get('x', 1.0)
                pose_bone.scale.y *= scale_delta.get('y', 1.0)
                pose_bone.scale.z *= scale_delta.get('z', 1.0)

            applied_count += 1

        print(f"[BpyPoseApplier] Applied transforms to {applied_count} bones")

        # Update the scene to apply transforms
        bpy.context.view_layer.update()

        # Export to FBX with current pose
        os.makedirs(os.path.dirname(output_fbx_path) if os.path.dirname(output_fbx_path) else '.', exist_ok=True)

        print(f"[BpyPoseApplier] Exporting posed FBX: {output_fbx_path}")
        bpy.ops.export_scene.fbx(
            filepath=output_fbx_path,
            use_selection=False,
            apply_scale_options='FBX_SCALE_ALL',
            bake_anim=False,
            add_leaf_bones=False,
        )

        print(f"[BpyPoseApplier] Export complete")
        return {"success": True, "output_path": output_fbx_path}


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
            # Apply same transform as mesh: flip Y and Z axes
            mesh_min = [mesh_min[0], -mesh_min[1], -mesh_min[2]]
            mesh_max = [mesh_max[0], -mesh_max[1], -mesh_max[2]]
            # Ensure min < max after flipping (signs reverse order)
            mesh_min, mesh_max = [min(mesh_min[i], mesh_max[i]) for i in range(3)], [max(mesh_min[i], mesh_max[i]) for i in range(3)]

            # Apply coordinate transform to joint positions to match mesh (flip Y and Z)
            joint_coords_flipped = joint_coords.copy()
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
            # Use isolated bpy exporter in sam3dbody venv
            exporter = BpyFBXExporter()
            result = exporter.export(
                input_obj_path=temp_obj_path,
                output_fbx_path=output_fbx_path,
                skeleton_json_path=skeleton_json_path
            )

            if not result.get("success"):
                raise RuntimeError(f"FBX export failed")

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
                f.write(f"v {v[0]:.6f} {-v[1]:.6f} {-v[2]:.6f}\n")
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
                "combine": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, exports all people into a single FBX file (works with Preview3D). When disabled, creates separate FBX files per person."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "export_multi_fbx"
    CATEGORY = "SAM3DBody/export"
    OUTPUT_NODE = True

    def export_multi_fbx(self, multi_mesh_data, output_filename, combine):
        """Export all meshes with skeletons to FBX file(s).

        Args:
            multi_mesh_data: Multi-person mesh data from SAM3D Body Process Multiple node
            output_filename: Output filename for the FBX file
            combine: If True, export all people into single FBX. If False, create separate FBX per person.
        """

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
                global_rots = person.get("pred_global_rots")  # Global joint rotations for bone orientations

                if vertices is None:
                    continue

                # Convert to numpy
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                if joint_coords is not None and isinstance(joint_coords, torch.Tensor):
                    joint_coords = joint_coords.cpu().numpy()
                if cam_t is not None and isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                if global_rots is not None and isinstance(global_rots, torch.Tensor):
                    global_rots = global_rots.cpu().numpy()

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
                    # Apply coordinate transform to joint positions (flip Y and Z)
                    joint_coords_flipped = joint_coords.copy()
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

                    # Add global joint rotations for better bone orientations in FBX
                    if global_rots is not None:
                        skeleton_info["global_rotations"] = global_rots.tolist()
                        print(f"[SAM3D Export] Person {i}: Including global_rots shape {global_rots.shape}")

                # Add person to combined data
                combined_data["people"].append({
                    "obj_path": temp_obj.name,
                    "skeleton": skeleton_info,
                    "index": i,
                })

            print(f"[SAM3D Export] people added to combined_data: {len(combined_data['people'])}")
            print(f"[SAM3D Export] combine: {combine}")

            if not combined_data["people"]:
                raise RuntimeError("No valid mesh data to export")

            exporter = BpyFBXExporter()

            if combine:
                # Combined mode: export all people into a single FBX file
                # Write skeleton JSON files for each person and build combined data
                people_data_for_export = []
                for person_data in combined_data["people"]:
                    idx = person_data["index"]
                    skeleton_info = person_data.get("skeleton", {})

                    # Write skeleton JSON for this person if available
                    person_skeleton_json = None
                    if skeleton_info:
                        person_skeleton_json = tempfile.NamedTemporaryFile(
                            suffix=f'_person{idx}_skeleton.json', delete=False, mode='w'
                        )
                        temp_files.append(person_skeleton_json.name)
                        json.dump(skeleton_info, person_skeleton_json)
                        person_skeleton_json.close()
                        person_skeleton_json = person_skeleton_json.name

                    people_data_for_export.append({
                        "obj_path": person_data["obj_path"],
                        "skeleton_json_path": person_skeleton_json,
                        "index": idx,
                    })

                # Write combined data JSON file
                combined_json = tempfile.NamedTemporaryFile(
                    suffix='_combined_export.json', delete=False, mode='w'
                )
                temp_files.append(combined_json.name)
                json.dump(people_data_for_export, combined_json)
                combined_json.close()

                # Export all people into single FBX via combined_json_path
                result = exporter.export(
                    input_obj_path=None,
                    output_fbx_path=output_fbx_path,
                    combined_json_path=combined_json.name
                )

                if not result.get("success"):
                    raise RuntimeError("Combined FBX export failed")

                print(f"[SAM3D Export] Combined FBX created: {output_fbx_path}")
                return (os.path.basename(output_fbx_path),)

            else:
                # Separate mode: export each person to individual FBX files
                exported_files = []

                for person_data in combined_data["people"]:
                    obj_path = person_data["obj_path"]
                    idx = person_data["index"]
                    skeleton_info = person_data.get("skeleton", {})

                    # Create per-person FBX filename
                    if len(combined_data["people"]) == 1:
                        person_fbx_path = output_fbx_path
                    else:
                        person_fbx_path = output_fbx_path.replace('.fbx', f'_person{idx}.fbx')

                    # Write skeleton JSON for this person if available
                    person_skeleton_json = None
                    if skeleton_info:
                        person_skeleton_json = tempfile.NamedTemporaryFile(
                            suffix=f'_person{idx}_skeleton.json', delete=False, mode='w'
                        )
                        temp_files.append(person_skeleton_json.name)
                        json.dump(skeleton_info, person_skeleton_json)
                        person_skeleton_json.close()
                        person_skeleton_json = person_skeleton_json.name

                    # Export using isolated bpy
                    result = exporter.export(
                        input_obj_path=obj_path,
                        output_fbx_path=person_fbx_path,
                        skeleton_json_path=person_skeleton_json
                    )

                    if result.get("success"):
                        exported_files.append(person_fbx_path)
                    else:
                        raise RuntimeError(f"FBX export failed for person {idx}")

                if not exported_files:
                    raise RuntimeError("No FBX files were exported")

                # Return the first exported file (separate mode returns first file for compatibility)
                output_fbx_path = exported_files[0]
                print(f"[SAM3D Export] Separate FBX files created: {len(exported_files)} files")
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
                f.write(f"v {v[0]:.6f} {-v[1]:.6f} {-v[2]:.6f}\n")
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
