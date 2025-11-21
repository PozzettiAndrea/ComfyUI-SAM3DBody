#!/usr/bin/env python3
# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Blender script to apply pose transforms to an FBX and export.

Usage:
    blender --background --python blender_apply_pose.py -- input.fbx output.fbx transforms.json
"""

import sys
import json
import bpy
import mathutils


def apply_pose_to_fbx(input_fbx_path, output_fbx_path, transforms_json_path):
    """
    Load an FBX, apply bone transforms, and export to new FBX.

    Args:
        input_fbx_path: Path to input FBX file
        output_fbx_path: Path to output FBX file
        transforms_json_path: Path to JSON file containing bone transforms
    """
    print(f"[Blender Pose] Loading FBX: {input_fbx_path}")

    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import the FBX
    bpy.ops.import_scene.fbx(filepath=input_fbx_path)

    # Load bone transforms from JSON
    with open(transforms_json_path, 'r') as f:
        bone_transforms = json.load(f)

    print(f"[Blender Pose] Loaded {len(bone_transforms)} bone transforms")

    # Find the armature
    armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break

    if not armature:
        print("[Blender Pose] ERROR: No armature found in FBX")
        sys.exit(1)

    print(f"[Blender Pose] Found armature: {armature.name}")
    print(f"[Blender Pose] Armature has {len(armature.pose.bones)} pose bones")

    # Apply transforms to pose bones
    applied_count = 0
    for bone_name, transform in bone_transforms.items():
        if bone_name not in armature.pose.bones:
            print(f"[Blender Pose] WARNING: Bone '{bone_name}' not found in armature")
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

    print(f"[Blender Pose] Applied transforms to {applied_count} bones")

    # Update the scene to apply transforms
    bpy.context.view_layer.update()

    # Export to FBX with current pose
    print(f"[Blender Pose] Exporting posed FBX: {output_fbx_path}")
    bpy.ops.export_scene.fbx(
        filepath=output_fbx_path,
        use_selection=False,
        apply_scale_options='FBX_SCALE_ALL',
        bake_anim=False,  # Don't bake animation, just export the current pose
        add_leaf_bones=False,
    )

    print(f"[Blender Pose] ✓ Export complete")


if __name__ == "__main__":
    # Parse command line arguments
    # Format: blender --background --python script.py -- arg1 arg2 arg3
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # Get all args after "--"

    if len(argv) < 3:
        print("Usage: blender --background --python blender_apply_pose.py -- input.fbx output.fbx transforms.json")
        sys.exit(1)

    input_fbx_path = argv[0]
    output_fbx_path = argv[1]
    transforms_json_path = argv[2]

    apply_pose_to_fbx(input_fbx_path, output_fbx_path, transforms_json_path)
