"""
Blender script to export SAM3D Body mesh and skeleton to FBX file.

Usage: blender --background --python blender_export_sam3d_fbx.py -- <input_obj> <output_fbx> [skeleton_json]
"""

import bpy
import sys
import os
import json
import numpy as np
from mathutils import Vector

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_export_sam3d_fbx.py -- <input_obj> <output_fbx> [skeleton_json]")
    sys.exit(1)

input_obj = argv[0]
output_fbx = argv[1]
skeleton_json = argv[2] if len(argv) > 2 else None

# Load skeleton data from JSON if provided
joints = None
num_joints = 0
joint_parents_list = None
skinning_weights = None

if skeleton_json and os.path.exists(skeleton_json):
    try:
        with open(skeleton_json, 'r') as f:
            skeleton_data = json.load(f)

        joint_positions = skeleton_data.get('joint_positions', [])
        num_joints = skeleton_data.get('num_joints', len(joint_positions))
        joint_parents_list = skeleton_data.get('joint_parents')
        skinning_weights = skeleton_data.get('skinning_weights')

        if joint_positions:
            joints = np.array(joint_positions, dtype=np.float32)
    except Exception as e:
        joints = None


# Clean default scene
def clean_bpy():
    """Remove all default Blender objects"""
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

clean_bpy()

# Create collection
collection = bpy.data.collections.new('SAM3D_Export')
bpy.context.scene.collection.children.link(collection)

# Import OBJ mesh
try:
    bpy.ops.wm.obj_import(filepath=input_obj)

    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not imported_objects:
        raise RuntimeError("No mesh found after OBJ import")

    mesh_obj = imported_objects[0]
    mesh_obj.name = 'SAM3D_Character'

    # Move to our collection
    if mesh_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(mesh_obj)
    collection.objects.link(mesh_obj)

except Exception as e:
    print(f"[SAM3D] Failed to import OBJ: {e}")
    sys.exit(1)

# Create armature from skeleton if provided
if joints is not None and num_joints > 0:
    try:
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

    except Exception as e:
        print(f"[SAM3D] Armature creation failed: {e}")

# Export to FBX
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    # Select all objects in our collection
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in collection.objects:
        obj.select_set(True)

    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        use_selection=True,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )

except Exception as e:
    print(f"[SAM3D] Export failed: {e}")
    sys.exit(1)
