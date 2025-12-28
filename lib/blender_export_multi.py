"""
Blender script to export multiple SAM3D Body meshes and skeletons to a single FBX file.

Usage: blender --background --python blender_export_multi.py -- <combined_json>

The combined JSON should contain:
{
    "output_path": "/path/to/output.fbx",
    "people": [
        {
            "obj_path": "/path/to/person0.obj",
            "skeleton": {
                "joint_positions": [...],
                "joint_parents": [...],
                "skinning_weights": [...]
            },
            "index": 0
        },
        ...
    ]
}
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

if len(argv) < 1:
    print("Usage: blender --background --python blender_export_multi.py -- <combined_json>")
    sys.exit(1)

combined_json_path = argv[0]

# Load combined data
with open(combined_json_path, 'r') as f:
    combined_data = json.load(f)

output_fbx = combined_data.get("output_path")
people_data = combined_data.get("people", [])

print(f"[SAM3D Multi] Output path: {output_fbx}")
print(f"[SAM3D Multi] Number of people in JSON: {len(people_data)}")

if not output_fbx or not people_data:
    print("[SAM3D Multi] Invalid combined data - missing output_path or people")
    sys.exit(1)


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


def make_mesh_double_sided(mesh_obj):
    """Make mesh double-sided by duplicating faces with flipped normals."""
    # Select the mesh and enter edit mode
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all geometry
    bpy.ops.mesh.select_all(action='SELECT')

    # Duplicate faces
    bpy.ops.mesh.duplicate()

    # Flip normals on duplicated faces
    bpy.ops.mesh.flip_normals()

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[SAM3D Multi] Made mesh double-sided: {mesh_obj.name}")


def create_armature_for_person(person_idx, skeleton_data, collection):
    """Create armature from skeleton data for a specific person."""
    joint_positions = skeleton_data.get('joint_positions', [])
    if not joint_positions:
        return None

    joints = np.array(joint_positions, dtype=np.float32)
    num_joints = len(joints)
    joint_parents_list = skeleton_data.get('joint_parents')
    skinning_weights = skeleton_data.get('skinning_weights')

    # Create armature in edit mode
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.active_object.data
    armature.name = f'Skeleton_Person{person_idx}'
    armature_obj = bpy.context.active_object
    armature_obj.name = f'Armature_Person{person_idx}'

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

    # Create all bones with person-specific names
    bones_dict = {}
    for i in range(num_joints):
        bone_name = f'P{person_idx}_Joint_{i:03d}'
        bone = edit_bones.new(bone_name)
        bone.head = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2]))
        bone.tail = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2] + extrude_size))
        bones_dict[i] = bone

    # Build hierarchical structure using joint parents if available
    if joint_parents_list and len(joint_parents_list) == num_joints:
        for i in range(num_joints):
            parent_idx = joint_parents_list[i]
            if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                bones_dict[i].parent = bones_dict[parent_idx]
                bones_dict[i].use_connect = False
    else:
        # Fallback: create flat hierarchy with first joint as root
        for i in range(1, num_joints):
            bones_dict[i].parent = bones_dict[0]
            bones_dict[i].use_connect = False

    # Switch to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Position armature at skeleton center
    skeleton_center_corrected = np.zeros(3)
    skeleton_center_corrected[0] = skeleton_center[0]
    skeleton_center_corrected[1] = -skeleton_center[2]
    skeleton_center_corrected[2] = skeleton_center[1]
    armature_obj.location = Vector((skeleton_center_corrected[0], skeleton_center_corrected[1], skeleton_center_corrected[2]))

    return {
        'armature_obj': armature_obj,
        'num_joints': num_joints,
        'skinning_weights': skinning_weights,
        'person_idx': person_idx,
    }


def apply_skinning_to_mesh(mesh_obj, armature_info):
    """Apply skinning weights from armature_info to mesh."""
    armature_obj = armature_info['armature_obj']
    num_joints = armature_info['num_joints']
    skinning_weights = armature_info['skinning_weights']
    person_idx = armature_info['person_idx']

    if skinning_weights:
        # Create vertex groups for each bone
        for i in range(num_joints):
            bone_name = f'P{person_idx}_Joint_{i:03d}'
            mesh_obj.vertex_groups.new(name=bone_name)

        # Assign weights to vertices
        num_vertices = len(mesh_obj.data.vertices)
        for vert_idx in range(min(num_vertices, len(skinning_weights))):
            influences = skinning_weights[vert_idx]
            if influences and len(influences) > 0:
                for bone_idx, weight in influences:
                    if 0 <= bone_idx < num_joints and weight > 0.0001:
                        bone_name = f'P{person_idx}_Joint_{bone_idx:03d}'
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

    # Deselect for next iteration
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


# Clean scene
clean_bpy()

# Create collection for all exports
collection = bpy.data.collections.new('SAM3D_MultiExport')
bpy.context.scene.collection.children.link(collection)

# Process each person
print(f"[SAM3D Multi] Starting to process {len(people_data)} people...")
for person in people_data:
    person_idx = person.get('index', 0)
    obj_path = person.get('obj_path')
    skeleton_data = person.get('skeleton', {})

    print(f"[SAM3D Multi] Processing person {person_idx}, OBJ: {obj_path}")

    if not obj_path or not os.path.exists(obj_path):
        print(f"[SAM3D Multi] Skipping person {person_idx}: OBJ not found at {obj_path}")
        continue

    # Import OBJ mesh
    try:
        # Get mesh count before import
        meshes_before = set(obj.name for obj in bpy.context.scene.objects if obj.type == 'MESH')

        bpy.ops.wm.obj_import(filepath=obj_path)

        # Find newly imported mesh
        meshes_after = set(obj.name for obj in bpy.context.scene.objects if obj.type == 'MESH')
        new_meshes = meshes_after - meshes_before

        if not new_meshes:
            print(f"[SAM3D Multi] No mesh imported for person {person_idx}")
            continue

        mesh_obj = bpy.data.objects[list(new_meshes)[0]]
        mesh_obj.name = f'Person_{person_idx}'
        print(f"[SAM3D Multi] Imported mesh: {mesh_obj.name}")

        # Move to our collection
        if mesh_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(mesh_obj)
        collection.objects.link(mesh_obj)

    except Exception as e:
        print(f"[SAM3D Multi] Failed to import OBJ for person {person_idx}: {e}")
        continue

    # Create armature if skeleton data provided
    if skeleton_data.get('joint_positions'):
        try:
            armature_info = create_armature_for_person(person_idx, skeleton_data, collection)
            if armature_info:
                print(f"[SAM3D Multi] Created armature for person {person_idx}")
                apply_skinning_to_mesh(mesh_obj, armature_info)
        except Exception as e:
            print(f"[SAM3D Multi] Failed to create armature for person {person_idx}: {e}")

    # Make mesh double-sided AFTER skinning (so duplicated vertices inherit weights)
    make_mesh_double_sided(mesh_obj)

# Export all to single FBX
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

# Debug: show what's in the collection
print(f"[SAM3D Multi] Objects in collection before export:")
for obj in collection.objects:
    print(f"  - {obj.name} ({obj.type})")

try:
    # Select all objects in our collection
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in collection.objects:
        obj.select_set(True)

    # Export FBX with all objects
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        use_selection=True,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )

except Exception as e:
    print(f"[SAM3D Multi] Export failed: {e}")
    sys.exit(1)
