"""
Blender script to export SAM3D Body mesh and skeleton to FBX file.
Adapted from ComfyUI-UniRig to work with SAM3D Body's OBJ + JSON interface.

Usage: blender --background --python blender_export_sam3d_fbx.py -- <input_obj> <output_fbx> [skeleton_json]
"""

import bpy
import sys
import os
import json
import numpy as np
from mathutils import Vector
from collections import defaultdict

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_export_sam3d_fbx.py -- <input_obj> <output_fbx> [skeleton_json]")
    sys.exit(1)

input_obj = argv[0]
output_fbx = argv[1]
skeleton_json = argv[2] if len(argv) > 2 else None

print(f"[SAM3D Blender Export] Input OBJ: {input_obj}")
print(f"[SAM3D Blender Export] Output FBX: {output_fbx}")
print(f"[SAM3D Blender Export] Skeleton JSON: {skeleton_json if skeleton_json else 'None'}")

# Load skeleton data from JSON if provided
joints = None
num_joints = 0
mesh_bounds_min = None
mesh_bounds_max = None
camera = None
focal_length = None

if skeleton_json and os.path.exists(skeleton_json):
    try:
        with open(skeleton_json, 'r') as f:
            skeleton_data = json.load(f)

        joint_positions = skeleton_data.get('joint_positions', [])
        num_joints = skeleton_data.get('num_joints', len(joint_positions))
        joint_parents_list = skeleton_data.get('joint_parents')

        # If no parent data provided in JSON, will use flat hierarchy
        if joint_parents_list is None and num_joints == 127:
            print(f"[SAM3D Blender Export] [WARNING] No joint parent hierarchy provided - using flat structure")
            print(f"[SAM3D Blender Export] [INFO] To get proper hierarchy, ensure the Export node loads it from MHR model")
        mesh_bounds_min = skeleton_data.get('mesh_vertices_bounds_min')
        mesh_bounds_max = skeleton_data.get('mesh_vertices_bounds_max')
        camera = skeleton_data.get('camera')
        focal_length = skeleton_data.get('focal_length')

        if joint_positions:
            joints = np.array(joint_positions, dtype=np.float32)
            print(f"[SAM3D Blender Export] ========== SKELETON DEBUG INFO ==========")
            print(f"[SAM3D Blender Export] Loaded skeleton with {num_joints} joints")
            print(f"[SAM3D Blender Export] Joints shape: {joints.shape}")
            print(f"[SAM3D Blender Export] Joints bounds: MIN={joints.min(axis=0)}, MAX={joints.max(axis=0)}")
            print(f"[SAM3D Blender Export] Joints center: {joints.mean(axis=0)}")
            print(f"[SAM3D Blender Export] Joints range (max-min): {joints.max(axis=0) - joints.min(axis=0)}")

            if mesh_bounds_min and mesh_bounds_max:
                mesh_min = np.array(mesh_bounds_min)
                mesh_max = np.array(mesh_bounds_max)
                print(f"[SAM3D Blender Export] ========== MESH DEBUG INFO ==========")
                print(f"[SAM3D Blender Export] Mesh bounds: MIN={mesh_min}, MAX={mesh_max}")
                print(f"[SAM3D Blender Export] Mesh center: {(mesh_min + mesh_max) / 2}")
                print(f"[SAM3D Blender Export] Mesh range (max-min): {mesh_max - mesh_min}")

                print(f"[SAM3D Blender Export] ========== COORDINATE COMPARISON ==========")
                joint_center = joints.mean(axis=0)
                mesh_center = (mesh_min + mesh_max) / 2
                print(f"[SAM3D Blender Export] Offset (joint_center - mesh_center): {joint_center - mesh_center}")
                print(f"[SAM3D Blender Export] Scale ratio (joint_range / mesh_range): {(joints.max(axis=0) - joints.min(axis=0)) / (mesh_max - mesh_min + 1e-8)}")

            if camera is not None:
                print(f"[SAM3D Blender Export] ========== CAMERA INFO ==========")
                print(f"[SAM3D Blender Export] Camera translation: {camera}")
                print(f"[SAM3D Blender Export] Focal length: {focal_length}")

            print(f"[SAM3D Blender Export] =====================================")
    except Exception as e:
        print(f"[SAM3D Blender Export] Warning: Failed to load skeleton JSON: {e}")
        import traceback
        traceback.print_exc()
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
print("[SAM3D Blender Export] Importing OBJ mesh...")
try:
    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=input_obj)

    # Get the imported object (should be the only object now)
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not imported_objects:
        raise RuntimeError("No mesh found after OBJ import")

    mesh_obj = imported_objects[0]
    mesh_obj.name = 'SAM3D_Character'

    # Move to our collection
    if mesh_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(mesh_obj)
    collection.objects.link(mesh_obj)

    vertices = mesh_obj.data.vertices
    faces = mesh_obj.data.polygons
    print(f"[SAM3D Blender Export] Imported mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Get mesh bounds for debugging
    if len(vertices) > 0:
        verts_np = np.array([v.co for v in vertices])
        print(f"[SAM3D Blender Export] ========== IMPORTED MESH VERIFICATION ==========")
        print(f"[SAM3D Blender Export] Mesh bounds in Blender: MIN={verts_np.min(axis=0)}, MAX={verts_np.max(axis=0)}")
        print(f"[SAM3D Blender Export] Mesh center in Blender: {verts_np.mean(axis=0)}")
        print(f"[SAM3D Blender Export] Mesh range in Blender: {verts_np.max(axis=0) - verts_np.min(axis=0)}")
        print(f"[SAM3D Blender Export] ========================================")

except Exception as e:
    print(f"[SAM3D Blender Export] Failed to import OBJ: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create armature from skeleton if provided
if joints is not None and num_joints > 0:
    print("[SAM3D Blender Export] Creating armature...")
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

        # SAM3D Body skeleton parameters
        extrude_size = 0.05

        # Remove default bone
        default_bone = edit_bones.get('Bone')
        if default_bone:
            edit_bones.remove(default_bone)

        # Calculate skeleton center for root bone placement
        skeleton_center = joints.mean(axis=0)
        print(f"[SAM3D Blender Export] Skeleton center: {skeleton_center}")

        # Make positions relative to skeleton center
        rel_joints = joints - skeleton_center

        # Apply coordinate system correction to match mesh orientation
        # The joint coords are already flipped (-X, -Y, -Z) from export.py, same as mesh
        # Convert to Blender Z-up with Y flipped: (-X, -Y, -Z) → (X, Z, -Y)
        rel_joints_corrected = np.zeros_like(rel_joints)
        rel_joints_corrected[:, 0] = rel_joints[:, 0]    # -X becomes X in Blender
        rel_joints_corrected[:, 1] = -rel_joints[:, 2]   # -Z becomes -Y (flip Y in Blender space)
        rel_joints_corrected[:, 2] = rel_joints[:, 1]    # -Y becomes Z in Blender

        print(f"[SAM3D Blender Export] Applied coordinate transform with Y flip: (-X, -Y, -Z) → (X, Z, -Y)")

        # Create bones with proper hierarchy
        # For each bone, calculate tail as pointing toward first child or just offset
        bones_dict = {}

        # Create all bones first (in edit mode)
        for i in range(num_joints):
            bone_name = f'Joint_{i:03d}'
            bone = edit_bones.new(bone_name)

            # Set head at relative joint position (with corrected coordinates)
            bone.head = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2]))

            # Set tail (will adjust later for proper orientation)
            bone.tail = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2] + extrude_size))

            bones_dict[bone_name] = bone

        print(f"[SAM3D Blender Export] Created {len(bones_dict)} bones")

        # Build hierarchical structure using joint parents if available
        if joint_parents_list and len(joint_parents_list) == num_joints:
            print(f"[SAM3D Blender Export] Using MHR skeleton hierarchy ({len(joint_parents_list)} joints)")

            # Apply parent relationships from MHR
            for i in range(num_joints):
                parent_idx = joint_parents_list[i]
                if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                    bone_name = f'Joint_{i:03d}'
                    parent_bone_name = f'Joint_{parent_idx:03d}'
                    bones_dict[bone_name].parent = bones_dict[parent_bone_name]
                    bones_dict[bone_name].use_connect = False

            # Find the root (joint with no parent or parent = -1)
            root_joints = [i for i in range(num_joints) if joint_parents_list[i] < 0 or joint_parents_list[i] >= num_joints]
            if root_joints:
                print(f"[SAM3D Blender Export] Root joints: {root_joints}")
            else:
                print(f"[SAM3D Blender Export] No explicit root found, using Joint_000")

        else:
            # Fallback: create flat hierarchy
            print(f"[SAM3D Blender Export] No parent hierarchy available, using flat structure")

            # Find approximate body center (pelvis area) as root
            pelvis_joints = []
            for i in range(min(20, num_joints)):
                if -0.3 < rel_joints[i, 2] < 0.3:  # Near center Z
                    pelvis_joints.append((i, rel_joints[i, 1]))

            if pelvis_joints:
                root_idx = min(pelvis_joints, key=lambda x: x[1])[0]
                root_bone_name = f'Joint_{root_idx:03d}'
                print(f"[SAM3D Blender Export] Using Joint_{root_idx:03d} as root (Y={rel_joints[root_idx, 1]:.3f})")
            else:
                root_idx = 0
                root_bone_name = 'Joint_000'
                print(f"[SAM3D Blender Export] Using Joint_000 as root (fallback)")

            # Parent all other bones to the root
            for i in range(num_joints):
                if i != root_idx:
                    bone_name = f'Joint_{i:03d}'
                    bones_dict[bone_name].parent = bones_dict[root_bone_name]
                    bones_dict[bone_name].use_connect = False

        print(f"[SAM3D Blender Export] Configured bone hierarchy")

        # Switch to object mode to set armature position
        bpy.ops.object.mode_set(mode='OBJECT')

        # Apply same coordinate transform to skeleton center for armature positioning
        # Input is already (-X, -Y, -Z), convert to Blender with Y flip: (X, Z, -Y)
        skeleton_center_corrected = np.zeros(3)
        skeleton_center_corrected[0] = skeleton_center[0]    # -X becomes X in Blender
        skeleton_center_corrected[1] = -skeleton_center[2]   # -Z becomes -Y (flip Y in Blender space)
        skeleton_center_corrected[2] = skeleton_center[1]    # -Y becomes Z in Blender

        # Move armature object to transformed skeleton center
        armature_obj.location = Vector((skeleton_center_corrected[0], skeleton_center_corrected[1], skeleton_center_corrected[2]))
        print(f"[SAM3D Blender Export] Positioned armature at corrected skeleton center: {armature_obj.location}")

        # Apply skinning weights if available
        skinning_weights = skeleton_data.get('skinning_weights')
        if skinning_weights:
            print(f"[SAM3D Blender Export] Applying skinning weights to mesh...")

            # Create vertex groups for each bone
            for i in range(num_joints):
                bone_name = f'Joint_{i:03d}'
                vertex_group = mesh_obj.vertex_groups.new(name=bone_name)

            # Assign weights to vertices
            num_vertices = len(mesh_obj.data.vertices)
            vertices_with_weights = 0
            total_influences = 0

            for vert_idx in range(min(num_vertices, len(skinning_weights))):
                influences = skinning_weights[vert_idx]
                if influences and len(influences) > 0:
                    vertices_with_weights += 1
                    for bone_idx, weight in influences:
                        if 0 <= bone_idx < num_joints and weight > 0.0001:  # Skip tiny weights
                            bone_name = f'Joint_{bone_idx:03d}'
                            vertex_group = mesh_obj.vertex_groups.get(bone_name)
                            if vertex_group:
                                vertex_group.add([vert_idx], weight, 'REPLACE')
                                total_influences += 1

            print(f"[SAM3D Blender Export] ✓ Applied {total_influences} bone influences to {vertices_with_weights} vertices")
        else:
            print(f"[SAM3D Blender Export] [WARNING] No skinning weights provided - mesh will not be rigged")

        # Deselect all
        for obj in bpy.context.selected_objects:
            obj.select_set(False)

        # Parent mesh to armature with vertex groups (proper rigging)
        mesh_obj.select_set(True)
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj

        if skinning_weights:
            # Use ARMATURE parent type to respect vertex groups
            bpy.ops.object.parent_set(type='ARMATURE')
            print("[SAM3D Blender Export] Armature created and mesh rigged with skinning weights")
        else:
            # Fallback: just name-based parenting without weights
            bpy.ops.object.parent_set(type='ARMATURE_NAME')
            print("[SAM3D Blender Export] Armature created and parented to mesh (no skinning)")

    except Exception as e:
        print(f"[SAM3D Blender Export] Warning: Armature creation failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue without armature
else:
    print("[SAM3D Blender Export] No skeleton data provided, exporting mesh only")

# Export to FBX
print("[SAM3D Blender Export] Exporting to FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    # Select all objects in our collection for export
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

    print(f"[SAM3D Blender Export] Successfully saved to: {output_fbx}")
    print("[SAM3D Blender Export] Done!")

except Exception as e:
    print(f"[SAM3D Blender Export] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
