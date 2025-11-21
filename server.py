# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Server extension for SAM3D Body to handle posed FBX exports.
"""

import os
import json
import time
import tempfile
import subprocess
import torch
import folder_paths
from aiohttp import web
from server import PromptServer

from .constants import BLENDER_TIMEOUT
from .nodes.base import BLENDER_EXE, BLENDER_SCRIPT


routes = PromptServer.instance.routes


@routes.post('/sam3d/save_glb')
async def save_glb(request):
    """
    Save GLB file from viewer to output directory.

    Receives:
        - glb_data: Base64 encoded GLB binary data
        - filename: Output filename

    Returns:
        - success: boolean
        - filename: Output filename
        - error: Error message if failed
    """
    try:
        data = await request.json()
        glb_data_b64 = data.get('glb_data')
        filename = data.get('filename')

        if not glb_data_b64 or not filename:
            return web.json_response({
                'success': False,
                'error': 'Missing required parameters'
            }, status=400)

        # Ensure filename has .glb extension
        if not filename.endswith('.glb'):
            filename = filename + '.glb'

        # Decode base64 data
        import base64
        glb_bytes = base64.b64decode(glb_data_b64)

        # Save to output directory
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'wb') as f:
            f.write(glb_bytes)

        print(f"[SAM3DBody Server] ✓ Saved GLB to: {output_path}")

        return web.json_response({
            'success': True,
            'filename': filename
        })

    except Exception as e:
        print(f"[SAM3DBody Server] Error saving GLB: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)


@routes.post('/sam3d/export_posed_fbx')
async def export_posed_fbx(request):
    """
    Export FBX with applied bone transforms from the viewer.

    Receives:
        - fbx_filename: Original FBX filename in output directory
        - bone_transforms: Dict of bone transforms (position, quaternion, scale deltas)
        - output_filename: New filename for the posed FBX

    Returns:
        - success: boolean
        - filename: Output filename
        - error: Error message if failed
    """
    try:
        data = await request.json()
        fbx_filename = data.get('fbx_filename')
        bone_transforms = data.get('bone_transforms')
        output_filename = data.get('output_filename')

        if not fbx_filename or not bone_transforms or not output_filename:
            return web.json_response({
                'success': False,
                'error': 'Missing required parameters'
            }, status=400)

        # Validate input FBX exists
        output_dir = folder_paths.get_output_directory()
        input_fbx_path = os.path.join(output_dir, fbx_filename)

        if not os.path.exists(input_fbx_path):
            return web.json_response({
                'success': False,
                'error': f'Input FBX not found: {fbx_filename}'
            }, status=404)

        # Ensure output filename has .fbx extension
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'

        output_fbx_path = os.path.join(output_dir, output_filename)

        print(f"[SAM3DBody Server] Exporting posed FBX...")
        print(f"[SAM3DBody Server] Input: {fbx_filename}")
        print(f"[SAM3DBody Server] Output: {output_filename}")
        print(f"[SAM3DBody Server] Bone transforms: {len(bone_transforms)}")

        # Save bone transforms to temporary JSON file
        temp_dir = folder_paths.get_temp_directory()
        transforms_json_path = os.path.join(temp_dir, f"bone_transforms_{int(time.time())}.json")

        with open(transforms_json_path, 'w') as f:
            json.dump(bone_transforms, f)

        print(f"[SAM3DBody Server] Saved transforms: {transforms_json_path}")

        try:
            # Check if Blender is available
            if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
                return web.json_response({
                    'success': False,
                    'error': 'Blender not found. Cannot export posed FBX.'
                }, status=500)

            # Create Blender script path for applying pose
            blender_pose_script = os.path.join(
                os.path.dirname(BLENDER_SCRIPT),
                'blender_apply_pose.py'
            )

            if not os.path.exists(blender_pose_script):
                return web.json_response({
                    'success': False,
                    'error': f'Blender pose script not found: {blender_pose_script}'
                }, status=500)

            # Run Blender with pose application script
            cmd = [
                BLENDER_EXE,
                '--background',
                '--python', blender_pose_script,
                '--',
                input_fbx_path,
                output_fbx_path,
                transforms_json_path,
            ]

            print(f"[SAM3DBody Server] Running Blender pose export...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)

            # Print Blender output for debugging
            if result.stdout:
                print(f"[SAM3DBody Server] Blender stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[SAM3DBody Server] Blender stderr:\n{result.stderr}")

            if result.returncode != 0:
                return web.json_response({
                    'success': False,
                    'error': f'Blender export failed with return code {result.returncode}'
                }, status=500)

            # Verify output file was created
            if not os.path.exists(output_fbx_path):
                return web.json_response({
                    'success': False,
                    'error': 'Export completed but output file not found'
                }, status=500)

            print(f"[SAM3DBody Server] ✓ Successfully exported posed FBX: {output_filename}")

            return web.json_response({
                'success': True,
                'filename': output_filename
            })

        finally:
            # Clean up temporary transforms file
            if os.path.exists(transforms_json_path):
                os.unlink(transforms_json_path)

    except Exception as e:
        print(f"[SAM3DBody Server] Error exporting posed FBX: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)


print("[SAM3DBody] Registered server routes:")
print("[SAM3DBody]   POST /sam3d/save_glb")
print("[SAM3DBody]   POST /sam3d/export_posed_fbx")
