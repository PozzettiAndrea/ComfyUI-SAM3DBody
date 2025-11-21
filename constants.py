# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Constants for SAM 3D Body ComfyUI nodes.
"""

# Timeouts (in seconds)
BLENDER_TIMEOUT = 120  # 2 minutes for Blender FBX export operations
INFERENCE_TIMEOUT = 600  # 10 minutes for SAM 3D Body inference

# Mesh processing
TARGET_FACE_COUNT = 50000  # Default target for mesh decimation

# Export settings
DEFAULT_EXTRUDE_SIZE = 0.03  # For bone visualization in FBX export

# File formats
SUPPORTED_EXPORT_FORMATS = ['fbx', 'obj', 'ply']
SUPPORTED_SKELETON_FORMATS = ['json', 'bvh', 'fbx']

# Skeleton structure
NUM_JOINTS = 127  # SAM 3D Body uses 127 joints

# MHR Model parameters
POSE_PARAM_DIMS = 133  # Body pose parameters
SHAPE_PARAM_DIMS = 45  # Shape parameters
SCALE_PARAM_DIMS = 28  # Scale parameters
EXPR_PARAM_DIMS = 50  # Expression parameters (for face)
HAND_POSE_DIMS = 90  # Hand pose parameters (45 per hand)
