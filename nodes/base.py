# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
#
# Blender auto-detection logic adapted from:
# ComfyUI-UniRig by Tsinghua University and Tripo AI
# https://github.com/VAST-AI-Research/UniRig
# Licensed under MIT License
"""
Base utilities for ComfyUI SAM 3D Body nodes.

Provides conversion functions between ComfyUI tensor formats and other formats
(PIL, numpy, OpenCV) commonly used in computer vision tasks.
Also handles Blender auto-detection and installation for FBX export.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import folder_paths


def comfy_image_to_pil(image):
    """
    Convert ComfyUI image tensor to PIL Image.

    Args:
        image: ComfyUI image tensor [B, H, W, C] in range [0, 1]

    Returns:
        PIL Image in RGB format
    """
    # Take first image from batch
    img_np = image[0].cpu().numpy()
    # Convert from [0, 1] to [0, 255]
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def pil_to_comfy_image(pil_image):
    """
    Convert PIL Image to ComfyUI image tensor.

    Args:
        pil_image: PIL Image in RGB format

    Returns:
        ComfyUI image tensor [1, H, W, C] in range [0, 1]
    """
    # Convert to numpy and normalize
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    # Add batch dimension: [H, W, C] -> [1, H, W, C]
    return torch.from_numpy(img_np).unsqueeze(0)


def comfy_image_to_numpy(image):
    """
    Convert ComfyUI image tensor to numpy array in BGR format (OpenCV format).

    Args:
        image: ComfyUI image tensor [B, H, W, C] in range [0, 1]

    Returns:
        numpy array in BGR format [H, W, C] in range [0, 255]
    """
    # Take first image from batch
    img_np = image[0].cpu().numpy()
    # Convert from [0, 1] to [0, 255]
    img_np = (img_np * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    img_bgr = img_np[..., ::-1].copy()
    return img_bgr


def numpy_to_comfy_image(np_image):
    """
    Convert numpy array in BGR format to ComfyUI image tensor.

    Args:
        np_image: numpy array in BGR format [H, W, C] in range [0, 255]

    Returns:
        ComfyUI image tensor [1, H, W, C] in range [0, 1]
    """
    # Convert BGR to RGB
    img_rgb = np_image[..., ::-1].copy()
    # Normalize to [0, 1]
    img_rgb = img_rgb.astype(np.float32) / 255.0
    # Add batch dimension
    return torch.from_numpy(img_rgb).unsqueeze(0)


def comfy_mask_to_numpy(mask):
    """
    Convert ComfyUI mask to numpy array.

    Args:
        mask: ComfyUI mask tensor [N, H, W] in range [0, 1]

    Returns:
        numpy array [N, H, W] in range [0, 1]
    """
    return mask.cpu().numpy()


def numpy_to_comfy_mask(np_mask):
    """
    Convert numpy array to ComfyUI mask.

    Args:
        np_mask: numpy array [H, W] or [N, H, W] in range [0, 1]

    Returns:
        ComfyUI mask tensor [N, H, W] in range [0, 1]
    """
    # Add batch dimension if needed
    if np_mask.ndim == 2:
        np_mask = np.expand_dims(np_mask, axis=0)
    return torch.from_numpy(np_mask.astype(np.float32))


def vertices_to_point_cloud(vertices):
    """
    Convert mesh vertices to point cloud format for visualization.

    Args:
        vertices: numpy array of shape [N, 3] representing mesh vertices

    Returns:
        Dictionary containing point cloud data
    """
    return {
        "points": vertices,
        "colors": None,  # Can be added if vertex colors are available
    }

