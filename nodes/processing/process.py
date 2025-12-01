# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Processing node for SAM 3D Body.

Performs 3D human mesh reconstruction from a single image.
"""

import os
import tempfile
import time
import torch
import numpy as np
import cv2
import folder_paths
from ..base import comfy_image_to_numpy, comfy_mask_to_numpy


def write_debug_obj(vertices, faces, filepath, label):
    """Write debug OBJ and print bounds to help trace coordinate issues."""
    print(f"[DEBUG {label}] Bounds:")
    print(f"  X: [{vertices[:,0].min():.4f}, {vertices[:,0].max():.4f}] range={vertices[:,0].max()-vertices[:,0].min():.4f}")
    print(f"  Y: [{vertices[:,1].min():.4f}, {vertices[:,1].max():.4f}] range={vertices[:,1].max()-vertices[:,1].min():.4f}")
    print(f"  Z: [{vertices[:,2].min():.4f}, {vertices[:,2].max():.4f}] range={vertices[:,2].max()-vertices[:,2].min():.4f}")
    with open(filepath, 'w') as f:
        f.write(f"# {label}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if faces is not None:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"[DEBUG] Wrote {filepath}")


class SAM3DBodyProcess:
    """
    Performs 3D human mesh reconstruction from a single image.

    Takes an input image and outputs 3D mesh data including vertices, faces,
    pose parameters, and camera parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM 3D Body model from Load node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image containing human subject"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for human detection bounding boxes"
                }),
                "inference_type": (["full", "body", "hand"], {
                    "default": "full",
                    "tooltip": "full: body+hand decoders, body: body decoder only, hand: hand decoder only"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional segmentation mask to guide reconstruction"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_OUTPUT", "SKELETON", "IMAGE")
    RETURN_NAMES = ("mesh_data", "skeleton", "debug_image")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody/processing"

    def _compute_bbox_from_mask(self, mask):
        """Compute bounding box from binary mask."""
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)

    def process(self, model, image, bbox_threshold=0.8, inference_type="full", mask=None):
        """Process image and reconstruct 3D human mesh."""

        from sam_3d_body import SAM3DBodyEstimator

        # Extract model components
        sam_3d_model = model["model"]
        model_cfg = model["model_cfg"]

        # Create estimator
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam_3d_model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

        # Convert ComfyUI image to numpy (BGR format for OpenCV)
        img_bgr = comfy_image_to_numpy(image)

        # Convert mask if provided and compute bounding box
        mask_np = None
        bboxes = None
        if mask is not None:
            mask_np = comfy_mask_to_numpy(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            bboxes = self._compute_bbox_from_mask(mask_np)

        # Save image to temporary file (required by SAM3DBodyEstimator)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name

        try:
            outputs = estimator.process_one_image(
                tmp_path,
                bboxes=bboxes,
                masks=mask_np,
                bbox_thr=bbox_threshold,
                use_mask=(mask is not None),
                inference_type=inference_type,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not outputs or len(outputs) == 0:
            raise RuntimeError("No people detected in image")

        # Take the first person if multiple detected
        output = outputs[0]

        # Get camera translation for world-space conversion
        cam_t = output.get("pred_cam_t", None)

        print(f"[DEBUG SAM3DBodyProcess] pred_cam_t: {cam_t}")
        print(f"[DEBUG SAM3DBodyProcess] pred_cam_t shape: {cam_t.shape if cam_t is not None else None}")

        # Extract mesh data (in camera/world space)
        vertices_orig = output.get("pred_vertices", None)
        faces = estimator.faces
        print(f"[DEBUG SAM3DBodyProcess] vertices_orig shape: {vertices_orig.shape if vertices_orig is not None else None}")
        if vertices_orig is not None:
            print(f"[DEBUG SAM3DBodyProcess] vertices_orig center (mean): {vertices_orig.mean(axis=0)}")
            # DEBUG: Write OBJ after Y/Z flip (this is what MHR outputs after flip)
            write_debug_obj(vertices_orig, faces, "/tmp/debug_1_after_mhr_flip.obj", "Stage1: After MHR Y/Z flip (body-centric)")

        if vertices_orig is not None and cam_t is not None:
            vertices = vertices_orig + cam_t  # Apply camera translation to vertices
            print(f"[DEBUG SAM3DBodyProcess] vertices after cam_t center (mean): {vertices.mean(axis=0)}")
            # DEBUG: Write OBJ after cam_t translation
            write_debug_obj(vertices, faces, "/tmp/debug_2_after_cam_t.obj", "Stage2: After cam_t (world space)")
        else:
            vertices = vertices_orig
            print(f"[DEBUG SAM3DBodyProcess] vertices NOT translated (cam_t is None: {cam_t is None})")

        joints_cam = output.get("pred_keypoints_3d_cam", None)
        joints_orig = output.get("pred_keypoints_3d", None)
        print(f"[DEBUG SAM3DBodyProcess] pred_keypoints_3d_cam available: {joints_cam is not None}")
        print(f"[DEBUG SAM3DBodyProcess] pred_keypoints_3d (centered) available: {joints_orig is not None}")
        if joints_cam is not None:
            print(f"[DEBUG SAM3DBodyProcess] joints_cam[0] (pelvis): {joints_cam[0]}")
        if joints_orig is not None:
            print(f"[DEBUG SAM3DBodyProcess] joints_orig[0] (pelvis, should be ~0): {joints_orig[0]}")

        # Apply cam_t to 127-joint skeleton for world positioning
        joint_coords_orig = output.get("pred_joint_coords", None)
        if joint_coords_orig is not None and cam_t is not None:
            joint_coords = joint_coords_orig + cam_t
        else:
            joint_coords = joint_coords_orig

        mesh_data = {
            "vertices": vertices,
            "faces": estimator.faces,
            "joints": joints_cam,  # Camera-space 3D keypoints
            "joint_coords": joint_coords,  # 127-joint skeleton with world coords
            "joint_rotations": output.get("pred_global_rots", None),
            "camera": cam_t,
            "focal_length": output.get("focal_length", None),
            "bbox": output.get("bbox", None),
            "pose_params": {
                "body_pose": output.get("body_pose_params", None),
                "hand_pose": output.get("hand_pose_params", None),
                "global_rot": output.get("global_rot", None),
                "shape": output.get("shape_params", None),
                "scale": output.get("scale_params", None),
                "expr": output.get("expr_params", None),
            },
            "raw_output": output,
            "all_people": outputs,
            "mhr_path": model.get("mhr_path", None),
        }

        # Extract skeleton data (in camera/world space)
        skeleton = {
            "joint_positions": joints_cam,  # Camera-space 3D keypoints
            "joint_rotations": output.get("pred_global_rots", None),
            "pose_params": output.get("body_pose_params", None),
            "shape_params": output.get("shape_params", None),
            "scale_params": output.get("scale_params", None),
            "hand_pose": output.get("hand_pose_params", None),
            "global_rot": output.get("global_rot", None),
            "expr_params": output.get("expr_params", None),
            "camera": cam_t,
            "focal_length": output.get("focal_length", None),
        }

        # Add joint parent hierarchy from the MHR model
        try:
            if hasattr(sam_3d_model, 'mhr_head') and hasattr(sam_3d_model.mhr_head, 'mhr'):
                mhr = sam_3d_model.mhr_head.mhr
                if hasattr(mhr, 'character_torch') and hasattr(mhr.character_torch, 'skeleton'):
                    skeleton_obj = mhr.character_torch.skeleton
                    if hasattr(skeleton_obj, 'joint_parents'):
                        parent_tensor = skeleton_obj.joint_parents
                        if isinstance(parent_tensor, torch.Tensor):
                            skeleton["joint_parents"] = parent_tensor.cpu().numpy()
        except Exception:
            pass

        # Create debug visualization
        from ..base import numpy_to_comfy_image
        debug_img = self._create_debug_visualization(img_bgr, outputs, estimator.faces)
        debug_img_comfy = numpy_to_comfy_image(debug_img)

        return (mesh_data, skeleton, debug_img_comfy)

    def _create_debug_visualization(self, img_bgr, outputs, faces):
        """Create a debug visualization of the results."""
        # Return original image for now
        return img_bgr


class SAM3DBodyProcessAdvanced:
    """
    Advanced processing node with full control over detection, segmentation, and FOV estimation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM 3D Body model from Load node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image containing human subject"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for human detection"
                }),
                "nms_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Non-maximum suppression threshold for detection"
                }),
                "inference_type": (["full", "body", "hand"], {
                    "default": "full",
                    "tooltip": "Inference mode: full (body+hand), body only, or hand only"
                }),
                "detector_name": (["none", "vitdet"], {
                    "default": "none",
                    "tooltip": "Human detector to use (requires detector_path)"
                }),
                "segmentor_name": (["none", "sam2"], {
                    "default": "none",
                    "tooltip": "Segmentation model to use (requires segmentor_path)"
                }),
                "fov_name": (["none", "moge2"], {
                    "default": "none",
                    "tooltip": "FOV estimator to use (requires fov_path)"
                }),
            },
            "optional": {
                "detector_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to detector model or set SAM3D_DETECTOR_PATH env var"
                }),
                "segmentor_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to segmentor model or set SAM3D_SEGMENTOR_PATH env var"
                }),
                "fov_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to FOV model or set SAM3D_FOV_PATH env var"
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional pre-computed segmentation mask"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_OUTPUT", "SKELETON", "IMAGE")
    RETURN_NAMES = ("mesh_data", "skeleton", "debug_image")
    FUNCTION = "process_advanced"
    CATEGORY = "SAM3DBody/advanced"

    def _compute_bbox_from_mask(self, mask):
        """Compute bounding box from binary mask."""
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)

    def process_advanced(self, model, image, bbox_threshold=0.8, nms_threshold=0.3,
                        inference_type="full", detector_name="none", segmentor_name="none",
                        fov_name="none", detector_path="", segmentor_path="", fov_path="", mask=None):
        """Process image with advanced options."""

        from sam_3d_body import SAM3DBodyEstimator

        # Extract model components
        sam_3d_model = model["model"]
        model_cfg = model["model_cfg"]
        device = torch.device(model["device"])

        # Initialize optional components
        detector = None
        segmentor = None
        fov_estimator = None

        # Load detector if specified
        if detector_name != "none":
            detector_path = detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
            if detector_path:
                from tools.build_detector import HumanDetector
                detector = HumanDetector(name=detector_name, device=device, path=detector_path)

        # Load segmentor if specified
        if segmentor_name != "none":
            segmentor_path = segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
            if segmentor_path:
                from tools.build_sam import HumanSegmentor
                segmentor = HumanSegmentor(name=segmentor_name, device=device, path=segmentor_path)

        # Load FOV estimator if specified
        if fov_name != "none":
            fov_path = fov_path or os.environ.get("SAM3D_FOV_PATH", "")
            if fov_path:
                from tools.build_fov_estimator import FOVEstimator
                fov_estimator = FOVEstimator(name=fov_name, device=device, path=fov_path)

        # Create estimator with optional components
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam_3d_model,
            model_cfg=model_cfg,
            human_detector=detector,
            human_segmentor=segmentor,
            fov_estimator=fov_estimator,
        )

        # Convert image and mask
        img_bgr = comfy_image_to_numpy(image)
        mask_np = None
        bboxes = None
        if mask is not None:
            mask_np = comfy_mask_to_numpy(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            bboxes = self._compute_bbox_from_mask(mask_np)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name

        try:
            outputs = estimator.process_one_image(
                tmp_path,
                bboxes=bboxes,
                masks=mask_np,
                bbox_thr=bbox_threshold,
                nms_thr=nms_threshold,
                use_mask=(mask is not None or segmentor is not None),
                inference_type=inference_type,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not outputs or len(outputs) == 0:
            raise RuntimeError("No people detected in image")

        # Take the first person if multiple detected
        output = outputs[0]

        # Get camera translation for world-space conversion
        cam_t = output.get("pred_cam_t", None)

        # Extract mesh data (in camera/world space)
        vertices = output.get("pred_vertices", None)
        if vertices is not None and cam_t is not None:
            vertices = vertices + cam_t  # Apply camera translation to vertices

        # Apply cam_t to 127-joint skeleton for world positioning
        joint_coords_orig = output.get("pred_joint_coords", None)
        if joint_coords_orig is not None and cam_t is not None:
            joint_coords = joint_coords_orig + cam_t
        else:
            joint_coords = joint_coords_orig

        mesh_data = {
            "vertices": vertices,
            "faces": estimator.faces,
            "joints": output.get("pred_keypoints_3d_cam", None),  # Camera-space 3D keypoints
            "joint_coords": joint_coords,  # 127-joint skeleton with world coords
            "joint_rotations": output.get("pred_global_rots", None),
            "camera": cam_t,
            "focal_length": output.get("focal_length", None),
            "bbox": output.get("bbox", None),
            "pose_params": {
                "body_pose": output.get("body_pose_params", None),
                "hand_pose": output.get("hand_pose_params", None),
                "global_rot": output.get("global_rot", None),
                "shape": output.get("shape_params", None),
                "scale": output.get("scale_params", None),
                "expr": output.get("expr_params", None),
            },
            "raw_output": output,
            "all_people": outputs,
        }

        # Extract skeleton data (in camera/world space)
        skeleton = {
            "joint_positions": output.get("pred_keypoints_3d_cam", None),  # Camera-space 3D keypoints
            "joint_rotations": output.get("pred_global_rots", None),
            "pose_params": output.get("body_pose_params", None),
            "shape_params": output.get("shape_params", None),
            "scale_params": output.get("scale_params", None),
            "hand_pose": output.get("hand_pose_params", None),
            "global_rot": output.get("global_rot", None),
            "expr_params": output.get("expr_params", None),
            "camera": cam_t,
            "focal_length": output.get("focal_length", None),
        }

        # Add joint parent hierarchy from the MHR model
        try:
            if hasattr(sam_3d_model, 'mhr_head') and hasattr(sam_3d_model.mhr_head, 'mhr'):
                mhr = sam_3d_model.mhr_head.mhr
                if hasattr(mhr, 'character_torch') and hasattr(mhr.character_torch, 'skeleton'):
                    skeleton_obj = mhr.character_torch.skeleton
                    if hasattr(skeleton_obj, 'joint_parents'):
                        parent_tensor = skeleton_obj.joint_parents
                        if isinstance(parent_tensor, torch.Tensor):
                            skeleton["joint_parents"] = parent_tensor.cpu().numpy()
        except Exception:
            pass

        # Create debug visualization
        from ..base import numpy_to_comfy_image
        debug_img = self._create_debug_visualization(img_bgr, outputs, estimator.faces)
        debug_img_comfy = numpy_to_comfy_image(debug_img)

        return (mesh_data, skeleton, debug_img_comfy)

    def _create_debug_visualization(self, img_bgr, outputs, faces):
        """Create debug visualization."""
        return img_bgr


class SAM3DBodyProcessDebug:
    """
    Debug version of SAM3DBodyProcess that exposes intermediate transformation stages.

    Use this to debug orientation issues by comparing mesh/skeleton at each stage:
    - Stage 0: Raw MHR output (before any coordinate flips)
    - Stage 1: After Z-axis flip (body-centric coordinates)
    - Stage 2: After cam_t translation (world space)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM 3D Body model from Load node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image containing human subject"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for human detection bounding boxes"
                }),
                "inference_type": (["full", "body", "hand"], {
                    "default": "full",
                    "tooltip": "full: body+hand decoders, body: body decoder only, hand: hand decoder only"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional segmentation mask to guide reconstruction"
                }),
                "bone_radius": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Radius of bone cylinders in skeleton visualization"
                }),
                "joint_radius": ("FLOAT", {
                    "default": 0.015,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Radius of joint spheres in skeleton visualization"
                }),
            }
        }

    RETURN_TYPES = (
        "SAM3D_OUTPUT",      # Final mesh data
        "SKELETON",          # Final skeleton
        "IMAGE",             # Debug image
        "STRING",            # Stage 0: Raw MHR mesh STL path
        "STRING",            # Stage 1: After Z-flip mesh STL path
        "STRING",            # Stage 2: World space mesh STL path
        "STRING",            # Stage 0: Raw skeleton STL path
        "STRING",            # Stage 1: After Z-flip skeleton STL path
        "STRING",            # Stage 2: World space skeleton STL path
    )
    RETURN_NAMES = (
        "mesh_data",
        "skeleton",
        "debug_image",
        "stage0_mesh_stl",
        "stage1_mesh_stl",
        "stage2_mesh_stl",
        "stage0_skel_stl",
        "stage1_skel_stl",
        "stage2_skel_stl",
    )
    FUNCTION = "process_debug"
    CATEGORY = "SAM3DBody/debug"

    def _compute_bbox_from_mask(self, mask):
        """Compute bounding box from binary mask."""
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)

    def _get_joint_parents(self, sam_3d_model):
        """Extract joint parent hierarchy from the MHR model."""
        try:
            if hasattr(sam_3d_model, 'mhr_head') and hasattr(sam_3d_model.mhr_head, 'mhr'):
                mhr = sam_3d_model.mhr_head.mhr
                if hasattr(mhr, 'character_torch') and hasattr(mhr.character_torch, 'skeleton'):
                    skeleton_obj = mhr.character_torch.skeleton
                    if hasattr(skeleton_obj, 'joint_parents'):
                        parent_tensor = skeleton_obj.joint_parents
                        if isinstance(parent_tensor, torch.Tensor):
                            return parent_tensor.cpu().numpy()
        except Exception:
            pass
        return None

    def _create_mesh_stage(self, stage_name, vertices, faces):
        """Create a mesh stage dictionary."""
        if vertices is None:
            return None
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        return {
            "stage": stage_name,
            "vertices": vertices,
            "faces": faces,
        }

    def process_debug(self, model, image, bbox_threshold=0.8, inference_type="full",
                      mask=None, bone_radius=0.01, joint_radius=0.015):
        """Process image and expose all intermediate transformation stages."""
        from .skeleton_mesh import create_skeleton_mesh
        from sam_3d_body import SAM3DBodyEstimator

        # Extract model components
        sam_3d_model = model["model"]
        model_cfg = model["model_cfg"]

        # Create estimator
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam_3d_model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

        # Convert ComfyUI image to numpy (BGR format for OpenCV)
        img_bgr = comfy_image_to_numpy(image)

        # Convert mask if provided and compute bounding box
        mask_np = None
        bboxes = None
        if mask is not None:
            mask_np = comfy_mask_to_numpy(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            bboxes = self._compute_bbox_from_mask(mask_np)

        # Save image to temporary file (required by SAM3DBodyEstimator)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name

        try:
            outputs = estimator.process_one_image(
                tmp_path,
                bboxes=bboxes,
                masks=mask_np,
                bbox_thr=bbox_threshold,
                use_mask=(mask is not None),
                inference_type=inference_type,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not outputs or len(outputs) == 0:
            raise RuntimeError("No people detected in image")

        # Take the first person if multiple detected
        output = outputs[0]

        # Get camera translation for world-space conversion
        cam_t = output.get("pred_cam_t", None)
        faces = estimator.faces

        # Extract mesh data
        vertices_orig = output.get("pred_vertices", None)
        if vertices_orig is not None and cam_t is not None:
            vertices = vertices_orig + cam_t
        else:
            vertices = vertices_orig

        # Get intermediate stages
        intermediate_stages = output.get("intermediate_stages", {})

        # Get joint parents for skeleton visualization
        joint_parents = self._get_joint_parents(sam_3d_model)

        # Apply cam_t to 127-joint skeleton for world positioning
        joint_coords_orig = output.get("pred_joint_coords", None)
        if joint_coords_orig is not None and cam_t is not None:
            joint_coords = joint_coords_orig + cam_t
        else:
            joint_coords = joint_coords_orig

        # Build stage outputs
        stage0_raw = intermediate_stages.get("stage_0_raw", {})
        stage1_zflip = intermediate_stages.get("stage_1_z_flip", {})

        # Import STL writer
        from .skeleton_mesh import write_stl

        # Get output directory and create timestamp for unique filenames
        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time())

        # Helper to ensure vertices are numpy
        def to_numpy(arr):
            if arr is None:
                return None
            if isinstance(arr, torch.Tensor):
                return arr.cpu().numpy()
            return np.asarray(arr)

        # Write mesh STL files
        stl_paths = []

        # Stage 0: Raw MHR mesh
        stage0_verts = to_numpy(stage0_raw.get("vertices"))
        if stage0_verts is not None and len(stage0_verts) > 0:
            path = os.path.join(output_dir, f"debug_stage0_mesh_raw_{timestamp}.stl")
            write_stl(path, stage0_verts, faces)
            stl_paths.append(path)
            print(f"[SAM3DBodyProcessDebug] Wrote {path}")
        else:
            stl_paths.append("")

        # Stage 1: After Z-flip mesh
        stage1_verts = to_numpy(stage1_zflip.get("vertices"))
        if stage1_verts is not None and len(stage1_verts) > 0:
            path = os.path.join(output_dir, f"debug_stage1_mesh_zflip_{timestamp}.stl")
            write_stl(path, stage1_verts, faces)
            stl_paths.append(path)
            print(f"[SAM3DBodyProcessDebug] Wrote {path}")
        else:
            stl_paths.append("")

        # Stage 2: World space mesh
        if vertices is not None and len(vertices) > 0:
            path = os.path.join(output_dir, f"debug_stage2_mesh_world_{timestamp}.stl")
            write_stl(path, vertices, faces)
            stl_paths.append(path)
            print(f"[SAM3DBodyProcessDebug] Wrote {path}")
        else:
            stl_paths.append("")

        # Skeleton mesh stages
        stage0_joints = to_numpy(stage0_raw.get("joints"))
        stage1_joints = to_numpy(stage1_zflip.get("joints"))

        # Stage 0: Raw skeleton
        if stage0_joints is not None and joint_parents is not None and len(stage0_joints) > 0:
            skel_mesh = create_skeleton_mesh(stage0_joints, joint_parents, bone_radius, joint_radius)
            path = os.path.join(output_dir, f"debug_stage0_skel_raw_{timestamp}.stl")
            write_stl(path, skel_mesh["vertices"], skel_mesh["faces"])
            stl_paths.append(path)
            print(f"[SAM3DBodyProcessDebug] Wrote {path}")
        else:
            stl_paths.append("")

        # Stage 1: Z-flip skeleton
        if stage1_joints is not None and joint_parents is not None and len(stage1_joints) > 0:
            skel_mesh = create_skeleton_mesh(stage1_joints, joint_parents, bone_radius, joint_radius)
            path = os.path.join(output_dir, f"debug_stage1_skel_zflip_{timestamp}.stl")
            write_stl(path, skel_mesh["vertices"], skel_mesh["faces"])
            stl_paths.append(path)
            print(f"[SAM3DBodyProcessDebug] Wrote {path}")
        else:
            stl_paths.append("")

        # Stage 2: World space skeleton
        if joint_coords is not None and joint_parents is not None and len(joint_coords) > 0:
            skel_mesh = create_skeleton_mesh(joint_coords, joint_parents, bone_radius, joint_radius)
            path = os.path.join(output_dir, f"debug_stage2_skel_world_{timestamp}.stl")
            write_stl(path, skel_mesh["vertices"], skel_mesh["faces"])
            stl_paths.append(path)
            print(f"[SAM3DBodyProcessDebug] Wrote {path}")
        else:
            stl_paths.append("")

        # Build final mesh_data output (same as regular node)
        joints_cam = output.get("pred_keypoints_3d_cam", None)
        mesh_data = {
            "vertices": vertices,
            "faces": faces,
            "joints": joints_cam,
            "joint_coords": joint_coords,
            "joint_rotations": output.get("pred_global_rots", None),
            "camera": cam_t,
            "focal_length": output.get("focal_length", None),
            "bbox": output.get("bbox", None),
            "pose_params": {
                "body_pose": output.get("body_pose_params", None),
                "hand_pose": output.get("hand_pose_params", None),
                "global_rot": output.get("global_rot", None),
                "shape": output.get("shape_params", None),
                "scale": output.get("scale_params", None),
                "expr": output.get("expr_params", None),
            },
            "raw_output": output,
            "all_people": outputs,
            "mhr_path": model.get("mhr_path", None),
            "intermediate_stages": intermediate_stages,
        }

        # Build skeleton output (same as regular node)
        skeleton = {
            "joint_positions": joints_cam,
            "joint_rotations": output.get("pred_global_rots", None),
            "joint_parents": joint_parents,
            "pose_params": output.get("body_pose_params", None),
            "shape_params": output.get("shape_params", None),
            "scale_params": output.get("scale_params", None),
            "hand_pose": output.get("hand_pose_params", None),
            "global_rot": output.get("global_rot", None),
            "expr_params": output.get("expr_params", None),
            "camera": cam_t,
            "focal_length": output.get("focal_length", None),
        }

        # Create debug visualization
        from ..base import numpy_to_comfy_image
        debug_img_comfy = numpy_to_comfy_image(img_bgr)

        return (
            mesh_data,
            skeleton,
            debug_img_comfy,
            stl_paths[0],  # stage0_mesh_stl
            stl_paths[1],  # stage1_mesh_stl
            stl_paths[2],  # stage2_mesh_stl
            stl_paths[3],  # stage0_skel_stl
            stl_paths[4],  # stage1_skel_stl
            stl_paths[5],  # stage2_skel_stl
        )


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyProcess": SAM3DBodyProcess,
    "SAM3DBodyProcessAdvanced": SAM3DBodyProcessAdvanced,
    "SAM3DBodyProcessDebug": SAM3DBodyProcessDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyProcess": "SAM 3D Body: Process Image",
    "SAM3DBodyProcessAdvanced": "SAM 3D Body: Process Image (Advanced)",
    "SAM3DBodyProcessDebug": "SAM 3D Body: Process Image (Debug Stages)",
}
