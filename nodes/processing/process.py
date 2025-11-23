# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Processing node for SAM 3D Body.

Performs 3D human mesh reconstruction from a single image.
"""

import os
import tempfile
import torch
import numpy as np
import cv2
from pathlib import Path
from ..base import comfy_image_to_numpy, comfy_mask_to_numpy


class SAM3DBodyProcess:
    """
    Performs 3D human mesh reconstruction from a single image.

    Takes an input image and outputs 3D mesh data including vertices, faces,
    pose parameters, and camera parameters. Optionally supports detection,
    segmentation, and FOV estimation for improved results.
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
                "use_detector": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use human detector (requires detector to be loaded)"
                }),
                "use_segmentor": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use segmentation model (requires segmentor to be loaded)"
                }),
                "use_fov_estimator": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use FOV estimator for camera parameters (requires FOV estimator)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_OUTPUT", "SKELETON", "IMAGE")
    RETURN_NAMES = ("mesh_data", "skeleton", "debug_image")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody/processing"

    def _compute_bbox_from_mask(self, mask):
        """Compute bounding box from binary mask."""
        # Find non-zero pixels
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            # Empty mask, return full image bbox
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Return bbox in [x1, y1, x2, y2] format
        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)

    def process(self, model, image, bbox_threshold=0.8, inference_type="full",
                mask=None, use_detector=False, use_segmentor=False, use_fov_estimator=False):
        """Process image and reconstruct 3D human mesh."""

        print(f"[SAM3DBody] Starting 3D mesh reconstruction...")
        print(f"[SAM3DBody] Inference type: {inference_type}")

        try:
            # Import SAM 3D Body modules
            from sam_3d_body import SAM3DBodyEstimator

            # Extract model components
            sam_3d_model = model["model"]
            model_cfg = model["model_cfg"]

            # Optional components (will be None if not loaded)
            detector = None
            segmentor = None
            fov_estimator = None

            # Create estimator
            estimator = SAM3DBodyEstimator(
                sam_3d_body_model=sam_3d_model,
                model_cfg=model_cfg,
                human_detector=detector,
                human_segmentor=segmentor,
                fov_estimator=fov_estimator,
            )

            # Convert ComfyUI image to numpy (BGR format for OpenCV)
            img_bgr = comfy_image_to_numpy(image)

            # Convert mask if provided and compute bounding box
            mask_np = None
            bboxes = None
            if mask is not None:
                mask_np = comfy_mask_to_numpy(mask)
                # Take first mask if multiple
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                print(f"[SAM3DBody] Using provided mask of shape {mask_np.shape}")

                # Compute bounding box from mask (required by SAM3DBody)
                bboxes = self._compute_bbox_from_mask(mask_np)
                if bboxes is not None:
                    print(f"[SAM3DBody] Computed bounding box from mask: {bboxes[0]}")

            # Save image to temporary file (required by SAM3DBodyEstimator)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, img_bgr)
                tmp_path = tmp.name

            try:
                # Process image - returns a list of results (one per detected person)
                outputs = estimator.process_one_image(
                    tmp_path,
                    bboxes=bboxes,
                    masks=mask_np,
                    bbox_thr=bbox_threshold,
                    use_mask=(mask is not None or use_segmentor),
                    inference_type=inference_type,
                )

                # Clean up temp file
                os.unlink(tmp_path)

            except Exception as e:
                # Make sure to clean up temp file even if processing fails
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise e

            # Handle outputs - it's a list of results (one per person)
            if not outputs or len(outputs) == 0:
                raise RuntimeError("No people detected in image")

            # For now, take the first person if multiple detected
            if len(outputs) > 1:
                print(f"[SAM3DBody] [INFO] Detected {len(outputs)} people, using first one")

            output = outputs[0]

            # Extract output data
            mesh_data = {
                "vertices": output.get("pred_vertices", None),  # [N, 3] mesh vertices
                "faces": estimator.faces,  # Face indices
                "joints": output.get("pred_keypoints_3d", None),  # 3D keypoints (70 MHR70 keypoints)
                "joint_coords": output.get("pred_joint_coords", None),  # Full 127 skeleton joints
                "joint_rotations": output.get("pred_global_rots", None),  # Joint rotation matrices [127, 3, 3]
                "camera": output.get("pred_cam_t", None),  # Camera translation
                "focal_length": output.get("focal_length", None),  # Focal length
                "bbox": output.get("bbox", None),  # Bounding box
                "pose_params": {  # All pose parameters
                    "body_pose": output.get("body_pose_params", None),
                    "hand_pose": output.get("hand_pose_params", None),
                    "global_rot": output.get("global_rot", None),
                    "shape": output.get("shape_params", None),
                    "scale": output.get("scale_params", None),
                    "expr": output.get("expr_params", None),
                },
                "raw_output": output,  # Full output dict for advanced use
                "all_people": outputs,  # All detected people
            }

            # Extract skeleton data
            skeleton = {
                "joint_positions": output.get("pred_joint_coords", None),  # 127 joints [N, 3]
                "joint_rotations": output.get("pred_global_rots", None),  # 127 rotation matrices [N, 3, 3]
                "pose_params": output.get("body_pose_params", None),  # MHR pose parameters (133 dims)
                "shape_params": output.get("shape_params", None),  # Shape parameters (45 dims)
                "scale_params": output.get("scale_params", None),  # Scale parameters (28 dims)
                "hand_pose": output.get("hand_pose_params", None),  # Hand pose (54 dims)
                "global_rot": output.get("global_rot", None),  # Global rotation
                "expr_params": output.get("expr_params", None),  # Expression parameters (72 dims)
                "camera": output.get("pred_cam_t", None),  # Camera translation
                "focal_length": output.get("focal_length", None),  # Focal length
            }

            # Create debug visualization
            from ..base import numpy_to_comfy_image
            debug_img = self._create_debug_visualization(img_bgr, outputs, estimator.faces)
            debug_img_comfy = numpy_to_comfy_image(debug_img)

            print(f"[SAM3DBody] [OK] Reconstruction complete")
            if mesh_data["vertices"] is not None:
                vertices = mesh_data["vertices"]
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                print(f"[SAM3DBody] Generated mesh with {len(vertices)} vertices")
            if skeleton["joint_positions"] is not None:
                joint_positions = skeleton["joint_positions"]
                if isinstance(joint_positions, torch.Tensor):
                    joint_positions = joint_positions.cpu().numpy()
                print(f"[SAM3DBody] Extracted skeleton with {len(joint_positions)} joints")

            return (mesh_data, skeleton, debug_img_comfy)

        except ImportError as e:
            print(f"[SAM3DBody] [ERROR] Failed to import required modules")
            print(f"[SAM3DBody] Make sure sam_3d_body is properly installed")
            raise RuntimeError(f"Missing dependencies. Run install.py first.") from e

        except Exception as e:
            print(f"[SAM3DBody] [ERROR] Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _create_debug_visualization(self, img_bgr, outputs, faces):
        """Create a debug visualization of the results."""
        try:
            # TODO: Implement visualization using vendored sam_3d_body package
            # For now, return original image
            print(f"[SAM3DBody] [INFO] Debug visualization not yet implemented")
            return img_bgr
        except Exception as e:
            print(f"[SAM3DBody] [WARNING] Could not create visualization: {e}")
            # Return original image if visualization fails
            return img_bgr


class SAM3DBodyProcessAdvanced:
    """
    Advanced processing node with full control over detection, segmentation, and FOV estimation.

    Allows loading custom detector, segmentor, and FOV estimator models.
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
        # Find non-zero pixels
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            # Empty mask, return full image bbox
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Return bbox in [x1, y1, x2, y2] format
        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)

    def process_advanced(self, model, image, bbox_threshold=0.8, nms_threshold=0.3,
                        inference_type="full", detector_name="none", segmentor_name="none",
                        fov_name="none", detector_path="", segmentor_path="", fov_path="", mask=None):
        """Process image with advanced options."""

        print(f"[SAM3DBody] Starting advanced reconstruction...")

        try:
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
                    print(f"[SAM3DBody] Loading detector: {detector_name}")
                    from tools.build_detector import HumanDetector
                    detector = HumanDetector(name=detector_name, device=device, path=detector_path)

            # Load segmentor if specified
            if segmentor_name != "none":
                segmentor_path = segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
                if segmentor_path:
                    print(f"[SAM3DBody] Loading segmentor: {segmentor_name}")
                    from tools.build_sam import HumanSegmentor
                    segmentor = HumanSegmentor(name=segmentor_name, device=device, path=segmentor_path)

            # Load FOV estimator if specified
            if fov_name != "none":
                fov_path = fov_path or os.environ.get("SAM3D_FOV_PATH", "")
                if fov_path:
                    print(f"[SAM3DBody] Loading FOV estimator: {fov_name}")
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
                print(f"[SAM3DBody] Using provided mask of shape {mask_np.shape}")

                # Compute bounding box from mask (required by SAM3DBody)
                bboxes = self._compute_bbox_from_mask(mask_np)
                if bboxes is not None:
                    print(f"[SAM3DBody] Computed bounding box from mask: {bboxes[0]}")

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, img_bgr)
                tmp_path = tmp.name

            try:
                # Process with advanced options - returns a list of results (one per detected person)
                outputs = estimator.process_one_image(
                    tmp_path,
                    bboxes=bboxes,
                    masks=mask_np,
                    bbox_thr=bbox_threshold,
                    nms_thr=nms_threshold,
                    use_mask=(mask is not None or segmentor is not None),
                    inference_type=inference_type,
                )
                os.unlink(tmp_path)
            except Exception as e:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise e

            # Handle outputs - it's a list of results (one per person)
            if not outputs or len(outputs) == 0:
                raise RuntimeError("No people detected in image")

            # For now, take the first person if multiple detected
            if len(outputs) > 1:
                print(f"[SAM3DBody] [INFO] Detected {len(outputs)} people, using first one")

            output = outputs[0]

            # Extract mesh data
            mesh_data = {
                "vertices": output.get("pred_vertices", None),
                "faces": estimator.faces,
                "joints": output.get("pred_keypoints_3d", None),  # 3D keypoints (70 MHR70 keypoints)
                "joint_coords": output.get("pred_joint_coords", None),  # Full 127 skeleton joints
                "joint_rotations": output.get("pred_global_rots", None),  # Joint rotation matrices [127, 3, 3]
                "camera": output.get("pred_cam_t", None),
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

            # Extract skeleton data
            skeleton = {
                "joint_positions": output.get("pred_joint_coords", None),  # 127 joints [N, 3]
                "joint_rotations": output.get("pred_global_rots", None),  # 127 rotation matrices [N, 3, 3]
                "pose_params": output.get("body_pose_params", None),  # MHR pose parameters (133 dims)
                "shape_params": output.get("shape_params", None),  # Shape parameters (45 dims)
                "scale_params": output.get("scale_params", None),  # Scale parameters (28 dims)
                "hand_pose": output.get("hand_pose_params", None),  # Hand pose (54 dims)
                "global_rot": output.get("global_rot", None),  # Global rotation
                "expr_params": output.get("expr_params", None),  # Expression parameters (72 dims)
                "camera": output.get("pred_cam_t", None),  # Camera translation
                "focal_length": output.get("focal_length", None),  # Focal length
            }

            # Create debug visualization
            from ..base import numpy_to_comfy_image
            debug_img = self._create_debug_visualization(img_bgr, outputs, estimator.faces)
            debug_img_comfy = numpy_to_comfy_image(debug_img)

            print(f"[SAM3DBody] [OK] Advanced reconstruction complete")
            return (mesh_data, skeleton, debug_img_comfy)

        except Exception as e:
            print(f"[SAM3DBody] [ERROR] Advanced processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _create_debug_visualization(self, img_bgr, outputs, faces):
        """Create debug visualization."""
        try:
            # TODO: Implement visualization using vendored sam_3d_body package
            # For now, return original image
            print(f"[SAM3DBody] [INFO] Debug visualization not yet implemented")
            return img_bgr
        except:
            return img_bgr


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyProcess": SAM3DBodyProcess,
    "SAM3DBodyProcessAdvanced": SAM3DBodyProcessAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyProcess": "SAM 3D Body: Process Image",
    "SAM3DBodyProcessAdvanced": "SAM 3D Body: Process Image (Advanced)",
}
