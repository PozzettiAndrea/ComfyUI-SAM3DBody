# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Multi-person processing node for SAM 3D Body.

Performs 3D human mesh reconstruction for multiple people from a single image.
"""

import os
import tempfile
import torch
import numpy as np
import cv2
from ..base import comfy_image_to_numpy, comfy_mask_to_numpy, numpy_to_comfy_image


class SAM3DBodyProcessMultiple:
    """
    Performs 3D human mesh reconstruction for multiple people.

    Takes an input image and multiple masks (one per person), processes each person,
    and outputs mesh data with each person at their model-predicted world coordinates.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM 3D Body model from Load node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image containing multiple people"
                }),
                "masks": ("MASK", {
                    "tooltip": "Batched masks - one per person (N, H, W)"
                }),
            },
            "optional": {
                "inference_type": (["full", "body"], {
                    "default": "full",
                    "tooltip": "full: body+hand decoders, body: body decoder only"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_MULTI_OUTPUT", "IMAGE")
    RETURN_NAMES = ("multi_mesh_data", "preview")
    FUNCTION = "process_multiple"
    CATEGORY = "SAM3DBody/processing"

    def _compute_bbox_from_mask(self, mask):
        """Compute bounding box from binary mask."""
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

    def _prepare_outputs(self, outputs):
        """Convert tensors to numpy and add person indices."""
        prepared = []
        for i, output in enumerate(outputs):
            prepared_output = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    prepared_output[key] = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    prepared_output[key] = value.copy()
                else:
                    prepared_output[key] = value

            prepared_output["person_index"] = i
            prepared.append(prepared_output)

        return prepared

    def process_multiple(self, model, image, masks, inference_type="full"):
        """Process image with multiple masks and reconstruct 3D meshes for all people."""

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

        # Convert ComfyUI image to numpy (BGR format)
        img_bgr = comfy_image_to_numpy(image)

        # Convert masks to numpy - shape should be (N, H, W)
        masks_np = comfy_mask_to_numpy(masks)
        if masks_np.ndim == 2:
            masks_np = masks_np[np.newaxis, ...]  # Add batch dim if single mask

        num_people = masks_np.shape[0]

        # Compute bounding boxes from each mask
        bboxes_list = []
        valid_mask_indices = []
        for i in range(num_people):
            bbox = self._compute_bbox_from_mask(masks_np[i])
            if bbox is not None:
                bboxes_list.append(bbox)
                valid_mask_indices.append(i)

        if len(bboxes_list) == 0:
            raise RuntimeError("No valid masks found (all masks are empty)")

        # Filter to valid masks only
        bboxes = np.stack(bboxes_list, axis=0)  # (N, 4)
        valid_masks = masks_np[valid_mask_indices]  # (N, H, W)

        # Add channel dimension for SAM3DBody: (N, H, W) -> (N, H, W, 1)
        masks_for_estimator = valid_masks[..., np.newaxis]

        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name

        try:
            # Process all people at once
            outputs = estimator.process_one_image(
                tmp_path,
                bboxes=bboxes,
                masks=masks_for_estimator,
                use_mask=True,
                inference_type=inference_type,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not outputs or len(outputs) == 0:
            raise RuntimeError("No people detected in image")

        # Prepare outputs (convert tensors, add indices)
        prepared_outputs = self._prepare_outputs(outputs)

        # Create combined mesh data - use model's world coordinates directly
        multi_mesh_data = {
            "num_people": len(prepared_outputs),
            "people": prepared_outputs,
            "faces": estimator.faces,
            "mhr_path": model.get("mhr_path", None),
            "all_vertices": [p["pred_vertices"] for p in prepared_outputs],
            "all_joints": [p.get("pred_joint_coords") for p in prepared_outputs],
            "all_cam_t": [p.get("pred_cam_t") for p in prepared_outputs],
        }

        # Create preview visualization
        preview = self._create_multi_person_preview(
            img_bgr, prepared_outputs, estimator.faces
        )
        preview_comfy = numpy_to_comfy_image(preview)

        return (multi_mesh_data, preview_comfy)

    def _create_multi_person_preview(self, img_bgr, outputs, faces):
        """Create a preview visualization showing all detected people."""
        try:
            from sam_3d_body.visualization.renderer import Renderer

            h, w = img_bgr.shape[:2]

            # Get vertices and camera translations
            vertices_list = [o["pred_vertices"] for o in outputs if o.get("pred_vertices") is not None]
            cam_t_list = [o["pred_cam_t"] for o in outputs if o.get("pred_cam_t") is not None]

            if len(vertices_list) == 0:
                return img_bgr

            # Get focal length from first output
            focal_length = outputs[0].get("focal_length", 5000.0)
            if isinstance(focal_length, np.ndarray):
                focal_length = float(focal_length[0])

            # Create renderer
            renderer = Renderer(
                focal_length=focal_length,
                img_w=w,
                img_h=h,
                faces=faces,
                same_mesh_color=False,
            )

            # Render all meshes
            render_result = renderer.render_rgba_multiple(
                vertices_list,
                cam_t_list,
                render_res=(w, h),
            )

            # Composite onto original image
            if render_result is not None:
                render_rgba = render_result[0] if isinstance(render_result, tuple) else render_result

                if render_rgba.shape[-1] == 4:
                    alpha = render_rgba[:, :, 3:4] / 255.0
                    render_rgb = render_rgba[:, :, :3]
                    render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
                    result = (img_bgr * (1 - alpha) + render_bgr * alpha).astype(np.uint8)
                    return result

            return img_bgr

        except Exception:
            # Fallback: draw skeleton points
            result = img_bgr.copy()

            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
            ]

            for i, output in enumerate(outputs):
                kpts_2d = output.get("pred_keypoints_2d")
                if kpts_2d is not None:
                    color = colors[i % len(colors)]
                    for pt in kpts_2d:
                        x, y = int(pt[0]), int(pt[1])
                        if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                            cv2.circle(result, (x, y), 3, color, -1)

            return result


# Register node
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyProcessMultiple": SAM3DBodyProcessMultiple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyProcessMultiple": "SAM 3D Body Process Multiple",
}
