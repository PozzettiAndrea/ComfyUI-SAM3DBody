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
                "depth_map": ("IMAGE", {
                    "tooltip": "Depth map from Depth Anything V3 (Raw mode) for scale correction - helps fix children/small people appearing too large"
                }),
                "adjust_position_from_depth": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adjust Z-position of each person based on depth map (requires depth_map)"
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

    def _sample_depth_at_point(self, depth_map, x, y, depth_h, depth_w, img_h, img_w):
        """Sample depth at a 2D point, handling coordinate scaling."""
        if x < 0 or y < 0 or x >= img_w or y >= img_h:
            return None

        # Scale to depth map coordinates
        dx = int(x * depth_w / img_w)
        dy = int(y * depth_h / img_h)
        dx = max(0, min(depth_w - 1, dx))
        dy = max(0, min(depth_h - 1, dy))

        depth = depth_map[dy, dx]
        return depth if depth > 0 else None

    def _compute_bone_scale(self, keypoints_2d, joint_coords_3d, depth_map, focal_length,
                            depth_h, depth_w, img_h, img_w, joint_pair):
        """
        Compute scale factor for a single bone using pinhole camera model.

        Returns (scale_factor, actual_3d_length, predicted_3d_length, depth) or None if invalid.
        """
        idx1, idx2 = joint_pair

        if idx1 >= len(keypoints_2d) or idx2 >= len(keypoints_2d):
            return None
        if idx1 >= len(joint_coords_3d) or idx2 >= len(joint_coords_3d):
            return None

        # 2D keypoint positions
        pt1_2d = keypoints_2d[idx1][:2]
        pt2_2d = keypoints_2d[idx2][:2]

        # Check visibility (points within image)
        if not (0 <= pt1_2d[0] < img_w and 0 <= pt1_2d[1] < img_h):
            return None
        if not (0 <= pt2_2d[0] < img_w and 0 <= pt2_2d[1] < img_h):
            return None

        # 2D pixel distance
        pixel_dist = np.sqrt((pt2_2d[0] - pt1_2d[0])**2 + (pt2_2d[1] - pt1_2d[1])**2)
        if pixel_dist < 5:  # Too small to be reliable
            return None

        # Sample depth at midpoint of the bone
        mid_x = (pt1_2d[0] + pt2_2d[0]) / 2
        mid_y = (pt1_2d[1] + pt2_2d[1]) / 2
        depth = self._sample_depth_at_point(depth_map, mid_x, mid_y, depth_h, depth_w, img_h, img_w)
        if depth is None or depth <= 0:
            return None

        # Predicted 3D distance from SAM3DBody
        pt1_3d = joint_coords_3d[idx1]
        pt2_3d = joint_coords_3d[idx2]
        predicted_3d = np.sqrt(np.sum((pt2_3d - pt1_3d)**2))
        if predicted_3d < 0.01:  # Too small
            return None

        # Actual 3D distance using pinhole camera model
        # X_3d = x_2d * Z / f
        actual_3d = (pixel_dist * depth) / focal_length

        # Scale factor
        scale = actual_3d / predicted_3d

        return {
            "scale": scale,
            "actual_3d": actual_3d,
            "predicted_3d": predicted_3d,
            "depth": depth,
            "pixel_dist": pixel_dist,
        }

    def _apply_depth_scale_correction(self, outputs, depth_map, adjust_position=False):
        """
        Correct mesh scales using pinhole camera model with metric depth.

        For each person:
        1. Measure bone length in 2D pixels (from pred_keypoints_2d)
        2. Get predicted 3D bone length (from pred_joint_coords)
        3. Calculate actual 3D bone length: actual = (pixel_dist × depth) / focal_length
        4. Scale factor = actual / predicted
        """
        if depth_map is None or len(outputs) == 0:
            return outputs

        depth_h, depth_w = depth_map.shape

        # Estimate image dimensions from keypoints range
        all_kpts = []
        for output in outputs:
            kpts = output.get("pred_keypoints_2d")
            if kpts is not None:
                all_kpts.extend(kpts[:, :2].tolist())
        if all_kpts:
            all_kpts = np.array(all_kpts)
            img_w = int(np.max(all_kpts[:, 0]) * 1.1)
            img_h = int(np.max(all_kpts[:, 1]) * 1.1)
        else:
            img_w, img_h = depth_w, depth_h

        # Bone pairs to try (in order of preference for torso stability)
        # (joint1_idx, joint2_idx): left_hip-left_shoulder, right_hip-right_shoulder, hip-hip, shoulder-shoulder
        bone_pairs = [
            (9, 5),   # left_hip to left_shoulder
            (10, 6),  # right_hip to right_shoulder
            (9, 10),  # left_hip to right_hip
            (5, 6),   # left_shoulder to right_shoulder
        ]

        # 1. Compute scale factors for each person
        person_data = []
        for i, output in enumerate(outputs):
            keypoints_2d = output.get("pred_keypoints_2d")
            joint_coords_3d = output.get("pred_joint_coords")
            focal_length = output.get("focal_length", 5000.0)

            if isinstance(focal_length, np.ndarray):
                focal_length = float(focal_length.flatten()[0])

            # Try each bone pair until we get a valid measurement
            bone_result = None
            bone_used = None
            for pair in bone_pairs:
                if keypoints_2d is None or joint_coords_3d is None:
                    continue
                result = self._compute_bone_scale(
                    keypoints_2d, joint_coords_3d, depth_map, focal_length,
                    depth_h, depth_w, img_h, img_w, pair
                )
                if result is not None:
                    bone_result = result
                    bone_used = pair
                    break

            if bone_result is None:
                # Fallback: no valid bone measurement
                person_data.append({
                    "scale": 1.0,
                    "depth": np.median(depth_map),
                    "actual_3d": None,
                    "predicted_3d": None,
                    "bone_used": None,
                    "valid": False,
                })
            else:
                person_data.append({
                    "scale": bone_result["scale"],
                    "depth": bone_result["depth"],
                    "actual_3d": bone_result["actual_3d"],
                    "predicted_3d": bone_result["predicted_3d"],
                    "pixel_dist": bone_result["pixel_dist"],
                    "bone_used": bone_used,
                    "valid": True,
                })

        # 2. Normalize scales relative to median (so relative sizes are preserved)
        valid_scales = [p["scale"] for p in person_data if p["valid"]]
        if len(valid_scales) == 0:
            print("[SAM3DBody] Warning: No valid bone measurements, skipping scale correction")
            return outputs

        median_scale = np.median(valid_scales)

        # 3. Apply scale and position correction to each mesh
        for i, output in enumerate(outputs):
            data = person_data[i]

            if not data["valid"]:
                print(f"[SAM3DBody] Person {i}: No valid bone measurement, skipping")
                continue

            # Normalize scale relative to median
            scale_factor = data["scale"] / median_scale

            # Scale vertices around mesh centroid
            vertices = output.get("pred_vertices")
            if vertices is not None:
                centroid = vertices.mean(axis=0)
                output["pred_vertices"] = (vertices - centroid) * scale_factor + centroid

            # Also scale joint coordinates if present
            joints = output.get("pred_joint_coords")
            if joints is not None:
                joint_centroid = joints.mean(axis=0)
                output["pred_joint_coords"] = (joints - joint_centroid) * scale_factor + joint_centroid

            # Adjust Z-position if enabled
            if adjust_position:
                cam_t = output.get("pred_cam_t")
                if cam_t is not None:
                    output["pred_cam_t"] = np.array([cam_t[0], cam_t[1], data["depth"]])

            # Calculate mesh height from vertices
            mesh_height = None
            if vertices is not None:
                mesh_height = float(np.max(output["pred_vertices"][:, 1]) - np.min(output["pred_vertices"][:, 1]))

            # Store metadata for reference
            output["depth_scale_factor"] = scale_factor
            output["measured_depth"] = data["depth"]
            output["mesh_height"] = mesh_height

            bone_name = f"{data['bone_used'][0]}-{data['bone_used'][1]}" if data["bone_used"] else "none"
            height_str = f", height={mesh_height:.2f}m" if mesh_height else ""
            print(f"[SAM3DBody] Person {i}: depth={data['depth']:.2f}m, "
                  f"bone_3d={data['actual_3d']:.3f}m (pred={data['predicted_3d']:.3f}m), "
                  f"scale={scale_factor:.3f}{height_str} (bone: {bone_name})")

        return outputs

    def process_multiple(self, model, image, masks, inference_type="full", depth_map=None, adjust_position_from_depth=False):
        """Process image with multiple masks and reconstruct 3D meshes for all people."""

        from sam_3d_body import SAM3DBodyEstimator

        # Process depth map input if provided
        depth_map_np = None
        if depth_map is not None:
            # Convert ComfyUI IMAGE tensor [B, H, W, C] to numpy depth map [H, W]
            if isinstance(depth_map, torch.Tensor):
                depth_map_np = depth_map[0, :, :, 0].cpu().numpy()
            else:
                depth_map_np = depth_map[0, :, :, 0] if depth_map.ndim == 4 else depth_map[:, :, 0]
            print(f"[SAM3DBody] Depth map provided: shape={depth_map_np.shape}, range=[{depth_map_np.min():.2f}, {depth_map_np.max():.2f}]")
            if adjust_position_from_depth:
                print("[SAM3DBody] Position adjustment from depth enabled")

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

        # Apply depth-based scale correction if depth map provided
        if depth_map_np is not None:
            prepared_outputs = self._apply_depth_scale_correction(
                prepared_outputs, depth_map_np, adjust_position=adjust_position_from_depth
            )

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
