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

    # Big bones - prioritize upper body (more visible), then torso, then legs
    BIG_BONES = [
        (5, 6),    # shoulder to shoulder (usually visible)
        (5, 7),    # left_shoulder to left_elbow (upper arm)
        (6, 8),    # right_shoulder to right_elbow (upper arm)
        (9, 5),    # left_hip to left_shoulder (torso side)
        (10, 6),   # right_hip to right_shoulder (torso side)
        (9, 10),   # hip to hip
        (9, 11),   # left_hip to left_knee (thigh)
        (10, 12),  # right_hip to right_knee (thigh)
    ]

    # Joint names for debug output
    JOINT_NAMES = {
        5: "left_shoulder", 6: "right_shoulder",
        7: "left_elbow", 8: "right_elbow",
        9: "left_hip", 10: "right_hip",
        11: "left_knee", 12: "right_knee",
    }

    def _compute_mask_depth_and_height(self, mask, depth_map, focal_length, img_h, img_w):
        """
        Compute person's depth and actual height using mask and depth map.

        This is more robust than bone-based measurement because:
        - Every pixel in mask IS this person's visible surface
        - Median depth handles outliers (extended arms)
        - Mask height + depth gives actual height via pinhole model

        Returns dict with depth, pixel_height, actual_height_m, or None if invalid.
        """
        if mask is None or depth_map is None:
            return None

        mask_h, mask_w = mask.shape
        depth_h, depth_w = depth_map.shape

        # Resize mask to depth map resolution if needed
        if (mask_h, mask_w) != (depth_h, depth_w):
            mask_resized = cv2.resize(mask.astype(np.float32), (depth_w, depth_h),
                                      interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        # Get all depth values within mask
        mask_bool = mask_resized > 0.5
        if not np.any(mask_bool):
            return None

        mask_depths = depth_map[mask_bool]
        valid_depths = mask_depths[mask_depths > 0]
        if len(valid_depths) == 0:
            return None

        # Median depth (robust to outliers like extended arms)
        median_depth = float(np.median(valid_depths))

        # Compute mask height in pixels (in original image coordinates)
        mask_rows = np.where(np.any(mask > 0.5, axis=1))[0]
        if len(mask_rows) < 2:
            return None

        pixel_height = float(mask_rows[-1] - mask_rows[0])

        # Scale pixel height to actual image coordinates
        pixel_height_img = pixel_height * img_h / mask_h

        # Convert to meters using pinhole camera model
        # actual_height = (pixel_height * depth) / focal_length
        actual_height_m = (pixel_height_img * median_depth) / focal_length

        # Debug: show formula values
        print(f"  [DEBUG] focal_length={focal_length:.1f}, mask_height_px={pixel_height_img:.1f}, "
              f"median_depth={median_depth:.3f}m")
        print(f"  [DEBUG] actual_height = {pixel_height_img:.1f} × {median_depth:.3f} / {focal_length:.1f} = {actual_height_m:.3f}m")

        return {
            "depth": median_depth,
            "pixel_height": pixel_height_img,
            "actual_height_m": actual_height_m,
        }

    def _compute_scale_from_depth_ratios(self, output, depth_map, mask, img_h, img_w):
        """
        Compute scale using depth ratios between joint pairs.

        For two visible joints j1, j2:
            D1 = z1 * scale + tz  (depth at j1's 2D location)
            D2 = z2 * scale + tz  (depth at j2's 2D location)

        Taking difference eliminates tz:
            D1 - D2 = (z1 - z2) * scale
            scale = (D1 - D2) / (z1 - z2)

        This gives scale from any pair of visible joints.
        Occluded joints give inconsistent scale estimates -> detected as outliers.

        Returns dict with scale, tz, visible_joints info, or None if insufficient data.
        """
        keypoints_2d = output.get("pred_keypoints_2d")
        joints_3d = output.get("pred_joint_coords")

        if keypoints_2d is None or joints_3d is None:
            return None

        depth_h, depth_w = depth_map.shape

        # Sample depth for each joint (if in mask)
        joint_depths = {}
        joint_mesh_z = {}

        for j in range(len(keypoints_2d)):
            u, v = keypoints_2d[j]

            # Check mask
            if mask is not None:
                mask_h, mask_w = mask.shape
                u_mask = int(u * mask_w / img_w)
                v_mask = int(v * mask_h / img_h)
                if not (0 <= u_mask < mask_w and 0 <= v_mask < mask_h):
                    continue
                if mask[v_mask, u_mask] < 0.5:
                    continue

            # Sample depth
            u_depth = int(u * depth_w / img_w)
            v_depth = int(v * depth_h / img_h)
            if not (0 <= u_depth < depth_w and 0 <= v_depth < depth_h):
                continue

            D = depth_map[v_depth, u_depth]
            if D > 0:
                joint_depths[j] = D
                joint_mesh_z[j] = joints_3d[j, 2]  # Mesh-local Z

        if len(joint_depths) < 2:
            return None

        # Compute scale from all pairs
        scale_estimates = []
        pair_info = []

        joints_list = list(joint_depths.keys())
        for i in range(len(joints_list)):
            for k in range(i + 1, len(joints_list)):
                j1, j2 = joints_list[i], joints_list[k]

                z1 = joint_mesh_z[j1]
                z2 = joint_mesh_z[j2]
                D1 = joint_depths[j1]
                D2 = joint_depths[j2]

                dz = z1 - z2  # Mesh Z difference
                dD = D1 - D2  # Depth map difference

                if abs(dz) < 0.01:  # Joints at same mesh depth
                    continue

                s = dD / dz

                if s > 0:  # Valid positive scale
                    scale_estimates.append(s)
                    pair_info.append({
                        'joints': (j1, j2),
                        'scale': s,
                        'dD': dD,
                        'dz': dz,
                    })

        if len(scale_estimates) == 0:
            return None

        # Robust estimate
        median_scale = float(np.median(scale_estimates))

        # Identify inliers (consistent pairs = both joints visible)
        threshold = 0.3 * median_scale  # 30% tolerance
        inliers = [p for p in pair_info if abs(p['scale'] - median_scale) < threshold]
        outliers = [p for p in pair_info if abs(p['scale'] - median_scale) >= threshold]

        # Count how often each joint appears in inlier vs outlier pairs
        joint_inlier_count = {}
        joint_outlier_count = {}
        for p in inliers:
            for j in p['joints']:
                joint_inlier_count[j] = joint_inlier_count.get(j, 0) + 1
        for p in outliers:
            for j in p['joints']:
                joint_outlier_count[j] = joint_outlier_count.get(j, 0) + 1

        # Visible joints: appear mostly in inliers
        visible_joints = []
        occluded_joints = []
        for j in joint_depths.keys():
            inlier = joint_inlier_count.get(j, 0)
            outlier = joint_outlier_count.get(j, 0)
            if inlier >= outlier:
                visible_joints.append(j)
            else:
                occluded_joints.append(j)

        # Recompute scale from inliers only
        if len(inliers) >= 3:
            final_scale = float(np.median([p['scale'] for p in inliers]))
        else:
            final_scale = median_scale

        # Compute tz from a visible joint with most inlier appearances
        # D = z * scale + tz  =>  tz = D - z * scale
        tz = None
        if visible_joints:
            best_joint = max(visible_joints, key=lambda j: joint_inlier_count.get(j, 0))
            tz = joint_depths[best_joint] - joint_mesh_z[best_joint] * final_scale

        return {
            'scale': final_scale,
            'tz': tz,
            'num_joints_sampled': len(joint_depths),
            'num_pairs': len(scale_estimates),
            'num_inlier_pairs': len(inliers),
            'visible_joints': visible_joints,
            'occluded_joints': occluded_joints,
            'joint_depths': joint_depths,
        }

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

    def _apply_depth_scale_correction(self, outputs, depth_map, masks_np=None, img_shape=None, adjust_position=False):
        """
        Correct mesh scales using 2D reprojection constraint with depth ratios.

        For each person:
        1. Sample depth at each joint's 2D location (within mask)
        2. Use depth ratios between joint pairs to compute scale
        3. Identify visible vs occluded joints from consistency
        4. Apply scale and set Z-position

        The key insight: for two visible joints j1, j2:
            D1 - D2 = (z1 - z2) * scale
        where D is sampled depth, z is mesh-local Z.
        """
        if depth_map is None or len(outputs) == 0:
            return outputs

        depth_h, depth_w = depth_map.shape

        # Use actual image dimensions if provided
        if img_shape is not None:
            img_h, img_w = img_shape
        else:
            img_h, img_w = depth_h, depth_w

        print(f"[SAM3DBody] Image: {img_w}x{img_h}, Depth map: {depth_w}x{depth_h}")

        # 1. Compute scale using mask-based height (always, for debug)
        person_data = []
        for i, output in enumerate(outputs):
            person_mask = masks_np[i] if masks_np is not None and i < len(masks_np) else None

            # Get focal length from model output
            focal_length = output.get("focal_length", 5000.0)
            if isinstance(focal_length, np.ndarray):
                focal_length = float(focal_length.flatten()[0])

            # Get mesh height
            vertices = output.get("pred_vertices")
            mesh_height = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1])) if vertices is not None else None

            print(f"[SAM3DBody] Person {i}:")
            print(f"  mesh_height={mesh_height:.3f}m" if mesh_height else "  mesh_height=N/A")

            # Always compute mask-based for debug
            mask_data = self._compute_mask_depth_and_height(
                person_mask, depth_map, focal_length, img_h, img_w
            )

            if mask_data is not None and mesh_height and mesh_height > 0.1:
                scale = mask_data["actual_height_m"] / mesh_height
                person_data.append({
                    "valid": True,
                    "scale": scale,
                    "tz": mask_data["depth"],
                    "visible_joints": [],
                    "occluded_joints": [],
                })
            else:
                print(f"  No valid mask measurement")
                person_data.append({"valid": False})

        # 2. Normalize scales relative to median
        valid_scales = [p["scale"] for p in person_data if p.get("valid")]
        if len(valid_scales) == 0:
            print("[SAM3DBody] Warning: No valid measurements, skipping scale correction")
            return outputs

        median_scale = float(np.median(valid_scales))
        print(f"[SAM3DBody] Median scale: {median_scale:.3f}")

        # 3. Apply corrections
        for i, output in enumerate(outputs):
            data = person_data[i]

            if not data.get("valid"):
                continue

            # Normalize scale relative to median
            scale_factor = data["scale"] / median_scale

            # Scale vertices around mesh centroid
            vertices = output.get("pred_vertices")
            if vertices is not None:
                centroid = vertices.mean(axis=0)
                output["pred_vertices"] = (vertices - centroid) * scale_factor + centroid

            # Also scale joint coordinates
            joints = output.get("pred_joint_coords")
            if joints is not None:
                joint_centroid = joints.mean(axis=0)
                output["pred_joint_coords"] = (joints - joint_centroid) * scale_factor + joint_centroid

            # Adjust Z-position using computed tz
            if adjust_position and data.get("tz") is not None:
                cam_t = output.get("pred_cam_t")
                if cam_t is not None:
                    # tz was computed as: D - z_mesh * scale
                    # After scaling mesh, need to recompute
                    new_tz = data["tz"]
                    output["pred_cam_t"] = np.array([cam_t[0], cam_t[1], new_tz])

            # Store metadata
            final_height = None
            if vertices is not None:
                final_height = float(np.max(output["pred_vertices"][:, 1]) - np.min(output["pred_vertices"][:, 1]))

            output["depth_scale_factor"] = scale_factor
            output["measured_depth"] = data.get("tz", 0)
            output["mesh_height"] = final_height
            output["visible_joints"] = data.get("visible_joints", [])
            output["occluded_joints"] = data.get("occluded_joints", [])

            height_str = f"{final_height:.2f}m" if final_height else "N/A"
            tz_val = data.get('tz')
            tz_str = f"{tz_val:.2f}" if tz_val is not None else "N/A"
            print(f"[SAM3DBody] Person {i} final: scale={scale_factor:.3f}, height={height_str}, tz={tz_str}")

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
            img_h, img_w = img_bgr.shape[:2]
            prepared_outputs = self._apply_depth_scale_correction(
                prepared_outputs, depth_map_np, masks_np=valid_masks,
                img_shape=(img_h, img_w), adjust_position=adjust_position_from_depth
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
