import os
import tempfile
import json
import glob
import torch
import numpy as np
import cv2
import folder_paths

# =============================================================================
# Helper functions (inlined to avoid relative import issues in worker)
# =============================================================================

def comfy_image_to_numpy(image):
    """Convert ComfyUI image tensor [B,H,W,C] to numpy BGR [H,W,C] for OpenCV."""
    img_np = image[0].cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return img_np[..., ::-1].copy()  # RGB -> BGR


def comfy_mask_to_numpy(mask):
    """Convert ComfyUI mask tensor to numpy [N,H,W].

    Handles:
    - Standard masks: [N, H, W] -> [N, H, W]
    - Single mask: [H, W] -> [1, H, W]
    - RGB/RGBA masks: [N, H, W, C] or [H, W, C] -> grayscale [N, H, W]
    """
    arr = mask.cpu().numpy()

    # Handle RGB/RGBA: convert to grayscale by averaging channels
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # [H, W, C] -> [H, W]
        arr = arr.mean(axis=-1)
    elif arr.ndim == 4 and arr.shape[-1] in (3, 4):
        # [N, H, W, C] -> [N, H, W]
        arr = arr.mean(axis=-1)

    # Ensure batch dimension
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    return arr


def numpy_to_comfy_image(np_image):
    """Convert numpy BGR [H,W,C] to ComfyUI image tensor [1,H,W,C]."""
    img_rgb = np_image[..., ::-1].copy()  # BGR -> RGB
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).unsqueeze(0)


# Module-level cache for loaded model (persists across calls in worker)
_MODEL_CACHE = {}


def _load_sam3d_model(model_config: dict):
    """
    Load SAM 3D Body model from config paths.

    Uses module-level caching to avoid reloading on every call.
    This runs inside the isolated worker subprocess.
    """
    cache_key = model_config["ckpt_path"]

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Import heavy dependencies only inside worker
    from ..sam_3d_body import load_sam_3d_body

    ckpt_path = model_config["ckpt_path"]
    device = model_config["device"]
    mhr_path = model_config.get("mhr_path", "")

    # Load model using the library's built-in function
    print(f"[SAM3DBody] Loading model from {ckpt_path}...")
    sam_3d_model, model_cfg, _ = load_sam_3d_body(
        checkpoint_path=ckpt_path,
        device=device,
        mhr_path=mhr_path,
    )

    print(f"[SAM3DBody] Model loaded successfully on {device}")

    # Cache for reuse
    result = {
        "model": sam_3d_model,
        "model_cfg": model_cfg,
        "device": device,
        "mhr_path": mhr_path,
    }
    _MODEL_CACHE[cache_key] = result

    return result


def find_mhr_model_path(mesh_data=None):
    """Find the MHR model path using multiple fallback strategies."""
    # Strategy 1: Check mesh_data for explicitly provided path
    if mesh_data and mesh_data.get("mhr_path"):
        mhr_path = mesh_data["mhr_path"]
        if os.path.exists(mhr_path):
            return mhr_path

    # Strategy 2: Check environment variable
    env_path = os.environ.get("SAM3D_MHR_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path

    # Strategy 3: Search ComfyUI models/sam3dbody/ folder
    sam3dbody_dir = os.path.join(folder_paths.models_dir, "sam3dbody", "assets", "mhr_model.pt")
    if os.path.exists(sam3dbody_dir):
        return sam3dbody_dir

    # Strategy 4 (legacy): Search HuggingFace cache
    hf_cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3")
    if os.path.exists(hf_cache_base):
        pattern = os.path.join(hf_cache_base, "snapshots", "*", "assets", "mhr_model.pt")
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]

    return None


# Import BpyFBXExporter from export module (runs in isolated bpy environment)
from .export import BpyFBXExporter


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
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics from Depth Anything V3 - [3,3] matrix with fx, fy, cx, cy. Provides accurate focal length instead of default 5000"
                }),
                "depth_confidence": ("IMAGE", {
                    "tooltip": "Confidence map from Depth Anything V3 - used to weight depth samples and filter unreliable measurements"
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

    # MHR70 skeleton - matches SAM3DBody output joint ordering
    # Joint indices from mhr70.py:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_hip, 10: right_hip, 11: left_knee, 12: right_knee
    # 13: left_ankle, 14: right_ankle, 15-20: feet
    # 21-41: right hand (wrist at 41), 42-62: left hand (wrist at 62)
    # 63-68: olecranon/cubital/acromion, 69: neck
    MHR70_SKELETON_BODY = [
        # Legs
        (13, 11),  # left_ankle -> left_knee
        (11, 9),   # left_knee -> left_hip
        (14, 12),  # right_ankle -> right_knee
        (12, 10),  # right_knee -> right_hip
        # Hips
        (9, 10),   # left_hip -> right_hip
        # Torso
        (5, 9),    # left_shoulder -> left_hip
        (6, 10),   # right_shoulder -> right_hip
        (5, 6),    # left_shoulder -> right_shoulder
        # Arms
        (5, 7),    # left_shoulder -> left_elbow
        (6, 8),    # right_shoulder -> right_elbow
        (7, 62),   # left_elbow -> left_wrist
        (8, 41),   # right_elbow -> right_wrist
        # Head/Face
        (1, 2),    # left_eye -> right_eye
        (0, 1),    # nose -> left_eye
        (0, 2),    # nose -> right_eye
        (1, 3),    # left_eye -> left_ear
        (2, 4),    # right_eye -> right_ear
        (3, 5),    # left_ear -> left_shoulder
        (4, 6),    # right_ear -> right_shoulder
        # Feet
        (13, 15),  # left_ankle -> left_big_toe
        (13, 16),  # left_ankle -> left_small_toe
        (13, 17),  # left_ankle -> left_heel
        (14, 18),  # right_ankle -> right_big_toe
        (14, 19),  # right_ankle -> right_small_toe
        (14, 20),  # right_ankle -> right_heel
    ]

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

        mask_h, mask_w = mask.shape[:2]
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
                mask_h, mask_w = mask.shape[:2]
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

    def _identify_visible_joints(self, smpl_joints_3d, smpl_joints_2d, depth_map, depth_conf,
                                  mask, img_h, img_w, cam_t=None, tolerance=0.20):
        """
        Identify which joints are truly visible vs self-occluded using depth ratio consistency.

        Key insight: For all VISIBLE joints, the ratio (measured_depth / predicted_camera_z)
        should be approximately 1.0. Self-occluded joints will have LOWER measured_depth
        (because we see the occluding body part which is closer), so their ratio will be < 1.0.

        Args:
            smpl_joints_3d: [J, 3] SMPL canonical joint positions (root-relative)
            smpl_joints_2d: [J, 2] projected joint positions in image coordinates
            depth_map: [H, W] depth values from DA3 (in meters)
            depth_conf: [H, W] confidence values (optional, can be None)
            mask: [H, W] segmentation mask for this person
            img_h, img_w: original image dimensions
            cam_t: [3] camera translation vector (required to convert to camera space)
            tolerance: relative tolerance for inlier detection (default 20%)

        Returns:
            visible_joints: list of joint indices that are visible
            scale_factor: the consensus ratio (z_measured / z_camera), or None if insufficient data
            confidence: quality metric for the estimation (0-1)
        """
        if smpl_joints_3d is None or smpl_joints_2d is None or depth_map is None:
            return [], None, 0.0

        # cam_t is required to convert SMPL coordinates to camera space
        if cam_t is None:
            print("      WARNING: cam_t not provided, cannot compute camera-space Z")
            return [], None, 0.0

        depth_h, depth_w = depth_map.shape
        print(f"    [DEBUG] mask type: {type(mask)}, mask shape: {mask.shape if mask is not None else 'None'}")
        if mask is not None:
            mask_h, mask_w = mask.shape[:2]
        else:
            mask_h, mask_w = depth_h, depth_w

        # Debug: print array shapes
        print(f"    [DEBUG] _identify_visible_joints:")
        print(f"      smpl_joints_3d shape: {smpl_joints_3d.shape}")
        print(f"      smpl_joints_2d shape: {smpl_joints_2d.shape}")
        print(f"      depth_map shape: {depth_map.shape}")
        print(f"      mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"      img_h={img_h}, img_w={img_w}")

        ratios = []
        confidences = []
        joint_indices = []

        # Use minimum of 3D and 2D joint counts (they may differ)
        num_joints = min(len(smpl_joints_3d), len(smpl_joints_2d))
        print(f"      Processing {num_joints} joints (3D has {len(smpl_joints_3d)}, 2D has {len(smpl_joints_2d)})")

        for j in range(num_joints):
            # Get 2D projection
            px, py = smpl_joints_2d[j, 0], smpl_joints_2d[j, 1]

            # Skip if outside image bounds
            if px < 0 or py < 0 or px >= img_w or py >= img_h:
                continue

            # Check if joint is inside mask (if provided)
            if mask is not None:
                mx = int(px * mask_w / img_w)
                my = int(py * mask_h / img_h)
                mx = max(0, min(mask_w - 1, mx))
                my = max(0, min(mask_h - 1, my))
                # Handle both (H,W) and (H,W,C) masks
                mask_val = mask[my, mx]
                if hasattr(mask_val, '__len__'):
                    mask_val = mask_val.mean()  # Average RGB channels
                if mask_val < 0.5:
                    continue

            # Sample depth at joint location
            dx = int(px * depth_w / img_w)
            dy = int(py * depth_h / img_h)
            dx = max(0, min(depth_w - 1, dx))
            dy = max(0, min(depth_h - 1, dy))

            z_measured = depth_map[dy, dx]
            # Convert SMPL joint Z to camera space by adding camera translation
            # smpl_joints_3d is root-relative, cam_t[2] is the depth of the root from camera
            z_camera = smpl_joints_3d[j, 2] + cam_t[2]

            # Skip invalid depth values (z_camera must be positive - in front of camera)
            if z_measured <= 0 or z_camera <= 0:
                continue

            # Ratio should be ~1.0 for visible joints (measured matches predicted)
            ratio = z_measured / z_camera

            # Get confidence weight
            if depth_conf is not None:
                conf_h, conf_w = depth_conf.shape
                cx = int(px * conf_w / img_w)
                cy = int(py * conf_h / img_h)
                cx = max(0, min(conf_w - 1, cx))
                cy = max(0, min(conf_h - 1, cy))
                conf = depth_conf[cy, cx]
            else:
                conf = 1.0

            ratios.append(ratio)
            confidences.append(conf)
            joint_indices.append(j)

        print(f"      Sampled {len(ratios)} joints with valid depth")

        if len(ratios) < 3:
            print(f"      ERROR: Not enough valid joints ({len(ratios)} < 3)")
            return [], None, 0.0

        ratios = np.array(ratios)
        confidences = np.array(confidences)

        print(f"      Ratios: min={ratios.min():.4f}, max={ratios.max():.4f}, mean={ratios.mean():.4f}")

        # Use UNWEIGHTED median for robust outlier detection
        # Weighted median can be dominated by a single high-confidence joint that happens to be an outlier
        # (e.g., one joint with high depth confidence but wrong ratio due to self-occlusion)
        # Simple unweighted median is more robust: majority vote of all visible joints
        median_ratio = float(np.median(ratios))

        print(f"      Median ratio (scale factor): {median_ratio:.4f}")

        # Identify inliers: joints where ratio matches consensus
        # Self-occluded joints have LOWER ratio (z_measured < expected because we see occluder)
        visible_mask = []
        for i, r in enumerate(ratios):
            relative_error = (r - median_ratio) / median_ratio
            # Allow asymmetric tolerance: more suspicious of joints with ratio << median
            # (indicates self-occlusion where we see a closer surface)
            is_inlier = relative_error > -tolerance and relative_error < tolerance * 1.5
            visible_mask.append(is_inlier)
            if not is_inlier:
                print(f"        Joint {joint_indices[i]}: ratio={r:.4f}, rel_error={relative_error:.3f} -> OUTLIER")

        visible_joints = [j for j, v in zip(joint_indices, visible_mask) if v]

        # The median ratio IS our scale factor
        scale_factor = float(median_ratio)

        # Confidence based on inlier ratio and average depth confidence
        inlier_ratio = sum(visible_mask) / len(visible_mask) if visible_mask else 0
        avg_conf = float(np.mean([c for c, v in zip(confidences, visible_mask) if v])) if any(visible_mask) else 0
        confidence = inlier_ratio * avg_conf

        print(f"      Inliers: {sum(visible_mask)}/{len(visible_mask)}, confidence={confidence:.3f}")

        return visible_joints, scale_factor, confidence

    def _compute_scale_with_intrinsics(self, output, depth_map, depth_conf, mask, intrinsics,
                                        img_h, img_w):
        """
        Compute scale factor using camera intrinsics and self-occlusion-aware joint visibility.

        This method:
        1. Projects SMPL joints to 2D using intrinsics
        2. Identifies visible vs self-occluded joints via depth ratio consistency
        3. Returns scale factor from the consensus of visible joints

        Args:
            output: SAM3DBody output dict containing pred_joint_coords, pred_keypoints_2d
            depth_map: [H, W] depth values from DA3
            depth_conf: [H, W] confidence values (optional)
            mask: [H, W] segmentation mask
            intrinsics: [3, 3] camera intrinsics matrix from DA3
            img_h, img_w: image dimensions

        Returns:
            dict with scale, visible_joints, occluded_joints, confidence, etc.
        """
        joints_3d = output.get("pred_joint_coords")
        keypoints_2d = output.get("pred_keypoints_2d")
        cam_t = output.get("pred_cam_t")

        if joints_3d is None or keypoints_2d is None:
            return None

        if cam_t is None:
            print("      WARNING: pred_cam_t not available, cannot use intrinsics method")
            return None

        # Extract focal length from intrinsics
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        focal_length = (fx + fy) / 2  # Average focal length

        # Use the 2D keypoints from SAM3DBody (already projected)
        # Note: These are in image coordinates
        # Pass cam_t to convert SMPL coords to camera space
        visible_joints, scale_factor, confidence = self._identify_visible_joints(
            joints_3d, keypoints_2d, depth_map, depth_conf,
            mask, img_h, img_w, cam_t=cam_t, tolerance=0.20
        )

        if scale_factor is None:
            return None

        # Compute median depth for position adjustment
        depth_h, depth_w = depth_map.shape
        if mask is not None:
            mask_h, mask_w = mask.shape[:2]
            if (mask_h, mask_w) != (depth_h, depth_w):
                mask_resized = cv2.resize(mask.astype(np.float32), (depth_w, depth_h),
                                          interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            mask_bool = mask_resized > 0.5
            if np.any(mask_bool):
                mask_depths = depth_map[mask_bool]
                valid_depths = mask_depths[mask_depths > 0]
                if len(valid_depths) > 0:
                    median_depth = float(np.median(valid_depths))
                else:
                    median_depth = None
            else:
                median_depth = None
        else:
            median_depth = None

        # Identify occluded joints (those not in visible list but were sampled)
        # Use only joints that could have been sampled (min of 3D and 2D counts)
        num_joints = min(len(joints_3d), len(keypoints_2d))
        all_joints = set(range(num_joints))
        occluded_joints = [j for j in all_joints if j not in visible_joints]

        print(f"    [DEBUG] _compute_scale_with_intrinsics result:")
        print(f"      scale_factor={scale_factor:.4f}, confidence={confidence:.3f}")
        print(f"      visible_joints={len(visible_joints)}, occluded_joints={len(occluded_joints)}")
        print(f"      median_depth={median_depth}")

        return {
            'scale': scale_factor,
            'tz': median_depth,
            'focal_length': focal_length,
            'visible_joints': visible_joints,
            'occluded_joints': occluded_joints,
            'confidence': confidence,
            'method': 'intrinsics_visibility'
        }

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

            # Log SMPL-X face/expression data availability for this person
            self._log_smplx_data_info(i, prepared_output)

        return prepared

    def _get_first_available(self, output, *keys):
        """Get the first non-None value from output for given keys."""
        for key in keys:
            val = output.get(key)
            if val is not None:
                return val
        return None

    def _log_smplx_data_info(self, person_idx, output):
        """Log information about available SMPL-X parameters for a person."""
        info_parts = []

        # Check for expression parameters (face blendshapes)
        expr_params = self._get_first_available(output, "expr_params", "pred_expr", "expression")
        if expr_params is not None:
            if isinstance(expr_params, np.ndarray):
                info_parts.append(f"expression[{expr_params.shape[-1]} params]")
            else:
                info_parts.append("expression[available]")

        # Check for jaw pose
        jaw_pose = self._get_first_available(output, "jaw_pose", "pred_jaw_pose")
        if jaw_pose is not None:
            info_parts.append("jaw_pose")

        # Check for hand poses
        left_hand = self._get_first_available(output, "left_hand_pose", "pred_lhand_pose")
        right_hand = self._get_first_available(output, "right_hand_pose", "pred_rhand_pose")
        if left_hand is not None:
            info_parts.append("left_hand")
        if right_hand is not None:
            info_parts.append("right_hand")

        # Check for global rotations (useful for FBX export)
        global_rots = self._get_first_available(output, "pred_global_rots", "global_orient")
        if global_rots is not None:
            if isinstance(global_rots, np.ndarray):
                info_parts.append(f"global_rots[{global_rots.shape}]")
            else:
                info_parts.append("global_rots")

        # Check for body pose parameters
        body_pose = self._get_first_available(output, "body_pose_params", "pred_body_pose")
        if body_pose is not None:
            if isinstance(body_pose, np.ndarray):
                info_parts.append(f"body_pose[{body_pose.shape[-1]} params]")

        # Check for shape parameters (betas)
        shape_params = self._get_first_available(output, "shape_params", "pred_betas", "betas")
        if shape_params is not None:
            if isinstance(shape_params, np.ndarray):
                info_parts.append(f"shape[{shape_params.shape[-1]} betas]")

        # Log summary
        if info_parts:
            print(f"[SAM3DBody] Person {person_idx} SMPL-X data: {', '.join(info_parts)}")
        else:
            # Log all available keys for debugging (first person only to avoid spam)
            if person_idx == 0:
                all_keys = [k for k in output.keys() if not k.startswith('_')]
                print(f"[SAM3DBody] Person {person_idx} available keys: {sorted(all_keys)}")

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

    def _apply_depth_scale_correction(self, outputs, depth_map, masks_np=None, img_shape=None,
                                        adjust_position=False, intrinsics=None, depth_conf=None):
        """
        Correct mesh scales using depth information and camera intrinsics.

        When intrinsics are provided (from Depth Anything V3):
        - Uses self-occlusion-aware visibility detection
        - Identifies visible vs occluded joints via depth ratio consistency
        - Uses accurate focal length from DA3 instead of default 5000

        Fallback (no intrinsics):
        - Uses mask-based height estimation with default focal length

        For each person:
        1. Compute scale factor from visible joints (intrinsics) or mask height (fallback)
        2. Normalize scales relative to median across all people
        3. Apply scale transformation to vertices and joints
        4. Optionally adjust Z-position based on depth

        Args:
            outputs: List of SAM3DBody output dicts
            depth_map: [H, W] depth values from DA3
            masks_np: [N, H, W] segmentation masks
            img_shape: (H, W) tuple of original image dimensions
            adjust_position: Whether to adjust Z-position based on depth
            intrinsics: [3, 3] camera intrinsics matrix from DA3 (optional)
            depth_conf: [H, W] confidence values from DA3 (optional)
        """
        if depth_map is None or len(outputs) == 0:
            return outputs

        depth_h, depth_w = depth_map.shape

        # Use actual image dimensions if provided
        if img_shape is not None:
            img_h, img_w = img_shape
        else:
            img_h, img_w = depth_h, depth_w

        # Check if we have intrinsics for the improved method
        use_intrinsics = intrinsics is not None

        if use_intrinsics:
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
            focal_from_intrinsics = (fx + fy) / 2
            print(f"[SAM3DBody] Using DA3 intrinsics: fx={fx:.1f}, fy={fy:.1f}")
            print(f"[SAM3DBody] Image: {img_w}x{img_h}, Depth map: {depth_w}x{depth_h}")
        else:
            print(f"[SAM3DBody] No intrinsics provided, using fallback method with default focal length")
            print(f"[SAM3DBody] Image: {img_w}x{img_h}, Depth map: {depth_w}x{depth_h}")

        # 1. Compute scale for each person
        person_data = []
        for i, output in enumerate(outputs):
            person_mask = masks_np[i] if masks_np is not None and i < len(masks_np) else None

            # Get mesh height for reference
            vertices = output.get("pred_vertices")
            mesh_height = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1])) if vertices is not None else None

            print(f"[SAM3DBody] Person {i}:")
            print(f"  mesh_height={mesh_height:.3f}m" if mesh_height else "  mesh_height=N/A")

            if use_intrinsics:
                # NEW: Use intrinsics-based visibility detection
                scale_data = self._compute_scale_with_intrinsics(
                    output, depth_map, depth_conf, person_mask, intrinsics, img_h, img_w
                )

                if scale_data is not None:
                    num_visible = len(scale_data['visible_joints'])
                    num_occluded = len(scale_data['occluded_joints'])
                    print(f"  [intrinsics method] scale={scale_data['scale']:.3f}, "
                          f"visible_joints={num_visible}, occluded_joints={num_occluded}, "
                          f"confidence={scale_data['confidence']:.2f}")
                    person_data.append({
                        "valid": True,
                        "scale": scale_data['scale'],
                        "tz": scale_data['tz'],
                        "visible_joints": scale_data['visible_joints'],
                        "occluded_joints": scale_data['occluded_joints'],
                        "confidence": scale_data['confidence'],
                        "method": "intrinsics_visibility"
                    })
                else:
                    print(f"  [intrinsics method] Failed, falling back to mask-based")
                    # Fallback to mask-based for this person
                    focal_length = focal_from_intrinsics
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
                            "confidence": 0.5,
                            "method": "mask_fallback"
                        })
                    else:
                        print(f"  No valid measurement")
                        person_data.append({"valid": False})
            else:
                # FALLBACK: Use mask-based height estimation (original method)
                focal_length = output.get("focal_length", 5000.0)
                if isinstance(focal_length, np.ndarray):
                    focal_length = float(focal_length.flatten()[0])

                mask_data = self._compute_mask_depth_and_height(
                    person_mask, depth_map, focal_length, img_h, img_w
                )

                if mask_data is not None and mesh_height and mesh_height > 0.1:
                    scale = mask_data["actual_height_m"] / mesh_height
                    print(f"  [mask method] scale={scale:.3f}, depth={mask_data['depth']:.2f}m")
                    person_data.append({
                        "valid": True,
                        "scale": scale,
                        "tz": mask_data["depth"],
                        "visible_joints": [],
                        "occluded_joints": [],
                        "confidence": 0.5,
                        "method": "mask_height"
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
            output["scale_method"] = data.get("method", "unknown")
            output["scale_confidence"] = data.get("confidence", 0)

            height_str = f"{final_height:.2f}m" if final_height else "N/A"
            tz_val = data.get('tz')
            tz_str = f"{tz_val:.2f}" if tz_val is not None else "N/A"
            method = data.get('method', 'unknown')
            conf = data.get('confidence', 0)
            print(f"[SAM3DBody] Person {i} final: scale={scale_factor:.3f}, height={height_str}, "
                  f"tz={tz_str}, method={method}, confidence={conf:.2f}")

        return outputs

    def process_multiple(self, model, image, masks, inference_type="full", depth_map=None,
                          intrinsics=None, depth_confidence=None, adjust_position_from_depth=False):
        """Process image with multiple masks and reconstruct 3D meshes for all people.

        Args:
            model: Config dict from LoadSAM3DBodyModel with paths (model loaded lazily here)
            image: Input image tensor
            masks: Batched masks - one per person (N, H, W)
            inference_type: "full" or "body"
            depth_map: Optional depth map for scale correction (from DA3)
            intrinsics: Optional camera intrinsics [3,3] tensor from DA3
            depth_confidence: Optional confidence map [B,H,W,C] from DA3
            adjust_position_from_depth: Adjust Z-position based on depth map
        """
        from ..sam_3d_body import SAM3DBodyEstimator

        # DEBUG: Print all input shapes at function entry
        print(f"[DEBUG] FUNCTION ENTRY - image type={type(image)}, shape={image.shape if hasattr(image, 'shape') else 'N/A'}")
        print(f"[DEBUG] FUNCTION ENTRY - masks type={type(masks)}, shape={masks.shape if hasattr(masks, 'shape') else 'N/A'}")

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

        # Process intrinsics if provided
        intrinsics_np = None
        if intrinsics is not None:
            if isinstance(intrinsics, torch.Tensor):
                # Intrinsics is [B, 3, 3] or [3, 3] - we need [3, 3]
                intrinsics_np = intrinsics.squeeze().cpu().numpy()
                if intrinsics_np.ndim == 3:
                    intrinsics_np = intrinsics_np[0]  # Take first batch element
            else:
                intrinsics_np = np.array(intrinsics)
                if intrinsics_np.ndim == 3:
                    intrinsics_np = intrinsics_np[0]
            print(f"[SAM3DBody] Intrinsics provided: fx={intrinsics_np[0,0]:.1f}, fy={intrinsics_np[1,1]:.1f}, "
                  f"cx={intrinsics_np[0,2]:.1f}, cy={intrinsics_np[1,2]:.1f}")

        # Process depth confidence if provided
        depth_conf_np = None
        if depth_confidence is not None:
            if isinstance(depth_confidence, torch.Tensor):
                # Confidence is [B, H, W, C] IMAGE format - take first channel
                depth_conf_np = depth_confidence[0, :, :, 0].cpu().numpy()
            else:
                depth_conf_np = depth_confidence[0, :, :, 0] if depth_confidence.ndim == 4 else depth_confidence[:, :, 0]
            print(f"[SAM3DBody] Depth confidence provided: shape={depth_conf_np.shape}, "
                  f"range=[{depth_conf_np.min():.2f}, {depth_conf_np.max():.2f}]")

        # Lazy load model (cached after first call)
        loaded = _load_sam3d_model(model)
        sam_3d_model = loaded["model"]
        model_cfg = loaded["model_cfg"]

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

        # DEBUG: Log raw mask tensor BEFORE conversion
        print(f"[DEBUG] RAW masks type: {type(masks)}")
        print(f"[DEBUG] RAW masks shape: {masks.shape if hasattr(masks, 'shape') else 'N/A'}")

        # Convert masks to numpy - always returns [N, H, W]
        masks_np = comfy_mask_to_numpy(masks)
        num_people = masks_np.shape[0]

        # DEBUG: Log mask info AFTER conversion
        print(f"[DEBUG] masks_np shape: {masks_np.shape}")
        print(f"[DEBUG] masks_np dtype: {masks_np.dtype}")
        for i in range(masks_np.shape[0]):
            mask_sum = masks_np[i].sum()
            mask_nonzero = (masks_np[i] > 0.5).sum()
            print(f"[DEBUG] Mask {i}: sum={mask_sum:.2f}, nonzero_pixels={mask_nonzero}")

        # Compute bounding boxes from each mask
        bboxes_list = []
        valid_mask_indices = []
        for i in range(num_people):
            bbox = self._compute_bbox_from_mask(masks_np[i])
            if bbox is not None:
                bboxes_list.append(bbox)
                valid_mask_indices.append(i)

        # DEBUG: Log valid bboxes
        print(f"[DEBUG] Valid bboxes: {len(bboxes_list)} out of {num_people} masks")
        print(f"[DEBUG] Valid mask indices: {valid_mask_indices}")

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
            # Convert intrinsics to tensor for SAM3DBody if available
            cam_int_tensor = None
            if intrinsics_np is not None:
                cam_int_tensor = torch.from_numpy(intrinsics_np).float().unsqueeze(0)

            # Process all people at once
            outputs = estimator.process_one_image(
                tmp_path,
                bboxes=bboxes,
                masks=masks_for_estimator,
                cam_int=cam_int_tensor,
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
                img_shape=(img_h, img_w), adjust_position=adjust_position_from_depth,
                intrinsics=intrinsics_np, depth_conf=depth_conf_np
            )

        # Create combined mesh data (internal structure for export)
        faces_np = estimator.faces.cpu().numpy() if isinstance(estimator.faces, torch.Tensor) else estimator.faces
        multi_mesh_data = {
            "num_people": len(prepared_outputs),
            "people": prepared_outputs,
            "faces": faces_np,
            "mhr_path": loaded.get("mhr_path", None),
        }

        # Create preview visualization
        preview = self._create_multi_person_preview(
            img_bgr, prepared_outputs, estimator.faces, valid_masks
        )
        preview_comfy = numpy_to_comfy_image(preview)

        return (multi_mesh_data, preview_comfy)

    def _draw_skeleton_overlay(self, image, keypoints_2d, color, thickness=2, joint_radius=4):
        """
        Draw skeleton overlay on image using 2D keypoints.

        Args:
            image: BGR image to draw on (modified in-place)
            keypoints_2d: [J, 2] array of 2D joint positions
            color: BGR color tuple for this person
            thickness: Line thickness for bones
            joint_radius: Radius for joint circles
        """
        h, w = image.shape[:2]
        num_joints = len(keypoints_2d)

        # Use MHR70 skeleton for SAM3DBody output (70 keypoints)
        skeleton = self.MHR70_SKELETON_BODY

        # Draw bones (lines between connected joints)
        for parent_idx, child_idx in skeleton:
            if parent_idx >= num_joints or child_idx >= num_joints:
                continue

            pt1 = keypoints_2d[parent_idx]
            pt2 = keypoints_2d[child_idx]

            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])

            # Skip if either point is outside image
            if not (0 <= x1 < w and 0 <= y1 < h):
                continue
            if not (0 <= x2 < w and 0 <= y2 < h):
                continue

            # Draw bone
            cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Draw joints (circles at each keypoint)
        for j, pt in enumerate(keypoints_2d):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                # Filled circle with outline
                cv2.circle(image, (x, y), joint_radius, color, -1, cv2.LINE_AA)
                cv2.circle(image, (x, y), joint_radius, (255, 255, 255), 1, cv2.LINE_AA)

    def _create_multi_person_preview(self, img_bgr, outputs, faces, masks_np=None):
        """Create a preview visualization showing all detected people."""
        try:
            from ..sam_3d_body.visualization.renderer import Renderer

            h, w = img_bgr.shape[:2]

            # Overlay masks with semi-transparent colors
            if masks_np is not None:
                result = img_bgr.copy()
                colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (255, 0, 255), (0, 255, 255),
                    (128, 255, 0), (255, 128, 0), (128, 0, 255),
                    (0, 128, 255), (255, 0, 128),
                ]
                for i, mask in enumerate(masks_np):
                    color = colors[i % len(colors)]
                    # Resize mask to image size if needed
                    if mask.shape != (h, w):
                        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_resized = mask
                    # Apply colored overlay where mask > 0.5
                    mask_bool = mask_resized > 0.5
                    overlay = result.copy()
                    overlay[mask_bool] = color
                    # Blend with alpha
                    alpha = 0.3
                    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
                img_bgr = result  # Use the mask-overlaid image as base

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

                    # Draw skeleton overlay on top of rendered mesh
                    skeleton_colors = [
                        (255, 100, 100), (100, 255, 100), (100, 100, 255),
                        (255, 255, 100), (255, 100, 255), (100, 255, 255),
                    ]
                    for i, output in enumerate(outputs):
                        kpts_2d = output.get("pred_keypoints_2d")
                        if kpts_2d is not None:
                            color = skeleton_colors[i % len(skeleton_colors)]
                            self._draw_skeleton_overlay(result, kpts_2d, color, thickness=2, joint_radius=3)

                    return result

            return img_bgr

        except Exception:
            # Fallback: draw skeleton with bones (not just points)
            result = img_bgr.copy()

            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
            ]

            for i, output in enumerate(outputs):
                kpts_2d = output.get("pred_keypoints_2d")
                if kpts_2d is not None:
                    color = colors[i % len(colors)]
                    self._draw_skeleton_overlay(result, kpts_2d, color, thickness=2, joint_radius=3)

            return result

    def _export_to_fbx(self, multi_mesh_data, output_filename, combine):
        """Export multi-person mesh data to FBX file(s)."""
        num_people = multi_mesh_data.get("num_people", 0)
        people = multi_mesh_data.get("people", [])
        faces = multi_mesh_data.get("faces")

        if num_people == 0 or len(people) == 0:
            raise RuntimeError("No mesh data to export")

        # Setup output path
        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'
        output_fbx_path = os.path.join(output_dir, output_filename)

        # Find MHR model path for skinning weights
        mhr_model_path = find_mhr_model_path(multi_mesh_data)

        # Load skinning data once (same for all people)
        vertex_weights = {}
        joint_parents = None
        if mhr_model_path and os.path.exists(mhr_model_path):
            try:
                mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                lbs = mhr_model.character_torch.linear_blend_skinning

                vert_indices = lbs.vert_indices_flattened.cpu().numpy().astype(int)
                skin_indices = lbs.skin_indices_flattened.cpu().numpy().astype(int)
                skin_weights = lbs.skin_weights_flattened.cpu().numpy().astype(float)

                for j in range(len(vert_indices)):
                    vert_idx = int(vert_indices[j])
                    bone_idx = int(skin_indices[j])
                    weight = float(skin_weights[j])
                    if vert_idx not in vertex_weights:
                        vertex_weights[vert_idx] = []
                    vertex_weights[vert_idx].append([bone_idx, weight])

                # Get joint parents
                joint_parents = mhr_model.character_torch.skeleton.joint_parents.cpu().numpy().astype(int).tolist()
            except Exception:
                pass

        # Build combined data structure for all people
        temp_files = []

        try:
            people_data_for_export = []

            for i, person in enumerate(people):
                vertices = person.get("pred_vertices")
                joint_coords = person.get("pred_joint_coords")
                cam_t = person.get("pred_cam_t")
                global_rots = person.get("pred_global_rots")

                if vertices is None:
                    continue

                # Convert to numpy
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                if joint_coords is not None and isinstance(joint_coords, torch.Tensor):
                    joint_coords = joint_coords.cpu().numpy()
                if cam_t is not None and isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                if global_rots is not None and isinstance(global_rots, torch.Tensor):
                    global_rots = global_rots.cpu().numpy()

                # Apply world position offset from camera translation
                if cam_t is not None:
                    vertices = vertices + cam_t
                    if joint_coords is not None:
                        joint_coords = joint_coords + cam_t

                # Write OBJ file for this person
                temp_obj = tempfile.NamedTemporaryFile(suffix=f'_person{i}.obj', delete=False)
                temp_files.append(temp_obj.name)
                self._write_obj_file(temp_obj.name, vertices, faces)

                # Prepare skeleton data
                skeleton_info = {}
                if joint_coords is not None:
                    # Apply coordinate transform to joint positions (flip Y and Z)
                    joint_coords_flipped = joint_coords.copy()
                    joint_coords_flipped[:, 1] = -joint_coords_flipped[:, 1]
                    joint_coords_flipped[:, 2] = -joint_coords_flipped[:, 2]

                    skeleton_info = {
                        "joint_positions": joint_coords_flipped.tolist(),
                        "num_joints": len(joint_coords),
                    }

                    # Add skinning weights
                    if vertex_weights:
                        num_vertices = len(vertices)
                        skinning_list = []
                        for vert_idx in range(num_vertices):
                            if vert_idx in vertex_weights:
                                skinning_list.append(vertex_weights[vert_idx])
                            else:
                                skinning_list.append([])
                        skeleton_info["skinning_weights"] = skinning_list

                    # Add joint parents
                    if joint_parents:
                        skeleton_info["joint_parents"] = joint_parents

                    # Add global joint rotations
                    if global_rots is not None:
                        skeleton_info["global_rotations"] = global_rots.tolist()

                # Write skeleton JSON for this person
                person_skeleton_json = None
                if skeleton_info:
                    person_skeleton_json = tempfile.NamedTemporaryFile(
                        suffix=f'_person{i}_skeleton.json', delete=False, mode='w'
                    )
                    temp_files.append(person_skeleton_json.name)
                    json.dump(skeleton_info, person_skeleton_json)
                    person_skeleton_json.close()
                    person_skeleton_json = person_skeleton_json.name

                people_data_for_export.append({
                    "obj_path": temp_obj.name,
                    "skeleton_json_path": person_skeleton_json,
                    "index": i,
                })

            if not people_data_for_export:
                raise RuntimeError("No valid mesh data to export")

            exporter = BpyFBXExporter()

            if combine:
                # Combined mode: export all people into a single FBX file
                combined_json = tempfile.NamedTemporaryFile(
                    suffix='_combined_export.json', delete=False, mode='w'
                )
                temp_files.append(combined_json.name)
                json.dump(people_data_for_export, combined_json)
                combined_json.close()

                result = exporter.export(
                    input_obj_path=None,
                    output_fbx_path=output_fbx_path,
                    combined_json_path=combined_json.name
                )

                if not result.get("success"):
                    raise RuntimeError("Combined FBX export failed")

                print(f"[SAM3DBody] Combined FBX created: {output_fbx_path}")
                return output_fbx_path

            else:
                # Separate mode: export each person to individual FBX files
                exported_files = []

                for person_data in people_data_for_export:
                    obj_path = person_data["obj_path"]
                    idx = person_data["index"]
                    person_skeleton_json = person_data.get("skeleton_json_path")

                    if len(people_data_for_export) == 1:
                        person_fbx_path = output_fbx_path
                    else:
                        person_fbx_path = output_fbx_path.replace('.fbx', f'_person{idx}.fbx')

                    result = exporter.export(
                        input_obj_path=obj_path,
                        output_fbx_path=person_fbx_path,
                        skeleton_json_path=person_skeleton_json
                    )

                    if result.get("success"):
                        exported_files.append(person_fbx_path)
                    else:
                        raise RuntimeError(f"FBX export failed for person {idx}")

                if not exported_files:
                    raise RuntimeError("No FBX files were exported")

                print(f"[SAM3DBody] Separate FBX files created: {len(exported_files)} files")
                return exported_files[0]

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass

    def _write_obj_file(self, filepath, vertices, faces):
        """Write mesh to OBJ file format."""
        with open(filepath, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {-v[1]:.6f} {-v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# Register node
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyProcessMultiple": SAM3DBodyProcessMultiple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyProcessMultiple": "SAM 3D Body Process Multiple",
}
