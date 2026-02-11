# Depth Correction Improvement Plan

## Problem Statement

The current depth correction implementation in `SAM3DBodyProcessMultiple` has several limitations that reduce accuracy and robustness when scaling 3D body meshes based on depth information.

---

## Current Implementation Issues

### 1. Hardcoded Focal Length

The current code uses a default focal length of `5000.0` when camera intrinsics aren't available. This is problematic because:

- A typical smartphone camera: f ≈ 800-1200 px
- A webcam: f ≈ 500-900 px
- A DSLR with kit lens: f ≈ 2000-3000 px
- 5000 px implies a telephoto lens

If the true focal length is 1000 and we assume 5000, height estimates are **off by 5x**.

**Solution:** Depth Anything V3 already outputs camera intrinsics! We need to wire this input to `SAM3DBodyProcessMultiple` and use it.

### 2. Mask-Based Median Depth Sampling

The current approach samples depth values within the segmentation mask and takes the median. Issues:

- Assumes the person is a flat billboard (humans have depth variation)
- Nose to ear: ~15-20 cm depth difference
- Side-view vs front-view gives completely different median values for same person at same distance

### 3. Bounding Box Height Measurement

Using pixel height from bounding box fails for:

- Seated people (pixel height ≈ 50% of standing)
- Bent/crouching poses
- Arms raised above head (inflates bounding box)
- Lying down (pixel height is body width)

### 4. No Self-Occlusion Handling

**This is critical.** The current approach assumes that if a point is inside the segmentation mask, it's visible. This is FALSE due to self-occlusion.

---

## Understanding Self-Occlusion

### What is Self-Occlusion?

When a person's body parts occlude other parts of their own body from the camera's viewpoint.

**Example:** Person turned sideways
```
Camera view:

    [Left Shoulder] ← BEHIND (occluded)
          ↓
    [Right Shoulder] ← VISIBLE (in front)
          ↓
      [Camera]
```

If we project the LEFT shoulder's 3D position to 2D and sample the depth map at that pixel, we get the RIGHT shoulder's depth (which is closer). The left shoulder is behind it, but its 2D projection still lands inside the segmentation mask.

### Why This Breaks Current Approach

```
Depth map at left_shoulder_2d:  1.5m  (actually right shoulder)
SMPL says left shoulder is at:   2.0m  (correct, but occluded)

If we use 1.5m for calculations → WRONG scale factor
```

### Common Self-Occlusion Scenarios

1. **Sideways pose:** One arm/shoulder behind the other
2. **Arms crossed:** Forearms occlude torso
3. **Hands in front of face:** Hands occlude head
4. **Twisted torso:** Parts of back visible, front occluded (or vice versa)
5. **Legs crossed:** One leg behind the other

---

## Proposed Solution

### Core Insight

For all **truly visible** joints, the ratio `measured_depth / smpl_predicted_depth` should be approximately constant (this constant IS the scale factor).

Self-occluded joints will have a **smaller measured depth** (because we're seeing the occluding body part which is closer), so their ratio will be **lower than the consensus**.

### Algorithm

```
1. Project ALL SMPL joints to 2D using camera intrinsics

2. For each joint inside the segmentation mask:
   - Sample depth from depth map: z_measured
   - Get SMPL's predicted depth: z_smpl
   - Compute ratio: r = z_measured / z_smpl
   - Store (joint_index, ratio, confidence)

3. Find consensus ratio using weighted median
   - Majority of joints should be visible
   - Median is robust to outlier occluded joints

4. Identify visible vs occluded joints:
   - ratio ≈ median → VISIBLE (inlier)
   - ratio << median → SELF-OCCLUDED (outlier)

5. The consensus ratio IS the scale factor!
   scale = median(z_measured / z_smpl)

6. Apply scale to mesh vertices and joints
```

### Why This Works

| Joint State | z_measured | z_smpl | Ratio | vs Median |
|-------------|------------|--------|-------|-----------|
| Visible (front) | 2.0m | 2.0 (canonical) | 1.0 | ≈ median |
| Visible (back) | 2.5m | 2.5 (canonical) | 1.0 | ≈ median |
| Self-occluded | 1.5m (sees occluder) | 2.0 (canonical) | 0.75 | << median |

The occluded joint has a lower ratio because we measured a closer surface.

---

## Implementation Plan

### Phase 1: Add New Inputs to SAM3DBodyProcessMultiple

**File:** `nodes/processing/process_multiple.py`

Add optional inputs:
- `intrinsics` (INTRINSICS type from DA3) - camera intrinsic matrix K
- `depth_confidence` (IMAGE type from DA3) - confidence map for depth values

### Phase 2: Implement Visibility Detection

Create new method `_identify_visible_joints()`:

```python
def _identify_visible_joints(
    self,
    smpl_joints_3d,      # [J, 3] SMPL canonical joint positions
    smpl_joints_2d,      # [J, 2] projected to image
    depth_map,           # [H, W] from DA3
    depth_conf,          # [H, W] confidence (optional)
    mask,                # [H, W] segmentation
    tolerance=0.20       # 20% tolerance for inlier detection
):
    """
    Identify which joints are truly visible vs self-occluded.

    Returns:
        visible_joints: list of joint indices that are visible
        scale_factor: the consensus ratio (z_measured / z_smpl)
        confidence: quality metric for the estimation
    """
```

### Phase 3: Implement Robust Scale Estimation

Create new method `_compute_scale_from_visible_joints()`:

```python
def _compute_scale_from_visible_joints(
    self,
    visible_joints,
    smpl_joints_3d,
    depth_map,
    smpl_joints_2d,
    intrinsics,
    depth_conf=None
):
    """
    Compute scale factor using only verified visible joints.

    Uses the ratio-based approach where scale = z_measured / z_smpl
    for visible joints.
    """
```

### Phase 4: Update Main Processing Flow

Modify `_apply_depth_scale_correction()` to:

1. Check if intrinsics are provided, use them instead of default 5000
2. Project joints using proper intrinsics
3. Call visibility detection
4. Use only visible joints for scale computation
5. Report confidence and which joints were used

### Phase 5: Update Node Registration

Modify the node's `INPUT_TYPES` to include new optional inputs:
- `intrinsics` from DepthAnything_V3
- `depth_confidence` from DepthAnything_V3

---

## Code Changes Summary

| File | Changes |
|------|---------|
| `nodes/processing/process_multiple.py` | Add inputs, implement visibility detection, update scale computation |
| `workflows/multiple_masks_depthcorrected.json` | Connect intrinsics output from DA3 to processing node |

---

## Expected Benefits

1. **Correct focal length** - Uses DA3's estimated intrinsics instead of hardcoded 5000
2. **Self-occlusion robust** - Automatically detects and excludes occluded joints
3. **More accurate scale** - Uses joint-based measurement instead of bounding box
4. **Confidence output** - Downstream knows when to trust the result
5. **Graceful degradation** - Falls back if too few visible joints

---

## Validation Plan

1. Test with front-facing poses (most joints visible) - should work as before
2. Test with sideways poses (self-occlusion) - should now be more accurate
3. Test with multiple people at different depths - relative scaling should be correct
4. Compare scale estimates with and without intrinsics input
5. Verify confidence correlates with actual accuracy
