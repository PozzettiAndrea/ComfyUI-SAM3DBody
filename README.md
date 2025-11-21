# ComfyUI-SAM3DBody

A ComfyUI custom node wrapper for **SAM 3D Body** - Meta's robust full-body 3D human mesh recovery model.

SAM 3D Body (3DB) is a promptable model for single-image full-body 3D human mesh recovery. It demonstrates state-of-the-art performance with strong generalization and consistent accuracy in diverse in-the-wild conditions.

![body](docs/body.png)

## Nodes

### Load SAM 3D Body Model

**Node:** `Load SAM 3D Body Model`
**Category:** `SAM3DBody`

Loads the SAM 3D Body model with global caching for efficient reuse.

**Inputs:**
- `model_source`: Load from "huggingface" or "local" checkpoint
  - **huggingface**: Downloads from HuggingFace (requires authentication)
  - **local**: Loads from local .ckpt file
- `model_path`:
  - For HuggingFace: Model ID (e.g., "facebook/sam-3d-body-dinov3")
  - For local: Path to .ckpt file
- `device`: Device to use ("auto", "cuda", or "cpu")
- `mhr_path` (optional): Path to MHR (Momentum Human Rig) asset

**Outputs:**
- `model`: Loaded model for use in processing nodes

**Available Models:**
- `facebook/sam-3d-body-dinov3` - DINOv3-H+ backbone (840M params, best quality)
- `facebook/sam-3d-body-vith` - ViT-H backbone (631M params, good quality)

### SAM 3D Body: Process Image

**Node:** `SAM 3D Body: Process Image`
**Category:** `SAM3DBody/processing`

Processes an input image to reconstruct 3D human mesh.

**Inputs:**
- `model`: Loaded model from Load node
- `image`: Input image (ComfyUI IMAGE format)
- `bbox_threshold`: Confidence threshold for detection (0.0-1.0, default: 0.8)
- `inference_type`:
  - **full**: Body + hand decoders (best quality, slower)
  - **body**: Body decoder only (faster)
  - **hand**: Hand decoder only
- `mask` (optional): Segmentation mask to guide reconstruction
- `use_detector` (optional): Enable human detection
- `use_segmentor` (optional): Enable segmentation
- `use_fov_estimator` (optional): Enable FOV estimation

**Outputs:**
- `mesh_data`: Dictionary containing vertices, faces, joints, camera parameters
- `debug_image`: Visualization of the reconstruction

### SAM 3D Body: Process Image (Advanced)

**Node:** `SAM 3D Body: Process Image (Advanced)`
**Category:** `SAM3DBody/advanced`

Advanced processing with full control over detection, segmentation, and FOV estimation.

**Additional Inputs:**
- `nms_threshold`: Non-maximum suppression threshold
- `detector_name`: Human detector model ("vitdet" or "none")
- `segmentor_name`: Segmentation model ("sam2" or "none")
- `fov_name`: FOV estimator ("moge2" or "none")
- `detector_path`: Path to detector model
- `segmentor_path`: Path to segmentor model
- `fov_path`: Path to FOV model

### SAM 3D Body: Visualize Mesh

**Node:** `SAM 3D Body: Visualize Mesh`
**Category:** `SAM3DBody/visualization`

Renders the 3D mesh reconstruction onto the image.

**Inputs:**
- `mesh_data`: Mesh data from Process node
- `image`: Original input image
- `render_mode`:
  - **overlay**: Mesh overlaid on image
  - **mesh_only**: Just the mesh
  - **side_by_side**: Original and mesh side-by-side

**Outputs:**
- `rendered_image`: Visualization result

### SAM 3D Body: Export Mesh

**Node:** `SAM 3D Body: Export Mesh`
**Category:** `SAM3DBody/io`

Exports the reconstructed mesh to file formats.

**Inputs:**
- `mesh_data`: Mesh data from Process node
- `filename`: Output filename (e.g., "output.obj" or "output.ply")
- `output_dir`: Output directory path (default: "output")

**Outputs:**
- `file_path`: Path to exported mesh file

**Supported Formats:**
- `.obj` - Wavefront OBJ format
- `.ply` - Stanford PLY format

### SAM 3D Body: Get Mesh Info

**Node:** `SAM 3D Body: Get Mesh Info`
**Category:** `SAM3DBody/utilities`

Displays information about the reconstructed mesh.

**Inputs:**
- `mesh_data`: Mesh data from Process node

**Outputs:**
- `info`: Text information about vertices, faces, and joints

## Credits

**SAM 3D Body** by Meta AI:
- Paper: [SAM 3D Body: Robust Full-Body Human Mesh Recovery](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/)
- Repository: https://github.com/facebookresearch/sam-3d-body
- Team: Xitong Yang, Devansh Kukreja, Don Pinkus, and many others at Meta AI

**ComfyUI Wrapper** created following Meta's open source guidelines.

## Citation

If you use SAM 3D Body in your research, please cite:

```bibtex
@article{yang2025sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint},
  year={2025}
}
```