# ComfyUI-SAM3DBody

A ComfyUI custom node wrapper for **SAM 3D Body** - Meta's robust full-body 3D human mesh recovery model.

SAM 3D Body (3DB) is a promptable model for single-image full-body 3D human mesh recovery. It demonstrates state-of-the-art performance with strong generalization and consistent accuracy in diverse in-the-wild conditions.

## Features

- **Single-Image 3D Reconstruction**: Recover full-body 3D human mesh from a single image
- **Model Loading with Caching**: Efficient model loading from HuggingFace or local checkpoints
- **Flexible Processing Options**: Support for body-only, hand-only, or full-body reconstruction
- **Mesh Visualization**: Render and visualize 3D mesh reconstructions
- **Mesh Export**: Export meshes to OBJ and PLY formats
- **Optional Enhancements**: Support for human detection, segmentation, and FOV estimation
- **ComfyUI Integration**: Seamless integration with ComfyUI workflows

## Installation

### Prerequisites

1. ComfyUI installed and working
2. Python 3.11+ (conda environment recommended)
3. PyTorch with CUDA support (for GPU acceleration)
4. Access to SAM 3D Body models on HuggingFace

### Method 1: Manual Installation (Recommended)

1. **Navigate to ComfyUI custom nodes directory:**
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. **Clone or copy this repository:**
   ```bash
   # If this repository is already set up, you're good to go
   # Otherwise, ensure the ComfyUI-SAM3DBody folder is in custom_nodes/
   ```

3. **Clone the SAM 3D Body repository (required):**
   ```bash
   cd ../../  # Go to parent directory (e.g., sam3db_node)
   git clone https://github.com/facebookresearch/sam-3d-body.git
   ```

4. **Run the installation script:**
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-SAM3DBody
   python install.py
   ```

### Method 2: ComfyUI Manager (If available)

1. Open ComfyUI Manager
2. Search for "SAM3DBody"
3. Click Install

### Post-Installation Steps

1. **Request access to SAM 3D Body models on HuggingFace:**
   - Visit: https://huggingface.co/facebook/sam-3d-body-dinov3
   - Visit: https://huggingface.co/facebook/sam-3d-body-vith
   - Click "Request Access" and wait for approval

2. **Login to HuggingFace:**
   ```bash
   huggingface-cli login
   ```
   Enter your HuggingFace token when prompted.

3. **Restart ComfyUI**

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

## Example Workflow

1. **Load Image**: Use "Load Image" node to load a photo containing a person
2. **Load Model**: Connect to "Load SAM 3D Body Model" node
   - Set `model_source` to "huggingface"
   - Set `model_path` to "facebook/sam-3d-body-dinov3"
3. **Process**: Connect both to "SAM 3D Body: Process Image"
   - Adjust `bbox_threshold` if needed
   - Set `inference_type` to "full" for best quality
4. **Visualize**: Connect output to "SAM 3D Body: Visualize Mesh"
5. **Export** (optional): Connect mesh_data to "SAM 3D Body: Export Mesh"

Example workflow JSON files can be found in the `workflows/` folder.

## Example Assets

The `assets/` folder contains example images that you can use for testing:
- Example images from the original SAM 3D Body repository
- Various poses and viewpoints for testing generalization

## Troubleshooting

### Model not loading from HuggingFace

**Solution:**
1. Ensure you've requested and received access at https://huggingface.co/facebook/sam-3d-body-dinov3
2. Login with: `huggingface-cli login`
3. Check that your token has read permissions
4. Restart ComfyUI after logging in

### ImportError: No module named 'sam_3d_body'

**Solution:**
1. Ensure the `sam-3d-body` repository is cloned in the correct location
2. Run `python install.py` again
3. Check that the installation completed without errors

### Out of memory (CUDA)

**Solution:**
1. Reduce image resolution before processing
2. Use CPU device instead of CUDA
3. Close other GPU-intensive applications
4. Use the smaller model: `facebook/sam-3d-body-vith`

### Visualization not working

**Solution:**
1. Ensure the `sam-3d-body` repository is available at the expected path
2. Check that visualization dependencies are installed
3. Use the debug_image output from the Process node

### Detectron2 installation fails

**Solution:**
Detectron2 is optional for this node. The core functionality works without it. If you need detection features:
- On Linux: Follow standard Detectron2 installation
- On Windows: Pre-built wheels may be available
- On macOS: Limited support, CPU only

## Environment Variables

Set these environment variables for convenience:

```bash
export SAM3D_MHR_PATH="/path/to/mhr_model.pt"
export SAM3D_DETECTOR_PATH="/path/to/detector/models"
export SAM3D_SEGMENTOR_PATH="/path/to/segmentor/models"
export SAM3D_FOV_PATH="/path/to/fov/models"
```

## Performance Notes

- **First Load**: Model download and loading may take several minutes
- **Subsequent Runs**: Models are cached globally for fast reuse
- **GPU Memory**: Requires ~8GB VRAM for DINOv3 model, ~6GB for ViT-H
- **Processing Time**: ~2-5 seconds per image on modern GPU

## License

This wrapper follows the same license as SAM 3D Body. The SAM 3D Body model checkpoints and code are licensed under the [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE).

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

## Contributing

Contributions are welcome! Please:
1. Test your changes thoroughly
2. Follow the existing code style
3. Update documentation as needed
4. Submit issues for bugs or feature requests

## Support

For issues related to:
- **This ComfyUI wrapper**: Open an issue in this repository
- **SAM 3D Body model**: Visit https://github.com/facebookresearch/sam-3d-body
- **ComfyUI**: Visit https://github.com/comfyanonymous/ComfyUI
