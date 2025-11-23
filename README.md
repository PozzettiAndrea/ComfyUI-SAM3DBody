# ComfyUI-SAM3DBody

ComfyUI wrapper for Meta's SAM 3D Body - single-image full-body 3D human mesh recovery.

![body](docs/body.png)

## Installation

Run the installation script: `python install.py`

**macOS Note**: If you encounter build errors with `xtcocotools`, the install script handles this automatically. For manual installation: `pip install --no-build-isolation xtcocotools`

## Nodes

- **Load SAM 3D Body Model** - Load model from HuggingFace (`facebook/sam-3d-body-dinov3`) or local checkpoint
- **Process Image** - Reconstruct 3D mesh from image with optional mask/detection (full/body/hand modes)
- **Process Image (Advanced)** - Full control over detection, segmentation, and FOV estimation
- **Visualize Mesh** - Render 3D mesh overlay on image
- **Export Mesh** - Save mesh as OBJ/PLY file
- **Get Mesh Info** - Display mesh statistics

## Credits

[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) by Meta AI ([paper](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/))