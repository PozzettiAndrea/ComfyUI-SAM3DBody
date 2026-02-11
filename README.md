# ComfyUI-SAM3DBody

ComfyUI wrapper for Meta's SAM 3D Body - single-image full-body 3D human mesh recovery.

![body](docs/body.png)


https://github.com/user-attachments/assets/5b6c0e24-5c64-4413-b4b5-e8b244c51cae

https://github.com/user-attachments/assets/8429690a-a251-458f-8b4f-aad5e723525e

https://github.com/user-attachments/assets/2906d9b5-bdf7-4593-a3ae-1f2c866d2b2e

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

## License

This project uses a **dual-license structure**:

- **Wrapper Code** (ComfyUI integration): **MIT License** - See [LICENSE-MIT](LICENSE-MIT)
  - This includes all nodes, UI components, installation scripts, and ComfyUI integration code
  - Copyright (c) 2025 Andrea Pozzetti

- **SAM 3D Body Library** (vendored in `sam_3d_body/`): **SAM License** - See [LICENSE-SAM](LICENSE-SAM)
  - The core SAM 3D Body model and inference code
  - Copyright (c) Meta Platforms, Inc. and affiliates
  - Permissive research license allowing commercial use and derivative works

See [LICENSE](LICENSE) for complete license information and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for attributions.

### Using This Project

- ✅ You can freely use, modify, and distribute the wrapper code under MIT terms
- ✅ You can use SAM 3D Body for research and commercial purposes under SAM License terms
- ⚠️ When redistributing, include both LICENSE-MIT and LICENSE-SAM
- ⚠️ Acknowledge SAM 3D Body in publications (required by SAM License)

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/PN743tE5) for help, updates, and chat about 3D workflows in ComfyUI.

## Credits

[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) by Meta AI ([paper](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/))

Blender installation code adapted from [ComfyUI-UniRig](https://github.com/VAST-AI-Research/UniRig) (MIT License)
