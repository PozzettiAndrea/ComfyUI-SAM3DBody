# ComfyUI-SAM3DBody Project Structure

This document describes the structure and organization of the ComfyUI-SAM3DBody custom node package.

## Directory Structure

```
ComfyUI-SAM3DBody/
├── __init__.py                           # Main package entry point with pytest guard
├── install.py                            # Installation script for dependencies
├── requirements.txt                      # Python dependencies
├── README.md                             # User documentation
├── .gitignore                           # Git ignore patterns
├── PROJECT_STRUCTURE.md                 # This file
│
├── nodes/                               # All node implementations
│   ├── __init__.py                      # Aggregates all node mappings
│   ├── base.py                          # Common utilities and conversions
│   │
│   └── processing/                      # Processing nodes category
│       ├── __init__.py
│       ├── load_model.py                # LoadSAM3DBodyModel node
│       ├── process.py                   # SAM3DBodyProcess nodes (basic & advanced)
│       └── visualize.py                 # Visualization and export nodes
│
├── assets/                              # Example files and images
│   ├── README.md
│   ├── model_diagram.png
│   └── *.png                           # Example test images
│
└── workflows/                           # Example workflow JSON files
    └── README.md
```

## Node Categories

### SAM3DBody (Main)
- **LoadSAM3DBodyModel**: Model loading with caching

### SAM3DBody/processing
- **SAM3DBodyProcess**: Basic image processing for mesh reconstruction
- **SAM3DBodyProcessAdvanced**: Advanced processing with full control

### SAM3DBody/visualization
- **SAM3DBodyVisualize**: Mesh visualization and rendering

### SAM3DBody/io
- **SAM3DBodyExportMesh**: Export meshes to OBJ/PLY formats

### SAM3DBody/utilities
- **SAM3DBodyGetVertices**: Extract mesh information

## Data Flow

1. **Load Image** (ComfyUI) → IMAGE tensor
2. **LoadSAM3DBodyModel** → SAM3D_MODEL custom type
3. **SAM3DBodyProcess** (IMAGE + SAM3D_MODEL) → SAM3D_OUTPUT + IMAGE (debug)
4. **SAM3DBodyVisualize** (SAM3D_OUTPUT + IMAGE) → IMAGE (rendered)
5. **SAM3DBodyExportMesh** (SAM3D_OUTPUT) → STRING (file path)

## Custom Data Types

### SAM3D_MODEL
Dictionary containing:
- `model`: The SAM 3D Body model instance
- `model_cfg`: Model configuration
- `device`: Device the model is loaded on
- `model_path`: Path or HF model ID
- `source`: "huggingface" or "local"

### SAM3D_OUTPUT
Dictionary containing:
- `vertices`: Mesh vertices [N, 3] numpy array or tensor
- `faces`: Face indices [F, 3] numpy array
- `joints`: 3D keypoints/joints
- `camera`: Camera parameters
- `bboxes`: Detected bounding boxes
- `raw_output`: Full output dictionary from SAM3DBodyEstimator

## Key Features

### Model Caching
- Global `_MODEL_CACHE` dictionary in `load_model.py`
- Cache key: `f"{model_path}_{device}_{source}"`
- Models persist across workflow executions

### Data Conversions (base.py)
- `comfy_image_to_pil()`: ComfyUI IMAGE → PIL Image
- `pil_to_comfy_image()`: PIL Image → ComfyUI IMAGE
- `comfy_image_to_numpy()`: ComfyUI IMAGE → OpenCV BGR numpy
- `numpy_to_comfy_image()`: OpenCV BGR numpy → ComfyUI IMAGE
- `comfy_mask_to_numpy()`: ComfyUI MASK → numpy
- `numpy_to_comfy_mask()`: numpy → ComfyUI MASK

### Error Handling
All nodes include:
- Try-catch blocks for graceful error handling
- Informative error messages with troubleshooting hints
- Logging with `[SAM3DBody]` prefix
- Status indicators: `[OK]`, `[WARNING]`, `[ERROR]`

## Installation Flow

1. User runs `python install.py`
2. Script installs requirements.txt dependencies
3. Checks for sam-3d-body repository at `../../sam-3d-body`
4. Installs sam-3d-body in editable mode if found
5. Installs Detectron2 (optional)
6. Verifies installation

## Pytest Guard

All `__init__.py` files include pytest guard:
```python
if 'pytest' not in sys.modules:
    # Load nodes normally
else:
    # Provide empty mappings for testing
```

This ensures the package can be tested without loading heavy dependencies.

## Dependencies

### Core
- torch, torchvision, numpy, pillow, opencv-python

### SAM 3D Body
- pytorch-lightning, pyrender, timm, einops, etc.
- See requirements.txt for full list

### Optional
- Detectron2 (for human detection)
- MoGe (for FOV estimation)
- SAM2 (for segmentation)

## Usage Workflow

1. **Setup**:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-SAM3DBody
   python install.py
   huggingface-cli login
   ```

2. **In ComfyUI**:
   - Add "Load Image" node
   - Add "Load SAM 3D Body Model" node
   - Add "SAM 3D Body: Process Image" node
   - Connect: Image + Model → Process
   - Add visualization/export nodes as needed

3. **Outputs**:
   - Visualized mesh overlays
   - Exported .obj/.ply mesh files
   - Mesh data for further processing

## Extension Points

To add new nodes:
1. Create node class in `nodes/processing/` (or new category)
2. Define `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`
3. Add to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
4. Import in `nodes/__init__.py`

## Standards Compliance

This package follows the unified ComfyUI wrapper guidelines:
- ✅ Pytest guard in __init__.py
- ✅ Tooltips for all inputs
- ✅ Model caching pattern
- ✅ Comprehensive error handling
- ✅ Data type conversions in base.py
- ✅ install.py for dependencies
- ✅ Example assets included
- ✅ Workflows folder ready
- ✅ Comprehensive README

## License

Follows SAM 3D Body license from Meta AI.
See: https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE
