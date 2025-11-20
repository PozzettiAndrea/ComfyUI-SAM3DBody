# Example Workflows

This folder is for ComfyUI workflow JSON files demonstrating SAM 3D Body usage.

## Creating Workflows

To create example workflows:

1. Open ComfyUI
2. Build a workflow using SAM 3D Body nodes
3. Save the workflow (File -> Save or Ctrl+S)
4. Place the .json file in this folder

## Basic Workflow Structure

A basic SAM 3D Body workflow includes:

1. **Load Image** node
   - Load an input image containing a person

2. **Load SAM 3D Body Model** node
   - model_source: "huggingface"
   - model_path: "facebook/sam-3d-body-dinov3"
   - device: "auto"

3. **SAM 3D Body: Process Image** node
   - Connect model from step 2
   - Connect image from step 1
   - Set inference_type to "full"

4. **SAM 3D Body: Visualize Mesh** node
   - Connect mesh_data from step 3
   - Connect original image from step 1
   - Set render_mode to "overlay"

5. **Preview Image** node (optional)
   - Connect rendered_image from step 4 to preview results

6. **SAM 3D Body: Export Mesh** node (optional)
   - Connect mesh_data from step 3 to export .obj or .ply files

## Advanced Workflows

For advanced usage, explore:
- Using mask inputs for guided reconstruction
- Chaining multiple processing nodes
- Batch processing multiple images
- Combining with other ComfyUI nodes for creative effects

## Contributing

If you create useful workflows, consider sharing them by:
1. Adding them to this folder
2. Documenting what they demonstrate
3. Contributing back to the repository
