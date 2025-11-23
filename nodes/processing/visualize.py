# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Visualization nodes for SAM 3D Body outputs.

Provides nodes for rendering and visualizing 3D mesh reconstructions.
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path
from ..base import numpy_to_comfy_image

# Add sam-3d-body to Python path if it exists
_SAM3D_BODY_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "sam-3d-body"
if _SAM3D_BODY_PATH.exists() and str(_SAM3D_BODY_PATH) not in sys.path:
    sys.path.insert(0, str(_SAM3D_BODY_PATH))


class SAM3DBodyVisualize:
    """
    Visualizes SAM 3D Body mesh reconstruction results.

    Renders the 3D mesh onto the input image for visualization purposes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Original input image to overlay mesh on"
                }),
                "render_mode": (["overlay", "mesh_only", "side_by_side"], {
                    "default": "overlay",
                    "tooltip": "How to display the mesh: overlay on image, mesh only, or side-by-side comparison"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_image",)
    FUNCTION = "visualize"
    CATEGORY = "SAM3DBody/visualization"

    def visualize(self, mesh_data, image, render_mode="overlay"):
        """Visualize the 3D mesh reconstruction."""

        print(f"[SAM3DBody] Visualizing mesh with mode: {render_mode}")

        try:
            from ..base import comfy_image_to_numpy

            # Get original image
            img_bgr = comfy_image_to_numpy(image)

            # Extract mesh components
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)
            camera = mesh_data.get("camera", None)
            raw_output = mesh_data.get("raw_output", {})

            if vertices is None or faces is None:
                print(f"[SAM3DBody] [WARNING] No mesh data available for visualization")
                return (image,)

            # Convert tensors to numpy if needed
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()

            print(f"[SAM3DBody] Rendering mesh with {len(vertices)} vertices, {len(faces)} faces")

            # Try to use the original visualization tools
            try:
                from pathlib import Path
                import sys
                sam_3d_body_path = Path(__file__).parent.parent.parent.parent.parent.parent / "sam-3d-body"
                if sam_3d_body_path.exists():
                    sys.path.insert(0, str(sam_3d_body_path))
                    from tools.vis_utils import visualize_sample_together

                    rendered = visualize_sample_together(img_bgr, raw_output, faces)

                    if render_mode == "mesh_only":
                        # Return just the rendered mesh (would need separate rendering)
                        result_img = rendered
                    elif render_mode == "side_by_side":
                        # Concatenate original and rendered side by side
                        result_img = np.hstack([img_bgr, rendered])
                    else:  # overlay
                        result_img = rendered

                    # Convert back to ComfyUI format
                    result_comfy = numpy_to_comfy_image(result_img)
                    print(f"[SAM3DBody] [OK] Visualization complete")
                    return (result_comfy,)

            except Exception as e:
                print(f"[SAM3DBody] [WARNING] Could not use visualization tools: {e}")
                print(f"[SAM3DBody] Returning original image")
                return (image,)

        except Exception as e:
            print(f"[SAM3DBody] [ERROR] Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original image on error
            return (image,)


class SAM3DBodyExportMesh:
    """
    Exports SAM 3D Body mesh to STL format.

    Saves the reconstructed 3D mesh as ASCII STL for use in 3D viewers and editors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
                "filename": ("STRING", {
                    "default": "output_mesh.stl",
                    "tooltip": "Output filename (exports as ASCII STL)"
                }),
                "output_dir": ("STRING", {
                    "default": "output",
                    "tooltip": "Output directory path (relative to ComfyUI root or absolute)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export_mesh"
    CATEGORY = "SAM3DBody/io"

    def export_mesh(self, mesh_data, filename="output_mesh.stl", output_dir="output"):
        """Export mesh to file."""

        print(f"[SAM3DBody] Exporting mesh to {filename}")

        try:
            import os
            from pathlib import Path

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Full output path
            full_path = output_path / filename

            # Extract mesh data
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)

            if vertices is None or faces is None:
                raise ValueError("No mesh data available to export")

            # Convert to numpy if needed
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()

            # Export to STL format
            self._export_stl(vertices, faces, full_path)

            print(f"[SAM3DBody] [OK] Mesh exported to {full_path}")
            return (filename,)

        except Exception as e:
            print(f"[SAM3DBody] [ERROR] Export failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _export_obj(self, vertices, faces, filepath):
        """Export mesh to OBJ format."""
        with open(filepath, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def _export_ply(self, vertices, faces, filepath):
        """Export mesh to PLY format."""
        with open(filepath, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def _export_stl(self, vertices, faces, filepath):
        """Export mesh to ASCII STL format."""
        import numpy as np

        # Apply 180° X-rotation to undo MHR coordinate transform (flip both Y and Z)
        # This matches what the renderer does for visualization
        vertices_flipped = vertices.copy()
        vertices_flipped[:, 1] = -vertices_flipped[:, 1]  # Flip Y
        vertices_flipped[:, 2] = -vertices_flipped[:, 2]  # Flip Z

        with open(filepath, 'w') as f:
            # Write STL header
            f.write("solid mesh\n")

            # Write each triangle face
            for face in faces:
                # Get the three vertices of the triangle
                v0 = vertices_flipped[int(face[0])]
                v1 = vertices_flipped[int(face[1])]
                v2 = vertices_flipped[int(face[2])]

                # Calculate face normal using cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)

                # Normalize the normal vector
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normal = normal / norm_length
                else:
                    normal = np.array([0.0, 0.0, 1.0])  # Default normal if degenerate

                # Write facet
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

            # Write STL footer
            f.write("endsolid mesh\n")


class SAM3DBodyGetVertices:
    """
    Extracts vertex data from SAM 3D Body output.

    Useful for custom processing or analysis of the reconstructed mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_vertices"
    CATEGORY = "SAM3DBody/utilities"

    def get_vertices(self, mesh_data):
        """Extract and display vertex information."""

        try:
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)
            joints = mesh_data.get("joints", None)

            info_lines = ["[SAM3DBody] Mesh Information:"]

            if vertices is not None:
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                info_lines.append(f"Vertices: {len(vertices)} points")
                info_lines.append(f"Vertex shape: {vertices.shape}")

            if faces is not None:
                if isinstance(faces, torch.Tensor):
                    faces = faces.cpu().numpy()
                info_lines.append(f"Faces: {len(faces)} triangles")

            if joints is not None:
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                info_lines.append(f"Joints: {len(joints)} keypoints")

            info = "\n".join(info_lines)
            print(info)

            return (info,)

        except Exception as e:
            error_msg = f"[SAM3DBody] [ERROR] Failed to get mesh info: {str(e)}"
            print(error_msg)
            return (error_msg,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyVisualize": SAM3DBodyVisualize,
    "SAM3DBodyExportMesh": SAM3DBodyExportMesh,
    "SAM3DBodyGetVertices": SAM3DBodyGetVertices,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyVisualize": "SAM 3D Body: Visualize Mesh",
    "SAM3DBodyExportMesh": "SAM 3D Body: Export Mesh",
    "SAM3DBodyGetVertices": "SAM 3D Body: Get Mesh Info",
}
