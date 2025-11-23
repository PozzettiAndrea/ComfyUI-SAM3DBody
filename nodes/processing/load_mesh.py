# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Mesh loading node for SAM 3D Body.

Loads mesh files (FBX, OBJ, PLY, STL, GLB, etc.) from ComfyUI folders.
"""

import os
import numpy as np
import torch
import folder_paths


class SAM3DBodyLoadMesh:
    """
    Load a mesh from ComfyUI input or output folder.

    Supports various formats: FBX, OBJ, PLY, STL, GLB, GLTF, etc.
    Returns mesh data in SAM3DBody format.
    """

    # Supported mesh extensions
    SUPPORTED_EXTENSIONS = ['.fbx', '.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.dae', '.3ds']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder": (["input", "output"], {
                    "default": "output",
                    "tooltip": "Source folder to load mesh from"
                }),
                "file_path": ("COMBO", {
                    "remote": {
                        "route": "/sam3d/mesh_files",
                        "refresh_button": True,
                    },
                    "tooltip": "Mesh file to load. Supports FBX, OBJ, PLY, STL, GLB, etc."
                }),
            },
        }

    RETURN_TYPES = ("SAM3D_OUTPUT",)
    RETURN_NAMES = ("mesh_data",)
    FUNCTION = "load_mesh"
    CATEGORY = "SAM3DBody/IO"

    @classmethod
    def get_mesh_files(cls):
        """Get list of available mesh files in input and output folders."""
        mesh_files = []

        # Scan input folder
        input_dir = folder_paths.get_input_directory()
        if os.path.exists(input_dir):
            # Check input/3d subdirectory first
            input_3d = os.path.join(input_dir, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file)
                if os.path.isfile(file_path):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(file)

        # Scan output folder
        output_dir = folder_paths.get_output_directory()
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        if f"[output] {file}" not in [f"[output] {f}" for f in mesh_files]:
                            mesh_files.append(f"[output] {file}")

        return sorted(mesh_files)

    @classmethod
    def IS_CHANGED(cls, source_folder, file_path):
        """Force re-execution when file changes."""
        full_path = cls._resolve_file_path(source_folder, file_path)
        if full_path and os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return f"{source_folder}:{file_path}"

    @classmethod
    def _resolve_file_path(cls, source_folder, file_path):
        """Resolve the full path to the mesh file."""
        # Remove [output] prefix if present
        clean_path = file_path.replace("[output] ", "")

        if source_folder == "input":
            input_dir = folder_paths.get_input_directory()
            # Try input/3d first
            input_3d_path = os.path.join(input_dir, "3d", clean_path)
            if os.path.exists(input_3d_path):
                return input_3d_path
            # Then try input root
            input_path = os.path.join(input_dir, clean_path)
            if os.path.exists(input_path):
                return input_path
        else:  # output
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, clean_path)
            if os.path.exists(output_path):
                return output_path

        # Try as absolute path
        if os.path.exists(file_path):
            return file_path

        return None

    def load_mesh(self, source_folder, file_path):
        """
        Load mesh from file.

        Args:
            source_folder: "input" or "output"
            file_path: Path to mesh file

        Returns:
            tuple: (mesh_data,)
        """
        print(f"[SAM3DBodyLoadMesh] Loading mesh from {source_folder} folder...")

        # Resolve full path
        full_path = self._resolve_file_path(source_folder, file_path)

        if full_path is None:
            raise FileNotFoundError(
                f"Mesh file not found: {file_path}\n"
                f"Searched in: {source_folder} folder\n"
                f"Make sure the file exists in the selected folder."
            )

        print(f"[SAM3DBodyLoadMesh] Loading: {full_path}")

        # Load mesh using trimesh
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh library not found. Install it with: pip install trimesh"
            )

        # Load the mesh
        loaded = trimesh.load(full_path, force='mesh')

        # Handle Scene vs Mesh
        if isinstance(loaded, trimesh.Scene):
            print(f"[SAM3DBodyLoadMesh] Converting Scene to mesh ({len(loaded.geometry)} geometries)")
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"Failed to load mesh or mesh is empty: {full_path}")

        print(f"[SAM3DBodyLoadMesh] Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Clean up mesh
        verts_before = len(mesh.vertices)
        faces_before = len(mesh.faces)

        mesh.merge_vertices()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()

        verts_after = len(mesh.vertices)
        faces_after = len(mesh.faces)

        if verts_before != verts_after or faces_before != faces_after:
            print(f"[SAM3DBodyLoadMesh] Cleanup: {verts_before}->{verts_after} vertices, {faces_before}->{faces_after} faces")

        # Convert to SAM3DBody format
        vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
        faces = torch.from_numpy(mesh.faces.astype(np.int64))

        # Create mesh_data in SAM3DBody format
        mesh_data = {
            "vertices": vertices,
            "faces": faces,
            "source_file": full_path,
            "file_name": os.path.basename(full_path),
        }

        print(f"[SAM3DBodyLoadMesh] Successfully loaded: {len(vertices)} vertices, {len(faces)} faces")

        return (mesh_data,)


class SAM3DBodySelectMesh:
    """
    Select a mesh file and return its path as a string.

    Lightweight node for selecting files without loading them.
    Useful for passing file paths to preview nodes.
    """

    # Supported mesh extensions
    SUPPORTED_EXTENSIONS = ['.fbx', '.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.dae', '.3ds']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder": (["input", "output"], {
                    "default": "output",
                    "tooltip": "Source folder to select mesh from"
                }),
                "file_path": ("COMBO", {
                    "remote": {
                        "route": "/sam3d/mesh_files",
                        "refresh_button": True,
                    },
                    "tooltip": "Mesh file to select"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "select_mesh"
    CATEGORY = "SAM3DBody/IO"

    @classmethod
    def IS_CHANGED(cls, source_folder, file_path):
        """Force re-execution when file changes."""
        full_path = SAM3DBodyLoadMesh._resolve_file_path(source_folder, file_path)
        if full_path and os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return f"{source_folder}:{file_path}"

    def select_mesh(self, source_folder, file_path):
        """
        Select mesh file and return its basename.

        Args:
            source_folder: "input" or "output"
            file_path: Path to mesh file

        Returns:
            tuple: (basename,)
        """
        print(f"[SAM3DBodySelectMesh] Selecting mesh from {source_folder} folder...")

        # Remove [output] prefix if present
        clean_path = file_path.replace("[output] ", "")

        # Resolve full path to verify file exists
        full_path = SAM3DBodyLoadMesh._resolve_file_path(source_folder, file_path)

        if full_path is None:
            raise FileNotFoundError(
                f"Mesh file not found: {file_path}\n"
                f"Searched in: {source_folder} folder"
            )

        # Return just the basename (for preview nodes)
        basename = os.path.basename(full_path)
        print(f"[SAM3DBodySelectMesh] Selected: {basename}")

        return (basename,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyLoadMesh": SAM3DBodyLoadMesh,
    "SAM3DBodySelectMesh": SAM3DBodySelectMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyLoadMesh": "SAM 3D Body: Load Mesh",
    "SAM3DBodySelectMesh": "SAM 3D Body: Select Mesh File",
}
