#!/usr/bin/env python3
# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
ComfyUI-SAM3DBody prestartup script.

Automatically copies FBX viewer files and example assets to ComfyUI directories on startup.
Runs before ComfyUI's main initialization.
"""

import os
import shutil
from pathlib import Path

# Get paths for viewer files
SCRIPT_DIR = Path(__file__).parent
WEB_DIR = SCRIPT_DIR / "web"
THREE_DIR = WEB_DIR / "three"


def copy_fbx_viewer():
    """Copy FBX viewer files from comfy-3d-viewers package."""
    try:
        from comfy_3d_viewers import get_fbx_html_path, get_fbx_bundle_path

        # Ensure directories exist
        WEB_DIR.mkdir(parents=True, exist_ok=True)
        THREE_DIR.mkdir(parents=True, exist_ok=True)

        # Copy viewer_fbx.html to web/
        src_html = get_fbx_html_path()
        dst_html = WEB_DIR / "viewer_fbx.html"
        if os.path.exists(src_html):
            # Always copy if source is newer or destination doesn't exist
            if not dst_html.exists() or os.path.getmtime(src_html) > os.path.getmtime(dst_html):
                shutil.copy2(src_html, dst_html)
                print(f"[SAM3DBody] Copied viewer_fbx.html from comfy-3d-viewers")
            else:
                print(f"[SAM3DBody] viewer_fbx.html is up to date")
        else:
            print(f"[SAM3DBody] Warning: viewer_fbx.html not found in comfy-3d-viewers package")

        # Copy viewer-bundle.js to web/three/
        src_bundle = get_fbx_bundle_path()
        dst_bundle = THREE_DIR / "viewer-bundle.js"
        if os.path.exists(src_bundle):
            if not dst_bundle.exists() or os.path.getmtime(src_bundle) > os.path.getmtime(dst_bundle):
                shutil.copy2(src_bundle, dst_bundle)
                print(f"[SAM3DBody] Copied viewer-bundle.js from comfy-3d-viewers")
            else:
                print(f"[SAM3DBody] viewer-bundle.js is up to date")
        else:
            print(f"[SAM3DBody] Warning: viewer-bundle.js not found in comfy-3d-viewers package")

    except ImportError:
        print("[SAM3DBody] Warning: comfy-3d-viewers package not installed, FBX viewer may not work")
        print("[SAM3DBody] Install with: pip install comfy-3d-viewers")
    except Exception as e:
        print(f"[SAM3DBody] Error copying FBX viewer files: {e}")
        import traceback
        traceback.print_exc()


def copy_assets():
    """Copy all files from assets/ to ComfyUI/input/"""
    try:
        # Determine paths
        custom_node_dir = Path(__file__).parent
        comfyui_dir = custom_node_dir.parent.parent
        input_dir = comfyui_dir / "input"
        assets_src = custom_node_dir / "assets"

        # Check if assets directory exists
        if not assets_src.exists():
            print("[SAM3DBody] No assets/ directory found, skipping asset copy")
            return

        # Create input directory if it doesn't exist
        input_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files and directories from assets/
        copied_count = 0
        skipped_count = 0

        for item in assets_src.iterdir():
            # Skip hidden files and directories (like .ipynb_checkpoints)
            if item.name.startswith('.'):
                continue

            # Destination path
            dest = input_dir / item.name

            # Handle directories
            if item.is_dir():
                if dest.exists():
                    skipped_count += 1
                    continue
                try:
                    shutil.copytree(item, dest)
                    print(f"[SAM3DBody] Copied asset directory: {item.name} -> {dest}")
                    copied_count += 1
                except Exception as e:
                    print(f"[SAM3DBody] Failed to copy directory {item.name}: {e}")
                continue

            # Skip if file already exists
            if dest.exists():
                skipped_count += 1
                continue

            # Copy file
            try:
                shutil.copy2(item, dest)
                print(f"[SAM3DBody] Copied asset: {item.name} -> {dest}")
                copied_count += 1
            except Exception as e:
                print(f"[SAM3DBody] Failed to copy {item.name}: {e}")

        # Print summary
        if copied_count > 0:
            print(f"[SAM3DBody] Copied {copied_count} asset file(s) to {input_dir}")
        if skipped_count > 0:
            print(f"[SAM3DBody] Skipped {skipped_count} existing asset file(s)")

    except Exception as e:
        print(f"[SAM3DBody] Error copying assets: {e}")

# Run on import
if __name__ == "__main__":
    print("[SAM3DBody] Running prestartup script...")
    copy_fbx_viewer()
    copy_assets()
    print("[SAM3DBody] Prestartup script completed")
else:
    # Also run when imported by ComfyUI
    print("[SAM3DBody] Running prestartup script...")
    copy_fbx_viewer()
    copy_assets()
    print("[SAM3DBody] Prestartup script completed")
