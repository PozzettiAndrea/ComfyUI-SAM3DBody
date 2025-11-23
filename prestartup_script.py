#!/usr/bin/env python3
# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
ComfyUI-SAM3DBody prestartup script.

Automatically copies example assets and workflows to ComfyUI directories on startup.
Runs before ComfyUI's main initialization.
"""

import os
import shutil
from pathlib import Path


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

        # Copy all files from assets/
        copied_count = 0
        skipped_count = 0

        for item in assets_src.iterdir():
            # Skip hidden files and directories (like .ipynb_checkpoints)
            if item.name.startswith('.'):
                continue

            # Skip directories (only copy files)
            if item.is_dir():
                continue

            # Destination path
            dest = input_dir / item.name

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


def copy_workflows():
    """Copy all workflow files from workflows/ to ComfyUI/user/default/workflows/"""
    try:
        # Determine paths
        custom_node_dir = Path(__file__).parent
        comfyui_dir = custom_node_dir.parent.parent
        workflow_dir = comfyui_dir / "user" / "default" / "workflows"
        workflows_src = custom_node_dir / "workflows"

        # Check if workflows directory exists
        if not workflows_src.exists():
            print("[SAM3DBody] No workflows/ directory found, skipping workflow copy")
            return

        # Create workflow directory if it doesn't exist
        workflow_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from workflows/
        copied_count = 0
        skipped_count = 0

        for item in workflows_src.iterdir():
            # Skip hidden files and directories (like .ipynb_checkpoints)
            if item.name.startswith('.'):
                continue

            # Skip directories (only copy files)
            if item.is_dir():
                continue

            # Destination path with SAM3DB- prefix
            dest = workflow_dir / f"SAM3DB-{item.name}"

            # Skip if file already exists
            if dest.exists():
                skipped_count += 1
                continue

            # Copy file
            try:
                shutil.copy2(item, dest)
                print(f"[SAM3DBody] Copied workflow: {item.name} -> {dest}")
                copied_count += 1
            except Exception as e:
                print(f"[SAM3DBody] Failed to copy {item.name}: {e}")

        # Print summary
        if copied_count > 0:
            print(f"[SAM3DBody] Copied {copied_count} workflow file(s) to {workflow_dir}")
        if skipped_count > 0:
            print(f"[SAM3DBody] Skipped {skipped_count} existing workflow file(s)")

    except Exception as e:
        print(f"[SAM3DBody] Error copying workflows: {e}")


# Run on import
if __name__ == "__main__":
    print("[SAM3DBody] Running prestartup script...")
    copy_assets()
    copy_workflows()
    print("[SAM3DBody] Prestartup script completed")
else:
    # Also run when imported by ComfyUI
    print("[SAM3DBody] Running prestartup script...")
    copy_assets()
    copy_workflows()
    print("[SAM3DBody] Prestartup script completed")
