#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Installation script for ComfyUI SAM 3D Body.

Handles dependency installation and setup for SAM 3D Body integration.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"[SAM3DBody] {description}...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=False
        )
        print(f"[SAM3DBody] [OK] {description} complete")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[SAM3DBody] [ERROR] {description} failed")
        if e.stderr:
            print(f"[SAM3DBody] Error: {e.stderr}")
        return False


def install():
    """Main installation function."""
    print("=" * 70)
    print("[SAM3DBody] Starting installation...")
    print("=" * 70)

    # Get paths
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"
    sam_3d_body_path = script_dir.parent.parent.parent / "sam-3d-body"

    # 1. Install Python dependencies from requirements.txt
    if requirements_file.exists():
        print(f"[SAM3DBody] Installing dependencies from {requirements_file}")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        if not run_command(cmd, "Installing Python dependencies"):
            print("[SAM3DBody] [WARNING] Some dependencies failed to install")
            print("[SAM3DBody] You may need to install them manually")

    # 2. Check SAM 3D Body library
    print("[SAM3DBody] Checking for SAM 3D Body library...")

    # Check if sam-3d-body repo exists in parent directory
    if sam_3d_body_path.exists():
        print(f"[SAM3DBody] [OK] Found sam-3d-body at: {sam_3d_body_path}")
        print(f"[SAM3DBody] NOTE: sam-3d-body doesn't need installation, it will be imported directly")
        print(f"[SAM3DBody] The nodes will add it to sys.path automatically")
    else:
        print(f"[SAM3DBody] [WARNING] sam-3d-body repository not found at {sam_3d_body_path}")
        print("[SAM3DBody] Please clone the repository:")
        print(f"[SAM3DBody] cd {script_dir.parent.parent.parent}")
        print("[SAM3DBody] git clone https://github.com/facebookresearch/sam-3d-body.git")
        print("[SAM3DBody] Then run this install script again")

    # 3. Install Detectron2 (required dependency)
    print("[SAM3DBody] Installing Detectron2...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
        "--no-build-isolation", "--no-deps"
    ]
    if not run_command(cmd, "Installing Detectron2"):
        print("[SAM3DBody] [WARNING] Detectron2 installation failed")
        print("[SAM3DBody] This is optional but recommended for detection features")

    # 4. Platform-specific setup
    if sys.platform.startswith('linux'):
        print("[SAM3DBody] Detected Linux platform")
    elif sys.platform == 'win32':
        print("[SAM3DBody] Detected Windows platform")
        print("[SAM3DBody] [WARNING] Windows support may be limited")
    elif sys.platform == 'darwin':
        print("[SAM3DBody] Detected macOS platform")
        print("[SAM3DBody] [WARNING] CUDA not available on macOS, will use CPU")

    # 5. Verify installation
    print("[SAM3DBody] Verifying installation...")
    try:
        import torch
        print(f"[SAM3DBody] PyTorch version: {torch.__version__}")
        print(f"[SAM3DBody] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[SAM3DBody] CUDA version: {torch.version.cuda}")

        # Try importing sam_3d_body
        try:
            # Add sam-3d-body to path temporarily for verification
            if sam_3d_body_path.exists():
                sys.path.insert(0, str(sam_3d_body_path))

            import sam_3d_body
            print(f"[SAM3DBody] sam_3d_body version: {sam_3d_body.__version__}")
            print("[SAM3DBody] [OK] sam_3d_body imported successfully")
        except ImportError as e:
            print(f"[SAM3DBody] [WARNING] Could not import sam_3d_body: {e}")
            print("[SAM3DBody] Make sure the sam-3d-body repository is cloned to:")
            print(f"[SAM3DBody] {sam_3d_body_path}")

        print("=" * 70)
        print("[SAM3DBody] Installation complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Request access to models at https://huggingface.co/facebook/sam-3d-body-dinov3")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Use the LoadSAM3DBodyModel node in ComfyUI to load models")
        print()
        return True

    except ImportError as e:
        print(f"[SAM3DBody] [ERROR] Installation verification failed: {e}")
        print("[SAM3DBody] Some dependencies may be missing")
        print("[SAM3DBody] Please check the error messages above and install missing packages")
        return False


if __name__ == "__main__":
    success = install()
    sys.exit(0 if success else 1)
