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

    # 1. Install build dependencies first (needed for packages that build C extensions)
    print("[SAM3DBody] Installing build dependencies...")
    cmd = [sys.executable, "-m", "pip", "install", "numpy", "cython", "wheel"]
    run_command(cmd, "Installing numpy, cython, and wheel")

    # 2. macOS-specific: Install xtcocotools with --no-build-isolation
    # This is needed because xtcocotools imports numpy in setup.py but pip's
    # build isolation prevents it from seeing the numpy we just installed
    if sys.platform == 'darwin':
        print("[SAM3DBody] macOS detected: Installing xtcocotools with special flags...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-build-isolation", "xtcocotools>=1.14"
        ]
        if not run_command(cmd, "Installing xtcocotools (macOS workaround)"):
            print("[SAM3DBody] [WARNING] xtcocotools installation failed")
            print("[SAM3DBody] You may need to install it manually:")
            print("[SAM3DBody]   pip install --no-build-isolation xtcocotools")

    # 3. Install Python dependencies from requirements.txt
    if requirements_file.exists():
        print(f"[SAM3DBody] Installing dependencies from {requirements_file}")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        if not run_command(cmd, "Installing Python dependencies"):
            print("[SAM3DBody] [WARNING] Some dependencies failed to install")
            print("[SAM3DBody] You may need to install them manually")

    # 4. Check SAM 3D Body library (vendored in sam_3d_body/)
    print("[SAM3DBody] Checking for vendored SAM 3D Body library...")
    vendored_path = script_dir / "sam_3d_body"

    if vendored_path.exists() and (vendored_path / "__init__.py").exists():
        print(f"[SAM3DBody] [OK] Found vendored sam_3d_body at: {vendored_path}")
        print(f"[SAM3DBody] The sam_3d_body package is vendored within this custom node")
    else:
        print(f"[SAM3DBody] [ERROR] Vendored sam_3d_body package not found!")
        print(f"[SAM3DBody] Expected at: {vendored_path}")
        print(f"[SAM3DBody] Please ensure the sam_3d_body directory exists with proper __init__.py files")

    # 5. Install Detectron2 (required dependency)
    print("[SAM3DBody] Installing Detectron2...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
        "--no-build-isolation", "--no-deps"
    ]
    if not run_command(cmd, "Installing Detectron2"):
        print("[SAM3DBody] [WARNING] Detectron2 installation failed")
        print("[SAM3DBody] This is optional but recommended for detection features")

    # 6. Platform-specific setup
    if sys.platform.startswith('linux'):
        print("[SAM3DBody] Detected Linux platform")
    elif sys.platform == 'win32':
        print("[SAM3DBody] Detected Windows platform")
        print("[SAM3DBody] [WARNING] Windows support may be limited")
    elif sys.platform == 'darwin':
        print("[SAM3DBody] Detected macOS platform")
        print("[SAM3DBody] [WARNING] CUDA not available on macOS, will use CPU")

    # 7. Verify installation
    print("[SAM3DBody] Verifying installation...")
    try:
        import torch
        print(f"[SAM3DBody] PyTorch version: {torch.__version__}")
        print(f"[SAM3DBody] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[SAM3DBody] CUDA version: {torch.version.cuda}")

        # Try importing vendored sam_3d_body
        try:
            # Add the custom node directory to path to import vendored package
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            import sam_3d_body
            print(f"[SAM3DBody] sam_3d_body version: {sam_3d_body.__version__}")
            print("[SAM3DBody] [OK] Vendored sam_3d_body imported successfully")
        except ImportError as e:
            print(f"[SAM3DBody] [WARNING] Could not import vendored sam_3d_body: {e}")
            print("[SAM3DBody] The sam_3d_body package should be located at:")
            print(f"[SAM3DBody] {script_dir / 'sam_3d_body'}")

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
