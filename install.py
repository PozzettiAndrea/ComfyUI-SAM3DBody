#!/usr/bin/env python3
# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
#
# Portions of this file (Blender installation functions) are adapted from:
# ComfyUI-UniRig by Tsinghua University and Tripo AI
# https://github.com/VAST-AI-Research/UniRig
# Licensed under MIT License
"""
Installation script for ComfyUI SAM 3D Body.

Handles dependency installation and setup for SAM 3D Body integration.
"""

import subprocess
import sys
import os
import platform
import shutil
import tarfile
import zipfile
import urllib.request
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


# ============================================================================
# Prebuilt Wheel Installation Functions
# ============================================================================

def get_torch_cuda_version():
    """
    Get PyTorch version and CUDA version for wheel selection.

    Returns:
        tuple: (torch_version, cuda_tag) e.g. ("2.5.0", "cu121") or ("2.5.0", "cpu")
    """
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # Strip +cu121 etc

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                # Convert "12.1" to "cu121", "12.8" to "cu128"
                cuda_tag = "cu" + cuda_version.replace(".", "")
            else:
                cuda_tag = "cpu"
        else:
            cuda_tag = "cpu"

        return torch_version, cuda_tag
    except ImportError:
        return None, None


def install_detectron2_wheel():
    """
    Install detectron2 from prebuilt wheels with fallback to source.

    Uses https://miropsota.github.io/torch_packages_builder for prebuilt wheels.
    """
    torch_version, cuda_tag = get_torch_cuda_version()

    if not torch_version:
        print("[SAM3DBody] PyTorch not found, skipping detectron2 wheel install")
        return False

    print(f"[SAM3DBody] Detected PyTorch {torch_version} with {cuda_tag}")

    # Try prebuilt wheel first
    # Format: detectron2==0.6+<commit>pt<torch_version><cuda_tag>
    # Available for PyTorch 2.1.0 - 2.8.0
    wheel_index = "https://miropsota.github.io/torch_packages_builder"

    # Use a recent stable commit
    commit = "2a420ed"

    # Build wheel specifier
    wheel_spec = f"detectron2==0.6+{commit}pt{torch_version}{cuda_tag}"

    print(f"[SAM3DBody] Trying prebuilt detectron2 wheel: {wheel_spec}")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--extra-index-url", wheel_index,
        wheel_spec
    ]

    if run_command(cmd, f"Installing detectron2 from prebuilt wheel"):
        return True

    # Fallback: try without specific version (let pip find compatible wheel)
    print("[SAM3DBody] Specific wheel not found, trying any compatible wheel...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--extra-index-url", wheel_index,
        "detectron2"
    ]

    if run_command(cmd, "Installing detectron2 (any compatible wheel)"):
        return True

    # Final fallback: build from source
    print("[SAM3DBody] No prebuilt wheel available, falling back to source build...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
        "--no-build-isolation", "--no-deps"
    ]

    return run_command(cmd, "Installing detectron2 from source")


def install_pyg_extensions():
    """
    Install PyTorch Geometric extensions (torch-scatter, torch-sparse, torch-cluster)
    from prebuilt wheels.

    Uses https://data.pyg.org/whl/ for prebuilt wheels.
    """
    torch_version, cuda_tag = get_torch_cuda_version()

    if not torch_version:
        print("[SAM3DBody] PyTorch not found, skipping PyG extensions")
        return False

    print(f"[SAM3DBody] Installing PyG extensions for PyTorch {torch_version} + {cuda_tag}")

    # PyG wheel index
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"

    packages = ["torch_scatter", "torch_sparse", "torch_cluster"]

    cmd = [
        sys.executable, "-m", "pip", "install",
        "-f", wheel_url
    ] + packages

    if run_command(cmd, "Installing PyG extensions from prebuilt wheels"):
        return True

    # Fallback: try without wheel index (will attempt to build from source)
    print("[SAM3DBody] Prebuilt PyG wheels not found, trying pip install...")
    cmd = [sys.executable, "-m", "pip", "install"] + packages

    return run_command(cmd, "Installing PyG extensions (may build from source)")


# ============================================================================
# Blender Installation Functions (adapted from ComfyUI-UniRig)
# ============================================================================

def get_platform_info():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        plat = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return plat, arch


def get_blender_download_url(platform_name, architecture):
    """
    Get Blender 4.2 LTS download URL for the platform.

    Args:
        platform_name: "linux", "macos", or "windows"
        architecture: "x64" or "arm64"

    Returns:
        tuple: (download_url, version, filename) or (None, None, None) if not found
    """
    version = "4.2.3"
    base_url = "https://download.blender.org/release/Blender4.2"

    urls = {
        ("linux", "x64"): (
            f"{base_url}/blender-{version}-linux-x64.tar.xz",
            version,
            f"blender-{version}-linux-x64.tar.xz"
        ),
        ("macos", "x64"): (
            f"{base_url}/blender-{version}-macos-x64.dmg",
            version,
            f"blender-{version}-macos-x64.dmg"
        ),
        ("macos", "arm64"): (
            f"{base_url}/blender-{version}-macos-arm64.dmg",
            version,
            f"blender-{version}-macos-arm64.dmg"
        ),
        ("windows", "x64"): (
            f"{base_url}/blender-{version}-windows-x64.zip",
            version,
            f"blender-{version}-windows-x64.zip"
        ),
    }

    key = (platform_name, architecture)
    if key in urls:
        url, ver, filename = urls[key]
        print(f"[SAM3DBody Install] Using Blender {ver} for {platform_name}-{architecture}")
        return url, ver, filename

    return None, None, None


def download_file(url, dest_path):
    """Download file with progress."""
    print(f"[SAM3DBody Install] Downloading: {url}")
    print(f"[SAM3DBody Install] Destination: {dest_path}")

    last_printed_percent = [-1]  # Use list to allow modification in nested function

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)

        # Only print every 10% to reduce verbosity
        if percent >= last_printed_percent[0] + 10 or percent >= 100:
            sys.stdout.write(f"\r[SAM3DBody Install] Progress: {percent}%")
            sys.stdout.flush()
            last_printed_percent[0] = percent

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        sys.stdout.write("\n")
        sys.stdout.flush()
        print("[SAM3DBody Install] Download complete!")
        return True
    except Exception as e:
        print(f"\n[SAM3DBody Install] Error downloading: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS)."""
    print(f"[SAM3DBody Install] Extracting: {archive_path}")

    try:
        if archive_path.endswith(('.tar.gz', '.tar.xz', '.tar.bz2')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.dmg'):
            print("[SAM3DBody Install] DMG detected - mounting disk image...")

            mount_result = subprocess.run(
                ['hdiutil', 'attach', '-nobrowse', archive_path],
                capture_output=True,
                text=True
            )

            if mount_result.returncode != 0:
                print(f"[SAM3DBody Install] Error mounting DMG: {mount_result.stderr}")
                return False

            mount_point = None
            for line in mount_result.stdout.split('\n'):
                if '/Volumes/' in line:
                    mount_point = line.split('\t')[-1].strip()
                    break

            if not mount_point:
                print("[SAM3DBody Install] Error: Could not find mount point")
                return False

            try:
                blender_app = Path(mount_point) / "Blender.app"
                if blender_app.exists():
                    dest_app = Path(extract_to) / "Blender.app"
                    shutil.copytree(blender_app, dest_app)
                    print(f"[SAM3DBody Install] Copied Blender.app to: {dest_app}")
                else:
                    print(f"[SAM3DBody Install] Error: Blender.app not found in {mount_point}")
                    return False

            finally:
                subprocess.run(['hdiutil', 'detach', mount_point], check=False)

        else:
            print(f"[SAM3DBody Install] Error: Unknown archive format: {archive_path}")
            return False

        print(f"[SAM3DBody Install] Extraction complete!")
        return True

    except Exception as e:
        print(f"[SAM3DBody Install] Error extracting: {e}")
        return False


def find_blender_executable(blender_dir):
    """Find the blender executable in the extracted directory."""
    plat, _ = get_platform_info()

    if plat == "windows":
        exe_pattern = "**/blender.exe"
    elif plat == "macos":
        # macOS Blender.app has executable at Contents/MacOS/Blender (capital B)
        exe_pattern = "**/MacOS/Blender"
    else:  # linux
        exe_pattern = "**/blender"

    executables = list(Path(blender_dir).glob(exe_pattern))

    if executables:
        return executables[0]
    return None


def install_blender(target_dir=None):
    """
    Install Blender for mesh preprocessing.

    Args:
        target_dir: Optional target directory. If None, uses lib/blender under script directory.

    Returns:
        str: Path to Blender executable, or None if installation failed.
    """
    print("\n" + "="*60)
    print("ComfyUI-SAM3DBody: Blender Installation")
    print("="*60 + "\n")

    if target_dir is None:
        script_dir = Path(__file__).parent.absolute()
        target_dir = script_dir / "lib" / "blender"
    else:
        target_dir = Path(target_dir)

    # Check if Blender already installed
    blender_exe = find_blender_executable(target_dir)
    if blender_exe and blender_exe.exists():
        print("[SAM3DBody Install] Blender already installed at:")
        print(f"[SAM3DBody Install]   {blender_exe}")
        print("[SAM3DBody Install] Skipping download.")
        return str(blender_exe)

    # Detect platform
    plat, arch = get_platform_info()
    if not plat or not arch:
        print("[SAM3DBody Install] Error: Could not detect platform")
        print("[SAM3DBody Install] Please install Blender manually from: https://www.blender.org/download/")
        return None

    print(f"[SAM3DBody Install] Detected platform: {plat}-{arch}")

    # Get download URL
    url, version, filename = get_blender_download_url(plat, arch)
    if not url:
        print("[SAM3DBody Install] Error: Could not find Blender download for your platform")
        print("[SAM3DBody Install] Please install Blender manually from: https://www.blender.org/download/")
        return None

    # Create temporary download directory
    temp_dir = target_dir.parent / "_temp_blender_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        download_path = temp_dir / filename
        if not download_file(url, str(download_path)):
            return None

        # Extract
        target_dir.mkdir(parents=True, exist_ok=True)
        if not extract_archive(str(download_path), str(target_dir)):
            return None

        print("\n[SAM3DBody Install] Blender installation complete!")
        print(f"[SAM3DBody Install] Location: {target_dir}")

        # Find blender executable
        blender_exe = find_blender_executable(target_dir)

        if blender_exe:
            print(f"[SAM3DBody Install] Blender executable: {blender_exe}")
            return str(blender_exe)
        else:
            print("[SAM3DBody Install] Warning: Could not find blender executable")
            return None

    except Exception as e:
        print(f"\n[SAM3DBody Install] Error during installation: {e}")
        return None

    finally:
        # Cleanup temp files
        if temp_dir.exists():
            print("[SAM3DBody Install] Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Main Installation Function
# ============================================================================

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

    # 5. Install Detectron2 (required dependency) - use prebuilt wheels when available
    print("[SAM3DBody] Installing Detectron2...")
    if not install_detectron2_wheel():
        print("[SAM3DBody] [WARNING] Detectron2 installation failed")
        print("[SAM3DBody] This is optional but recommended for detection features")

    # 6. Install PyG extensions (torch-scatter, torch-sparse, torch-cluster) if needed
    print("[SAM3DBody] Checking PyG extensions...")
    try:
        import torch_scatter
        print("[SAM3DBody] [OK] PyG extensions already installed")
    except ImportError:
        print("[SAM3DBody] Installing PyG extensions...")
        if not install_pyg_extensions():
            print("[SAM3DBody] [WARNING] PyG extensions installation failed")
            print("[SAM3DBody] Some features may not be available")

    # 7. Platform-specific setup
    if sys.platform.startswith('linux'):
        print("[SAM3DBody] Detected Linux platform")
    elif sys.platform == 'win32':
        print("[SAM3DBody] Detected Windows platform")
    elif sys.platform == 'darwin':
        print("[SAM3DBody] Detected macOS platform")
        print("[SAM3DBody] [NOTE] CUDA not available on macOS, will use CPU")

    # 8. Verify installation
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
