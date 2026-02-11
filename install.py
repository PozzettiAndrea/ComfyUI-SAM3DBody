#!/usr/bin/env python3
# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Installation script for ComfyUI-SAM3DBody.

Uses comfy-env for declarative dependency management via comfy-env.toml.
"""

import sys
from pathlib import Path


def log(msg: str) -> None:
    """Log a message with prefix."""
    print(f"[SAM3DBody] {msg}")


def install() -> bool:
    """Main installation function using comfy-env."""
    log("=" * 60)
    log("Starting installation...")
    log("=" * 60)

    try:
        from comfy_env import install as comfy_install
        node_dir = Path(__file__).parent
        log("Installing isolated environment via comfy-env...")
        comfy_install(node_dir=node_dir, mode="isolated")
    except ImportError:
        log("[ERROR] comfy-env not found. Install it first:")
        log("  pip install comfy-env")
        return False
    except Exception as e:
        log(f"[ERROR] Installation failed: {e}")
        return False

    log("=" * 60)
    log("Installation complete!")
    log("=" * 60)
    log("")
    log("Next steps:")
    log("1. Request access: https://huggingface.co/facebook/sam-3d-body-dinov3")
    log("2. Login: huggingface-cli login")
    log("3. Use LoadSAM3DBodyModel node in ComfyUI")
    log("")

    return True


if __name__ == "__main__":
    success = install()
    sys.exit(0 if success else 1)
