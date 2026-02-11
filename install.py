#!/usr/bin/env python3
# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Installation script for ComfyUI-SAM3DBody.

Uses comfy-env for declarative dependency management via comfy-env.toml.
"""

import sys
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("ComfyUI-UniRig Installation")
    print("=" * 60)

    from comfy_env import install, IsolatedEnvManager, discover_config
    from comfy_env.tools import find_blender

    node_root = Path(__file__).parent.absolute()

    # Run comfy-env install
    try:
        install(config=node_root / "comfy-env.toml", mode="isolated", node_dir=node_root)
    except Exception as e:
        print(f"\n[UniRig] Installation FAILED: {e}")
        print("[UniRig] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-UniRig/issues")
        return 1

    print("\n" + "=" * 60)
    print("[UniRig] Installation completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())