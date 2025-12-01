# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Model loading node for SAM 3D Body.

Loads the SAM 3D Body model with caching support.
"""

import os
import torch
import folder_paths

# Global cache - persists across node executions
_MODEL_CACHE = {}

# Default model path in ComfyUI models folder
DEFAULT_MODEL_PATH = os.path.join(folder_paths.models_dir, "sam3dbody")


class LoadSAM3DBodyModel:
    """
    Loads the SAM 3D Body model with caching.

    Models are cached globally and reused across executions to save memory
    and loading time. Checks local path first, then tries HuggingFace download.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": DEFAULT_MODEL_PATH,
                    "tooltip": "Path to SAM 3D Body model folder (contains model.ckpt and assets/mhr_model.pt)"
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace API token. If model not found locally, will attempt to download. Get token from https://huggingface.co/settings/tokens"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3DBody"

    def load_model(self, model_path, hf_token=""):
        """Load and cache the SAM 3D Body model."""

        # Auto-detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve to absolute path for clear error messages
        model_path = os.path.abspath(model_path)

        # Check cache
        cache_key = f"{model_path}_{device}"
        if cache_key in _MODEL_CACHE:
            return (_MODEL_CACHE[cache_key],)

        # Expected file paths
        ckpt_path = os.path.join(model_path, "model.ckpt")
        mhr_path = os.path.join(model_path, "assets", "mhr_model.pt")

        # Check if model exists locally
        model_exists = os.path.exists(ckpt_path) and os.path.exists(mhr_path)

        if not model_exists:
            # If no token provided, give instructions immediately
            if not hf_token:
                raise RuntimeError(
                    f"\n[SAM3DBody] Model not found.\n\n"
                    f"Please place the model files at:\n"
                    f"  {DEFAULT_MODEL_PATH}/\n"
                    f"    ├── model.ckpt          (SAM 3D Body checkpoint)\n"
                    f"    ├── model_config.yaml   (model configuration)\n"
                    f"    └── assets/\n"
                    f"        └── mhr_model.pt    (Momentum Human Rig model)\n\n"
                    f"To download automatically, provide your HuggingFace token:\n"
                    f"  1. Request access at https://huggingface.co/facebook/sam-3d-body-dinov3\n"
                    f"  2. Get your token from https://huggingface.co/settings/tokens\n"
                    f"  3. Enter the token in the 'hf_token' input field"
                )

            # Try to download with token
            os.environ["HF_TOKEN"] = hf_token

            try:
                from huggingface_hub import snapshot_download

                os.makedirs(model_path, exist_ok=True)
                snapshot_download(
                    repo_id="facebook/sam-3d-body-dinov3",
                    local_dir=model_path
                )

            except Exception as e:
                # Download failed - give user instructions
                raise RuntimeError(
                    f"\n[SAM3DBody] Download failed.\n\n"
                    f"Please manually place the model files at:\n"
                    f"  {DEFAULT_MODEL_PATH}/\n"
                    f"    ├── model.ckpt          (SAM 3D Body checkpoint)\n"
                    f"    ├── model_config.yaml   (model configuration)\n"
                    f"    └── assets/\n"
                    f"        └── mhr_model.pt    (Momentum Human Rig model)\n\n"
                    f"Download error: {e}"
                ) from e

        # Now load the model
        try:
            from sam_3d_body import load_sam_3d_body

            result = load_sam_3d_body(
                checkpoint_path=ckpt_path,
                device=device,
                mhr_path=mhr_path
            )
            # Handle both 2-value and 3-value returns from different sam_3d_body versions
            if len(result) == 3:
                model, model_cfg, mhr_path_used = result
            else:
                model, model_cfg = result
                mhr_path_used = mhr_path

            # Create model dictionary
            model_dict = {
                "model": model,
                "model_cfg": model_cfg,
                "device": device,
                "model_path": model_path,
                "mhr_path": mhr_path_used,
            }

            # Cache it
            _MODEL_CACHE[cache_key] = model_dict

            return (model_dict,)

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import sam_3d_body module. Check installation."
            ) from e


# Register node
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DBodyModel": LoadSAM3DBodyModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DBodyModel": "Load SAM 3D Body Model",
}
