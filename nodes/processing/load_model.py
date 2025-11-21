# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Model loading node for SAM 3D Body.

Loads the SAM 3D Body model with caching support.
"""

import os
import torch
from pathlib import Path

# Global cache - persists across node executions
_MODEL_CACHE = {}


class LoadSAM3DBodyModel:
    """
    Loads the SAM 3D Body model with caching.

    Models are cached globally and reused across executions to save memory
    and loading time. Supports both HuggingFace model IDs and local checkpoint paths.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_source": (["huggingface", "local"], {
                    "default": "huggingface",
                    "tooltip": "Load from HuggingFace (requires authentication) or local checkpoint"
                }),
                "model_path": ("STRING", {
                    "default": "facebook/sam-3d-body-dinov3",
                    "tooltip": "HuggingFace model ID (e.g., facebook/sam-3d-body-dinov3) or local checkpoint path"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on. Auto uses CUDA if available."
                }),
            },
            "optional": {
                "mhr_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to MHR (Momentum Human Rig) asset. Leave empty to use default or set SAM3D_MHR_PATH env var."
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace API token for accessing gated models. Get from https://huggingface.co/settings/tokens or leave empty to use HF_TOKEN env var / huggingface-cli login"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3DBody"

    def load_model(self, model_source, model_path, device="auto", mhr_path="", hf_token=""):
        """Load and cache the SAM 3D Body model."""

        # Determine actual device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check cache
        cache_key = f"{model_path}_{device}_{model_source}"
        if cache_key in _MODEL_CACHE:
            print(f"[SAM3DBody] [OK] Using cached model")
            return (_MODEL_CACHE[cache_key],)

        print(f"[SAM3DBody] Loading model from {model_path}...")
        print(f"[SAM3DBody] Using device: {device}")

        try:
            # Get MHR path from env or parameter
            mhr_asset_path = mhr_path or os.environ.get("SAM3D_MHR_PATH", "")

            # Set HuggingFace token if provided
            # The huggingface_hub library will automatically use HF_TOKEN env var
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                print(f"[SAM3DBody] Using provided HuggingFace token for authentication")
            elif os.environ.get("HF_TOKEN"):
                print(f"[SAM3DBody] Using HF_TOKEN from environment variable")
            else:
                print(f"[SAM3DBody] No HF token provided, using cached credentials from huggingface-cli login")

            # Import SAM 3D Body modules
            from sam_3d_body import load_sam_3d_body, load_sam_3d_body_hf

            # Load model based on source
            if model_source == "huggingface":
                print(f"[SAM3DBody] Loading from HuggingFace...")
                model, model_cfg = load_sam_3d_body_hf(
                    model_path,
                    device=torch.device(device)
                )
            else:  # local
                print(f"[SAM3DBody] Loading from local checkpoint...")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"Checkpoint not found at: {model_path}\n"
                        f"Please download the checkpoint first or use HuggingFace mode."
                    )

                model, model_cfg = load_sam_3d_body(
                    model_path,
                    device=torch.device(device),
                    mhr_path=mhr_asset_path if mhr_asset_path else None
                )

            # Create model dictionary
            model_dict = {
                "model": model,
                "model_cfg": model_cfg,
                "device": device,
                "model_path": model_path,
                "source": model_source,
            }

            # Cache it
            _MODEL_CACHE[cache_key] = model_dict

            print(f"[SAM3DBody] [OK] Model loaded successfully on {device}")
            return (model_dict,)

        except ImportError as e:
            print(f"[SAM3DBody] [ERROR] Failed to import vendored sam_3d_body module")
            print(f"[SAM3DBody] The sam_3d_body package should be vendored in this node's directory")
            print(f"[SAM3DBody] Please ensure ComfyUI-SAM3DBody/sam_3d_body/ exists with all files")
            print(f"[SAM3DBody] Run the install.py script to verify the installation")
            raise RuntimeError(f"Vendored sam_3d_body module not found. Check installation.") from e

        except Exception as e:
            print(f"[SAM3DBody] [ERROR] Failed to load model: {str(e)}")
            print(f"[SAM3DBody] Tried loading from: {model_path}")

            if model_source == "huggingface":
                print(f"[SAM3DBody] HuggingFace loading requires:")
                print(f"[SAM3DBody]   1. Request access at https://huggingface.co/{model_path}")
                print(f"[SAM3DBody]   2. Get your token from https://huggingface.co/settings/tokens")
                print(f"[SAM3DBody]   3. Either:")
                print(f"[SAM3DBody]      - Provide token in the 'hf_token' input field, OR")
                print(f"[SAM3DBody]      - Set HF_TOKEN environment variable, OR")
                print(f"[SAM3DBody]      - Run: huggingface-cli login")
            else:
                print(f"[SAM3DBody] For local loading:")
                print(f"[SAM3DBody]   1. Download checkpoint from HuggingFace")
                print(f"[SAM3DBody]   2. Set correct path to .ckpt file")

            raise


# Register node
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DBodyModel": LoadSAM3DBodyModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DBodyModel": "Load SAM 3D Body Model",
}
