# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch
import comfy.model_management

from .models.meta_arch import SAM3DBody
from .utils.config import get_config


def load_sam_3d_body(checkpoint_path: str = "", device: str = None, mhr_path: str = "", attn_backend: str = "sdpa", dtype: torch.dtype = None):

    if device is None:
        device = str(comfy.model_management.get_torch_device())

    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    tried_paths = [model_cfg]

    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )
        tried_paths.append(model_cfg)

    if not os.path.exists(model_cfg):
        # Use bundled default config
        bundled_config = os.path.join(
            os.path.dirname(__file__), "configs", "model_config.yaml"
        )
        tried_paths.append(bundled_config)
        if os.path.exists(bundled_config):
            model_cfg = bundled_config
        else:
            raise FileNotFoundError(
                f"Could not find model_config.yaml in any of these locations:\n" +
                "\n".join(f"  - {p}" for p in tried_paths) +
                f"\n\nFor local model loading, please ensure model_config.yaml is in the same directory as your checkpoint."
            )

    model_cfg = get_config(model_cfg)

    # Configure model
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.MODEL.BACKBONE.ATTN_BACKEND = attn_backend
    model_cfg.freeze()

    # Load checkpoint to CPU first
    import comfy.utils
    state_dict = comfy.utils.load_torch_file(str(checkpoint_path))

    # Build model on meta device (zero memory, no random init)
    with torch.device("meta"):
        model = SAM3DBody(model_cfg)
    model.load_state_dict(state_dict, strict=False, assign=True)

    model = model.to(device)
    model.eval()
    return model, model_cfg, mhr_path


