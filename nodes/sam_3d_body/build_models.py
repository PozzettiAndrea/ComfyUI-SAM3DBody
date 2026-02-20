# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import os
import torch
import comfy.model_management

from .model import SAM3DBody
from .utils.config import get_config

log = logging.getLogger("sam3dbody")


def load_sam_3d_body(checkpoint_path: str = "", device: str = None, mhr_path: str = "", dtype: torch.dtype = None):

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
    model_cfg.freeze()

    # Load safetensors checkpoint to CPU
    from safetensors.torch import load_file
    state_dict = load_file(str(checkpoint_path), device="cpu")

    # Build model on meta device (zero memory, no random init)
    # Note: convert_to_fp16() runs during __init__ but is a no-op on meta tensors.
    # We re-run it below after loading real weights.
    with torch.device("meta"):
        model = SAM3DBody(model_cfg)

    # Load checkpoint weights — assign=True replaces meta tensors with real data
    model.load_state_dict(state_dict, strict=False, assign=True)

    # Materialize any remaining meta tensors (params/buffers not in checkpoint)
    for name, param in list(model.named_parameters()):
        if param.device.type == 'meta':
            parts = name.split('.')
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._parameters[parts[-1]] = torch.nn.Parameter(
                torch.zeros(param.shape, dtype=param.dtype, device='cpu'),
                requires_grad=param.requires_grad,
            )
    for name, buf in list(model.named_buffers()):
        if buf.device.type == 'meta':
            parts = name.split('.')
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._buffers[parts[-1]] = torch.zeros(buf.shape, dtype=buf.dtype, device='cpu')

    # Re-initialize persistent=False buffers from config (not saved in checkpoint)
    model.image_mean = torch.tensor(model_cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1)
    model.image_std = torch.tensor(model_cfg.MODEL.IMAGE_STD).view(-1, 1, 1)

    # Debug: show key model info at load time
    import sys
    print(f"[DEBUG] build_models: USE_FP16={model_cfg.TRAIN.USE_FP16} FP16_TYPE={model_cfg.TRAIN.get('FP16_TYPE', 'float16')}", file=sys.stderr, flush=True)
    for name, p in model.named_parameters():
        if any(k in name for k in ['backbone.blocks.0.attn.qkv.weight', 'decoder.layers.0', 'init_pose.weight', 'head_pose.']):
            print(f"[DEBUG] build_models param: {name} dtype={p.dtype} device={p.device} shape={p.shape}", file=sys.stderr, flush=True)
    print(f"[DEBUG] build_models: backbone_dtype={model.backbone_dtype}", file=sys.stderr, flush=True)
    print(f"[DEBUG] build_models: image_mean={model.image_mean.flatten().tolist()} dtype={model.image_mean.dtype}", file=sys.stderr, flush=True)
    print(f"[DEBUG] build_models: image_std={model.image_std.flatten().tolist()} dtype={model.image_std.dtype}", file=sys.stderr, flush=True)

    log.info(f" backbone_dtype: {model.backbone_dtype}")
    log.info(f" image_mean: {model.image_mean.flatten().tolist()}")
    log.info(f" image_std: {model.image_std.flatten().tolist()}")

    model.eval()
    return model, model_cfg, mhr_path
