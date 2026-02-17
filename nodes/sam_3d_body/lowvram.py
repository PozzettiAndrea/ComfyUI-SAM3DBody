"""Low-VRAM compatibility for third-party models with ComfyUI's ModelPatcher.

Provides _enable_lowvram_cast() which swaps standard PyTorch nn.* layers to
comfy.ops.disable_weight_init.* equivalents, enabling ModelPatcher's per-layer
VRAM streaming.
"""

import torch


def _enable_lowvram_cast(model: torch.nn.Module) -> None:
    """Swap leaf modules to comfy.ops.disable_weight_init versions for lowvram support.

    ComfyUI's ModelPatcher can only partially-load modules that have the
    `comfy_cast_weights` attribute (from CastWeightBiasOp).  Native ComfyUI models
    use `comfy.ops.disable_weight_init.*` layers, but third-party models use plain
    `torch.nn.*`.  This function retroactively swaps __class__ on every leaf module
    so that ModelPatcher's lowvram layer-streaming works transparently.

    Also installs forward pre-hooks on non-leaf modules that own direct parameters
    or buffers (e.g. cls_token, pos_embed on ViT) so they get moved to the input
    device on-the-fly during lowvram inference.
    """
    try:
        from comfy.ops import disable_weight_init
    except ImportError:
        return

    _CLASS_MAP = {
        torch.nn.Linear: disable_weight_init.Linear,
        torch.nn.Conv1d: disable_weight_init.Conv1d,
        torch.nn.Conv2d: disable_weight_init.Conv2d,
        torch.nn.Conv3d: disable_weight_init.Conv3d,
        torch.nn.GroupNorm: disable_weight_init.GroupNorm,
        torch.nn.LayerNorm: disable_weight_init.LayerNorm,
        torch.nn.ConvTranspose2d: disable_weight_init.ConvTranspose2d,
        torch.nn.ConvTranspose1d: disable_weight_init.ConvTranspose1d,
        torch.nn.Embedding: disable_weight_init.Embedding,
    }

    cast_count = 0
    for _name, module in model.named_modules():
        comfy_cls = _CLASS_MAP.get(type(module))
        if comfy_cls is not None:
            module.__class__ = comfy_cls
            cast_count += 1

    # Install forward pre-hooks on non-leaf modules that own orphan params/buffers.
    # These are things like cls_token, pos_embed on ViT — tiny tensors that aren't
    # inside any leaf layer, so comfy_cast_weights doesn't cover them.
    hook_count = 0
    for _name, module in model.named_modules():
        if hasattr(module, 'comfy_cast_weights'):
            continue  # Leaf module — handled by cast_weights mechanism
        direct_params = list(module.named_parameters(recurse=False))
        direct_bufs = list(module.named_buffers(recurse=False))
        if not direct_params and not direct_bufs:
            continue  # No orphan tensors

        def _move_orphans_hook(mod, args, kwargs=None):
            # Infer target device from first tensor in args or kwargs
            device = None
            for a in args:
                if isinstance(a, torch.Tensor):
                    device = a.device
                    break
                if hasattr(a, 'feats') and isinstance(a.feats, torch.Tensor):
                    device = a.feats.device
                    break
            if device is None and kwargs:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
                    if hasattr(v, 'feats') and isinstance(v.feats, torch.Tensor):
                        device = v.feats.device
                        break
            if device is None:
                for p in mod.parameters():
                    if p.device.type == 'cuda':
                        device = p.device
                        break
            if device is None:
                return
            for _, p in mod.named_parameters(recurse=False):
                if p.data.device != device:
                    p.data = p.data.to(device)
            for _, b in mod.named_buffers(recurse=False):
                if b.device != device:
                    b.data = b.data.to(device)

        module.register_forward_pre_hook(_move_orphans_hook, with_kwargs=True)
        hook_count += 1

    import logging
    log = logging.getLogger("sam3dbody")
    if cast_count or hook_count:
        log.debug("Enabled lowvram: %d cast modules, %d orphan-param hooks", cast_count, hook_count)
