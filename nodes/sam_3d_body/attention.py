import logging

from comfy.ldm.modules.attention import optimized_attention_for_device

log = logging.getLogger("sam3dbody")


def sam3d_attention(q, k, v, heads, mask=None, skip_reshape=False):
    """Dispatch attention using ComfyUI's device-appropriate backend."""
    fn = optimized_attention_for_device(q.device, mask=mask is not None)
    return fn(q, k, v, heads=heads, mask=mask, skip_reshape=skip_reshape)
