"""Lazy proxy for comfy.ops.manual_cast to avoid importing comfy.ops at module load time.

comfy.ops may transitively trigger torch.cuda initialization, which crashes
in isolated pixi environments during metadata scans. This proxy defers the
import until first attribute access (i.e., when a nn.Module __init__ runs).
"""


class _LazyOps:
    __slots__ = ()
    _resolved = None

    def __getattr__(self, name):
        if _LazyOps._resolved is None:
            import comfy.ops
            _LazyOps._resolved = comfy.ops.manual_cast
        return getattr(_LazyOps._resolved, name)


ops = _LazyOps()
