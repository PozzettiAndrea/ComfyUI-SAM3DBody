import logging
import os
import folder_paths
from comfy_api.latest import io

log = logging.getLogger("sam3dbody")


def _comfy_tqdm():
    """tqdm that shows download progress in ComfyUI's UI."""
    try:
        import comfy.utils
        import tqdm as _tqdm_mod
    except ImportError:
        return None
    holder = {"pbar": None, "total": 0, "done": 0}
    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)
        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret
    return _T


# Default model path in ComfyUI models folder
DEFAULT_MODEL_PATH = os.path.join(folder_paths.models_dir, "sam3dbody")
os.makedirs(DEFAULT_MODEL_PATH, exist_ok=True)
folder_paths.add_model_folder_path("sam3dbody", DEFAULT_MODEL_PATH)


class LoadSAM3DBodyModel(io.ComfyNode):
    """
    Prepares SAM 3D Body model configuration.

    Returns a config dict with model paths. The actual model is loaded
    lazily inside the isolated worker when inference runs.
    """

    REPO_ID = "apozz/sam-3d-body-safetensors"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSAM3DBodyModel",
            display_name="(Down)Load SAM 3D Body Model",
            category="SAM3DBody",
            inputs=[
                io.Combo.Input("precision", options=["fp32", "auto", "bf16", "fp16"],
                               default="fp32",
                               tooltip="Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."),
            ],
            outputs=[
                io.Custom("SAM3D_MODEL").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, precision="auto"):
        """Prepare model config (actual loading happens in inference nodes)."""
        import comfy.model_management as mm
        device = mm.get_torch_device()

        # Resolve "auto" precision to concrete string (bf16/fp16/fp32)
        # Stored as string because comfy_env serializes via JSON (torch.dtype not JSON-safe)
        if precision == "auto":
            if mm.should_use_bf16(device):
                precision = "bf16"
            elif mm.should_use_fp16(device):
                precision = "fp16"
            else:
                precision = "fp32"

        log.info(f"Precision: {precision}")

        model_path = DEFAULT_MODEL_PATH

        # Expected file paths
        model_file = os.path.join(model_path, "model.safetensors")
        mhr_path = os.path.join(model_path, "assets", "mhr_model.pt")

        # Check if model exists locally, download if not
        model_exists = os.path.exists(model_file) and os.path.exists(mhr_path)

        if not model_exists:
            try:
                from huggingface_hub import snapshot_download

                log.info(f"Model not found locally. Downloading from HuggingFace...")
                os.makedirs(model_path, exist_ok=True)
                snapshot_download(
                    repo_id=cls.REPO_ID,
                    local_dir=model_path,
                    tqdm_class=_comfy_tqdm(),
                )
                log.info(f"Download complete.")

            except Exception as e:
                raise RuntimeError(
                    f"\n[SAM3DBody] Download failed.\n\n"
                    f"Please manually download from:\n"
                    f"  https://huggingface.co/{cls.REPO_ID}\n\n"
                    f"And place the model files at:\n"
                    f"  {DEFAULT_MODEL_PATH}/\n"
                    f"    +-- model.safetensors   (SAM 3D Body weights)\n"
                    f"    +-- model_config.yaml   (model configuration)\n"
                    f"    \\-- assets/\n"
                    f"        \\-- mhr_model.pt    (Momentum Human Rig model)\n\n"
                    f"Download error: {e}"
                ) from e

        # Return config dict (not the actual model -- loading happens in inference nodes)
        model_config = {
            "model_path": model_path,
            "ckpt_path": model_file,
            "mhr_path": mhr_path,
            "precision": precision,
        }

        return io.NodeOutput(model_config)


# Register node
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DBodyModel": LoadSAM3DBodyModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DBodyModel": "(Down)Load SAM 3D Body Model",
}
