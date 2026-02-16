import logging
import os
import folder_paths

log = logging.getLogger("sam3dbody")

# Default model path in ComfyUI models folder
DEFAULT_MODEL_PATH = os.path.join(folder_paths.models_dir, "sam3dbody")


class LoadSAM3DBodyModel:
    """
    Prepares SAM 3D Body model configuration.

    Returns a config dict with model paths. The actual model is loaded
    lazily inside the isolated worker when inference runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": DEFAULT_MODEL_PATH,
                    "tooltip": "Path to SAM 3D Body model folder (contains model.ckpt and assets/mhr_model.pt)"
                }),
                "attn_backend": (["sdpa", "flash_attn"], {
                    "default": "sdpa",
                    "tooltip": "Attention backend: sdpa (PyTorch built-in, no extra deps) or flash_attn (requires flash-attn package)"
                }),
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "memory": (["cache_gpu", "cpu_offload", "delete"], {
                    "default": "cpu_offload",
                    "tooltip": "Model memory strategy: cache_gpu = keep on GPU between runs, cpu_offload = move to CPU RAM after use, delete = free after use"
                }),
            },
        }

    RETURN_TYPES = ("SAM3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3DBody"

    def load_model(self, model_path, attn_backend="sdpa", precision="auto", memory="cpu_offload"):
        """Prepare model config (actual loading happens in inference nodes)."""
        import comfy.model_management

        # Auto-detect device
        device = str(comfy.model_management.get_torch_device())

        # Resolve to absolute path
        model_path = os.path.abspath(model_path)

        # Expected file paths
        ckpt_path = os.path.join(model_path, "model.ckpt")
        mhr_path = os.path.join(model_path, "assets", "mhr_model.pt")

        # Check if model exists locally, download if not
        model_exists = os.path.exists(ckpt_path) and os.path.exists(mhr_path)

        if not model_exists:
            try:
                from huggingface_hub import snapshot_download

                log.info(f"Model not found locally. Downloading from HuggingFace...")
                os.makedirs(model_path, exist_ok=True)
                snapshot_download(
                    repo_id="jetjodh/sam-3d-body-dinov3",
                    local_dir=model_path
                )
                log.info(f"Download complete.")

            except Exception as e:
                raise RuntimeError(
                    f"\n[SAM3DBody] Download failed.\n\n"
                    f"Please manually download from:\n"
                    f"  https://huggingface.co/jetjodh/sam-3d-body-dinov3\n\n"
                    f"And place the model files at:\n"
                    f"  {DEFAULT_MODEL_PATH}/\n"
                    f"    +-- model.ckpt          (SAM 3D Body checkpoint)\n"
                    f"    +-- model_config.yaml   (model configuration)\n"
                    f"    \\-- assets/\n"
                    f"        \\-- mhr_model.pt    (Momentum Human Rig model)\n\n"
                    f"Download error: {e}"
                ) from e

        # Return config dict (not the actual model)
        model_config = {
            "model_path": model_path,
            "ckpt_path": ckpt_path,
            "mhr_path": mhr_path,
            "device": device,
            "precision": precision,
            "attn_backend": attn_backend,
            "memory": memory,
        }

        return (model_config,)


# Register node
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DBodyModel": LoadSAM3DBodyModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DBodyModel": "Load SAM 3D Body Model",
}
