import logging
import os
import folder_paths
import comfy.model_management as mm

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
                    "tooltip": "Path to SAM 3D Body model folder (contains model.safetensors and assets/mhr_model.pt)"
                }),
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
            },
        }

    RETURN_TYPES = ("SAM3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3DBody"

    REPO_ID = "apozz/sam-3d-body-safetensors"

    def load_model(self, model_path, precision="auto"):
        """Prepare model config (actual loading happens in inference nodes)."""
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

        # Resolve to absolute path
        model_path = os.path.abspath(model_path)

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
                    repo_id=self.REPO_ID,
                    local_dir=model_path
                )
                log.info(f"Download complete.")

            except Exception as e:
                raise RuntimeError(
                    f"\n[SAM3DBody] Download failed.\n\n"
                    f"Please manually download from:\n"
                    f"  https://huggingface.co/{self.REPO_ID}\n\n"
                    f"And place the model files at:\n"
                    f"  {DEFAULT_MODEL_PATH}/\n"
                    f"    +-- model.safetensors   (SAM 3D Body weights)\n"
                    f"    +-- model_config.yaml   (model configuration)\n"
                    f"    \\-- assets/\n"
                    f"        \\-- mhr_model.pt    (Momentum Human Rig model)\n\n"
                    f"Download error: {e}"
                ) from e

        # Return config dict (not the actual model — loading happens in inference nodes)
        model_config = {
            "model_path": model_path,
            "ckpt_path": model_file,
            "mhr_path": mhr_path,
            "precision": precision,
        }

        return (model_config,)


# Register node
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DBodyModel": LoadSAM3DBodyModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DBodyModel": "Load SAM 3D Body Model",
}
