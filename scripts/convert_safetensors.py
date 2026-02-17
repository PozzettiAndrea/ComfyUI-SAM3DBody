#!/usr/bin/env python3
"""Convert SAM3DBody checkpoints to safetensors and upload to HuggingFace.

Usage:
    python convert_safetensors.py                     # Convert only (local)
    python convert_safetensors.py --upload             # Convert + upload to HF
    python convert_safetensors.py --upload --token FILE # Upload with token from file

The MHR model (mhr_model.pt) is TorchScript and cannot be converted to safetensors.
It is copied as-is to the output directory and uploaded alongside the safetensors files.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


SOURCE_REPO = "jetjodh/sam-3d-body-dinov3"
TARGET_REPO = "apozz/sam-3d-body-safetensors"


def convert(download_dir: Path, output_dir: Path):
    """Download original checkpoints and convert to safetensors."""
    from huggingface_hub import snapshot_download

    # Download originals
    print(f"Downloading from {SOURCE_REPO}...")
    snapshot_download(SOURCE_REPO, local_dir=str(download_dir), local_dir_use_symlinks=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert model.ckpt → model.safetensors
    ckpt_path = download_dir / "model.ckpt"
    if not ckpt_path.exists():
        sys.exit(f"ERROR: {ckpt_path} not found")

    print(f"Loading {ckpt_path}...")
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        sys.exit(f"ERROR: unexpected checkpoint format: {type(state_dict)}")

    # safetensors requires all values to be tensors
    clean = {}
    skipped = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.contiguous().clone()
        else:
            skipped.append(k)
    if skipped:
        print(f"Skipped {len(skipped)} non-tensor keys: {skipped[:5]}...")

    out_path = output_dir / "model.safetensors"
    print(f"Saving {out_path} ({len(clean)} tensors)...")
    save_file(clean, str(out_path))
    print(f"  Size: {out_path.stat().st_size / 1e9:.2f} GB")

    # Copy MHR model as-is (TorchScript — can't convert)
    mhr_src = download_dir / "assets" / "mhr_model.pt"
    if mhr_src.exists():
        mhr_dst = output_dir / "assets"
        mhr_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(mhr_src), str(mhr_dst / "mhr_model.pt"))
        print(f"Copied MHR model (TorchScript): {mhr_dst / 'mhr_model.pt'}")

    # Copy config
    config_src = download_dir / "model_config.yaml"
    if config_src.exists():
        shutil.copy2(str(config_src), str(output_dir / "model_config.yaml"))
        print(f"Copied config: {output_dir / 'model_config.yaml'}")

    print("Conversion complete!")
    return output_dir


def upload(output_dir: Path, token: str = None):
    """Upload converted files to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Create repo if needed
    try:
        api.create_repo(TARGET_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    print(f"Uploading to {TARGET_REPO}...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=TARGET_REPO,
        repo_type="model",
    )
    print(f"Upload complete: https://huggingface.co/{TARGET_REPO}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-dir", type=Path, default=Path("/tmp/sam3dbody_original"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sam3dbody_safetensors"))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--token", type=str, default=None, help="HF token or path to token file")
    args = parser.parse_args()

    convert(args.download_dir, args.output_dir)

    if args.upload:
        token = args.token
        if token and Path(token).is_file():
            token = Path(token).read_text().strip()
        upload(args.output_dir, token=token)


if __name__ == "__main__":
    main()
