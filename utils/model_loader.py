# SPDX-License-Identifier: GPL-3.0-or-later
# Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra
# Copyright (C) 2025 ComfyUI-CADabra Contributors

"""
Model loader utilities for SECAD-Net
Downloads and caches pretrained models from Google Drive
"""

from pathlib import Path
from typing import Optional

# SECAD-Net model URLs and metadata
SECADNET_MODELS = {
    "secadnet_abc": {
        "url": "https://drive.google.com/uc?export=download&id=1hLJ6JYI1eXoFG-Qc13Bx3o1s-nt9yHkH",
        "filename": "ModelParameters/best.pth",
        "description": "SECAD-Net pretrained on ABC dataset (4 primitives)",
        "folder_id": "1hLJ6JYI1eXoFG-Qc13Bx3o1s-nt9yHkH",
    },
}


def get_secadnet_models_dir() -> Path:
    """
    Get the models directory for SECAD-Net models.
    Creates ComfyUI/models/cadrecon/secadnet/ if it doesn't exist.
    """
    current_dir = Path(__file__).parent.parent  # ComfyUI-SECADNET/
    comfyui_dir = current_dir.parent.parent  # ComfyUI/
    models_dir = comfyui_dir / "models" / "cadrecon" / "secadnet"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_secadnet_model_path(model_name: str) -> Optional[Path]:
    """Get the path to a specific SECAD-Net model."""
    if model_name not in SECADNET_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(SECADNET_MODELS.keys())}")

    models_dir = get_secadnet_models_dir()
    filename = SECADNET_MODELS[model_name]["filename"]
    model_path = models_dir / filename

    if model_path.exists():
        return model_path
    return None


def download_secadnet_model(model_name: str, force_download: bool = False) -> Optional[Path]:
    """
    Download a SECAD-Net model from Google Drive if not already cached.

    Note: Google Drive downloads require gdown for large files.
    Install with: pip install gdown
    """
    if model_name not in SECADNET_MODELS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"   Available models: {list(SECADNET_MODELS.keys())}")
        return None

    model_info = SECADNET_MODELS[model_name]
    models_dir = get_secadnet_models_dir()
    model_path = models_dir / model_info["filename"]

    if model_path.exists() and not force_download:
        print(f"[OK] Model already downloaded: {model_path}")
        return model_path

    print(f"[SECAD-Net] Model not found locally, downloading {model_name}...")
    print(f"   Description: {model_info['description']}")

    try:
        import gdown
        folder_id = model_info.get("folder_id")
        if folder_id:
            print(f"[SECAD-Net] Downloading from Google Drive folder...")
            output_dir = str(models_dir)
            gdown.download_folder(id=folder_id, output=output_dir, quiet=False)

            pth_files = list(models_dir.glob("**/*.pth"))
            if pth_files:
                best_pth = None
                for pth in pth_files:
                    if pth.name == "best.pth":
                        best_pth = pth
                        break
                if best_pth is None:
                    best_pth = pth_files[0]

                model_path.parent.mkdir(parents=True, exist_ok=True)
                if best_pth != model_path:
                    import shutil
                    shutil.copy2(best_pth, model_path)
                print(f"[OK] Model downloaded: {model_path}")
                return model_path
            else:
                print("[WARN] gdown completed but no .pth files found")
    except ImportError:
        print("[WARN] gdown not installed. Install with: pip install gdown")
        print("[WARN] Falling back to direct download (may fail for large files)...")

    print(f"\n[WARN] Automatic download failed. Please download manually:")
    print(f"   1. Visit: https://drive.google.com/drive/folders/{model_info.get('folder_id', '')}")
    print(f"   2. Download the .pth file")
    print(f"   3. Save to: {model_path}")
    return None
