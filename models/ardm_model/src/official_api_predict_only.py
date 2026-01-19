from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _add_repo_to_syspath(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def _infer_device(device: str) -> torch.device:
    want = (device or "cpu").lower()
    if want in ["cuda", "gpu"] and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_only_official(
    repo_dir: Path,
    checkpoint_path: Path,
    test_npy_path: Path,
    out_dir: Path,
    history_len: int,
    pred_len: int,
    batch_size: int,
    device: str = "cuda",
) -> Path:
    """
    Predict ONLY using a trained checkpoint, with the OFFICIAL ARMD repo.

    Inputs:
      - repo_dir: path to cloned official ARMD repo root (contains engine/, Models/, Utils/, main.py)
      - checkpoint_path: checkpoint file produced by official training (whatever their Trainer expects)
      - test_npy_path: npy of shape [N, window_len, C] where window_len = history_len + pred_len
                       IMPORTANT: future region should be masked/zeroed (your dataset class does this too)
      - out_dir: where to write pred.npy
    Output:
      - path to pred.npy
    """
    repo_dir = Path(repo_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    test_npy_path = Path(test_npy_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        raise FileNotFoundError(f"repo_dir not found: {repo_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_path not found: {checkpoint_path}")
    if not test_npy_path.exists():
        raise FileNotFoundError(f"test_npy_path not found: {test_npy_path}")

    # Ensure both project + official repo are importable
    project_root = Path(__file__).resolve().parents[3]
    os.environ["PYTHONPATH"] = f"{project_root}:{repo_dir}:{os.environ.get('PYTHONPATH','')}"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    _add_repo_to_syspath(repo_dir)

    # Official imports
    from engine.solver import Trainer  # type: ignore
    from Utils.io_utils import instantiate_from_config  # type: ignore

    torch_device = _infer_device(device)
    print(f"[predict_only] device={torch_device} cuda_available={torch.cuda.is_available()}")

    window_len = int(history_len) + int(pred_len)

    # Your dataset wrapper (independent of official repo)
    from torch.utils.data import DataLoader
    from models.ardm_model.src.ardm_npy_dataset import NpyWindowDataset

    test_ds = NpyWindowDataset(
        name="leukemia_ctgan",
        data_root=str(test_npy_path),
        window=window_len,
        period="test",
        output_dir=str(out_dir),
        predict_length=int(pred_len),
        save2npy=False,
    )
    test_dl = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, drop_last=False, num_workers=0)

    # Build model from config stored in checkpoint OR from cfg-like dict saved with checkpoint.


    # The ARMD code sometimes requires setting global pred_len
    try:
        import Models.autoregressive_diffusion.armd as armd_module  # type: ignore
        armd_module.pred_len = int(pred_len)
    except Exception:
        pass

    # Create a minimal "cfg" dict that Trainer expects. We'll load actual weights from checkpoint.
    # IMPORTANT: this must match the training cfg used for that checkpoint.
    # If the official checkpoint includes config internally, Trainer will use it.
    cfg = {
        "device": "cuda" if torch_device.type == "cuda" else "cpu",
        "data": {"pred_len": int(pred_len), "history_len": int(history_len)},
    }

    # Try to load checkpoint payload to recover model config if present
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model_cfg = None
    if isinstance(ckpt, dict):
        # common patterns
        if "model_config" in ckpt and isinstance(ckpt["model_config"], dict):
            model_cfg = ckpt["model_config"]
        elif "config" in ckpt and isinstance(ckpt["config"], dict):
            # sometimes the whole config is saved
            model_cfg = ckpt.get("config", {}).get("model", None)

    if model_cfg is None:
        raise RuntimeError(
            "Could not find model config inside checkpoint. "
            "Easiest fix: save your training config_resolved.yaml and pass it in here "
            "so we can instantiate model = instantiate_from_config(cfg['model'])."
        )

    model = instantiate_from_config(model_cfg).to(torch_device)
    model.fast_sampling = True

    trainer = Trainer(config=cfg, args={}, model=model, dataloader={"dataloader": test_dl})

    # Load weights (common patterns)
    loaded = False
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                model.load_state_dict(ckpt[key], strict=False)
                loaded = True
                break
    if not loaded:
        # try direct load (if checkpoint is raw state_dict)
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)
            loaded = True

    if not loaded:
        raise RuntimeError("Failed to load checkpoint weights into model.")

    feat_num = test_ds.samples.shape[-1]
    sample, _ = trainer.sample_forecast(test_dl, shape=[int(history_len), int(feat_num)])

    pred_path = out_dir / "pred.npy"
    np.save(pred_path, sample.astype(np.float32))
    print(f"[predict_only] Wrote: {pred_path}")
    return pred_path
