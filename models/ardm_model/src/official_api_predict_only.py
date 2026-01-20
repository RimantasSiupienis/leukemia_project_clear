from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np
import torch
import yaml


def _add_repo_to_syspath(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def _infer_device(device: str) -> torch.device:
    want = (device or "cpu").lower()
    if want in ["cuda", "gpu"] and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"YAML did not parse to dict: {path}")
    return obj


def _extract_model_cfg_from_checkpoint(ckpt: Any) -> Optional[dict]:
    if not isinstance(ckpt, dict):
        return None
    if "model_config" in ckpt and isinstance(ckpt["model_config"], dict):
        return ckpt["model_config"]
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ckpt["config"]
        if isinstance(cfg.get("model"), dict):
            return cfg["model"]
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        cfg = ckpt["cfg"]
        if isinstance(cfg.get("model"), dict):
            return cfg["model"]
    return None


def _extract_state_dict_from_checkpoint(ckpt: Any) -> Optional[Dict[str, Any]]:
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "network"]:
            v = ckpt.get(key)
            if isinstance(v, dict):
                return v
        return ckpt  # sometimes ckpt is already a state_dict
    return None


def predict_only_official(
    repo_dir: Path,
    checkpoint_path: Path,
    test_npy_path: Path,
    out_dir: Path,
    history_len: int,
    pred_len: int,
    batch_size: int,
    device: str = "cuda",
    config_yaml_path: Optional[Path] = None,
) -> Path:
    """
    Predict ONLY using a trained checkpoint, with the OFFICIAL ARMD repo.

    IMPORTANT: The official Trainer requires a full config that includes 'solver'.
    Therefore you MUST provide config_yaml_path (your training config_resolved.yaml).
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

    if config_yaml_path is None:
        raise RuntimeError(
            "config_yaml_path is required for predict-only because ARMD Trainer expects config['solver'] etc."
        )
    config_yaml_path = Path(config_yaml_path).resolve()
    if not config_yaml_path.exists():
        raise FileNotFoundError(f"config_yaml_path not found: {config_yaml_path}")

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

    # Dataset wrapper (yours)
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

    # Some ARMD code uses global pred_len
    try:
        import Models.autoregressive_diffusion.armd as armd_module  # type: ignore
        armd_module.pred_len = int(pred_len)
    except Exception:
        pass

    # Load checkpoint
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    # Prefer model config from checkpoint; fallback to YAML
    model_cfg = _extract_model_cfg_from_checkpoint(ckpt)
    if model_cfg is None:
        cfg_yml = _load_yaml(config_yaml_path)
        if "model" not in cfg_yml or not isinstance(cfg_yml["model"], dict):
            raise RuntimeError(f"config_yaml_path must contain top-level 'model:' dict: {config_yaml_path}")
        model_cfg = cfg_yml["model"]

    # Build model + load weights
    model = instantiate_from_config(model_cfg).to(torch_device)
    model.fast_sampling = True

    state_dict = _extract_state_dict_from_checkpoint(ckpt)
    if state_dict is None:
        raise RuntimeError("Failed to extract a state_dict from checkpoint payload.")
    model.load_state_dict(state_dict, strict=False)

    # Use FULL config for Trainer (must include solver, etc.)
    cfg_full = _load_yaml(config_yaml_path)

    # Override minimal runtime things (safe)
    cfg_full["device"] = "cuda" if torch_device.type == "cuda" else "cpu"
    cfg_full.setdefault("data", {})
    cfg_full["data"]["pred_len"] = int(pred_len)
    cfg_full["data"]["history_len"] = int(history_len)

    # Make sure train batch size doesn't accidentally matter; we can keep it, but ensure it's present
    cfg_full.setdefault("train", {})
    cfg_full["train"].setdefault("batch_size", int(batch_size))

    trainer = Trainer(config=cfg_full, args={}, model=model, dataloader={"dataloader": test_dl})

    feat_num = test_ds.samples.shape[-1]
    sample, _ = trainer.sample_forecast(test_dl, shape=[int(history_len), int(feat_num)])

    pred_path = out_dir / "pred.npy"
    np.save(pred_path, sample.astype(np.float32))
    print(f"[predict_only] Wrote: {pred_path}")
    return pred_path
