from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _add_to_syspath(path: Path) -> None:
    path = path.resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _maybe_clone_official_repo(cfg: dict) -> Path:
    """
    Ensures cfg["official"]["repo_dir"] exists.
    If missing and cfg["official"]["repo_url"] exists -> clones.
    Optional: cfg["official"]["checkout"] (commit/tag/branch).
    """
    official = cfg.get("official", {}) or {}
    repo_dir = Path(official.get("repo_dir", "")).expanduser()
    if repo_dir and repo_dir.exists():
        return repo_dir.resolve()

    repo_url = official.get("repo_url", None)
    if not repo_url:
        raise RuntimeError(
            "Official ARMD repo not found.\n"
            "Either clone it into cfg.official.repo_dir, or set cfg.official.repo_url so I can clone it."
        )

    if not repo_dir:
        raise RuntimeError('cfg["official"]["repo_dir"] is empty/missing.')

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    checkout = official.get("checkout", None)
    if checkout:
        subprocess.run(["git", "checkout", str(checkout)], cwd=str(repo_dir), check=True)

    return repo_dir.resolve()


def get_official_git_commit(cfg: dict) -> str:
    try:
        repo = Path(cfg["official"]["repo_dir"]).resolve()
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo)).decode().strip()
    except Exception:
        return "unknown"


def _select_device() -> torch.device:
    # GPU-first, always.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_and_predict_official(
    cfg: dict,
    train_path: str,
    val_path: str,
    test_path: str,
    out_dir: Path,
) -> Path:
    """
    Trains official ARMD on train.npy windows and writes pred.npy for test.npy windows.
    Expects your adapter to have produced windows of shape (N, window_len, C).
    """
    repo = _maybe_clone_official_repo(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make repo + project importable
    project_root = Path(__file__).resolve().parents[3]
    _add_to_syspath(project_root)
    _add_to_syspath(repo)

    # IMPORTANT: ARMD repo uses absolute imports like "engine.solver", "Utils.io_utils", "Models...."
    # so repo root must be on sys.path (done above).

    # Official imports (from the cloned repo)
    from engine.solver import Trainer  # type: ignore
    from Utils.io_utils import instantiate_from_config  # type: ignore

    # Some official code reads pred_len as a module-level global
    import Models.autoregressive_diffusion.armd as armd_module  # type: ignore
    armd_module.pred_len = int(cfg["data"]["pred_len"])

    device = _select_device()
    print("Using device:", device)

    model = instantiate_from_config(cfg["model"]).to(device)
    # keep this if official supports it
    try:
        model.fast_sampling = True
    except Exception:
        pass

    from torch.utils.data import DataLoader
    from models.ardm_model.src.ardm_npy_dataset import NpyWindowDataset

    history_len = int(cfg["data"]["history_len"])
    pred_len = int(cfg["data"]["pred_len"])
    window_len = history_len + pred_len
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))

    # ---- TRAIN ----
    train_ds = NpyWindowDataset(
        name="leukemia",
        data_root=train_path,
        window=window_len,
        period="train",
        output_dir=str(out_dir),
        predict_length=None,
        save2npy=False,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    trainer = Trainer(
        config=cfg,
        args=cfg.get("official_args", {}),
        model=model,
        dataloader={"dataloader": train_dl},
    )

    trainer.train()

    # ---- TEST / SAMPLE ----
    test_ds = NpyWindowDataset(
        name="leukemia",
        data_root=test_path,
        window=window_len,
        period="test",
        output_dir=str(out_dir),
        predict_length=pred_len,
        save2npy=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feat_num = int(test_ds.samples.shape[-1])

    # Official API expects shape=[history_len, feat_num]
    sample, _real_future = trainer.sample_forecast(test_dl, shape=[history_len, feat_num])

    pred_path = out_dir / "pred.npy"
    np.save(pred_path, np.asarray(sample, dtype=np.float32))
    print("Wrote:", pred_path)
    return pred_path
