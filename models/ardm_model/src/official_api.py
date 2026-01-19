import os
import sys
from pathlib import Path

import numpy as np
import torch

from models.mamba4cast_model.m4c_official.Mamba4Cast.models.armd_model.ardm_official.ARMD.Utils.io_utils import instantiate_from_config


def _add_repo_to_syspath(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def get_official_git_commit(cfg: dict) -> str:
    repo = Path(cfg["official"]["repo_dir"])
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo)).decode().strip()
    except Exception:
        return "unknown"


def train_and_predict_official(
    cfg: dict,
    train_path: str,
    val_path: str,
    test_path: str,
    out_dir: Path,
) -> Path:
    repo = Path(cfg["official"]["repo_dir"]).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make project + official repo importable
    project_root = Path(__file__).resolve().parents[3]
    os.environ["PYTHONPATH"] = f"{project_root}:{repo}:{os.environ.get('PYTHONPATH','')}"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    _add_repo_to_syspath(repo)

    # Official imports
    from engine.solver import Trainer # type: ignore
    from Utils.io_utils import instantiate_from_config # type: ignore

    device = torch.device("cuda" if (cfg.get("device", "cpu") == "cuda" and torch.cuda.is_available()) else "cpu")

    import Models.autoregressive_diffusion.armd as armd_module # type: ignore
    armd_module.pred_len = int(cfg["data"]["pred_len"])

    model = instantiate_from_config(cfg["model"]).to(device)
    model.fast_sampling = True

    from torch.utils.data import DataLoader
    from models.ardm_model.src.ardm_npy_dataset import NpyWindowDataset

    history_len = int(cfg["data"]["history_len"])
    pred_len = int(cfg["data"]["pred_len"])
    window_len = history_len + pred_len
    batch_size = int(cfg["train"]["batch_size"])

    train_ds = NpyWindowDataset(
        name="leukemia",
        data_root=train_path,
        window=window_len,
        period="train",
        output_dir=str(out_dir),
        predict_length=None,
        save2npy=False,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Trainer expects dict with the key "dataloader"
    trainer = Trainer(config=cfg, args=cfg.get("official_args", {}), model=model, dataloader={"dataloader": train_dl})
    trainer.train()

    test_ds = NpyWindowDataset(
        name="leukemia",
        data_root=test_path,
        window=window_len,
        period="test",
        output_dir=str(out_dir),
        predict_length=pred_len,
        save2npy=False,
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    feat_num = test_ds.samples.shape[-1]
    sample, real_future = trainer.sample_forecast(test_dl, shape=[history_len, feat_num])

    pred_path = out_dir / "pred.npy"
    np.save(pred_path, sample.astype(np.float32))
    return pred_path
