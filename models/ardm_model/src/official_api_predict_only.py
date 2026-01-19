from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")


def predict_only_official(
    repo_dir: Path,
    checkpoint_path: Path,
    test_npy_path: Path,
    out_dir: Path,
    history_len: int,
    pred_len: int,
    batch_size: int = 128,
    device: str = "cpu",
) -> Path:
    """
    Prediction wrapper for the official ARDM codebase.
    Runs:
      official_predict_from_npy.py --repo_dir ... --ckpt ... --windows_npy ... --out_dir ...
    Returns: path to the produced prediction .npy.
    """
    repo_dir = Path(repo_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    test_npy_path = Path(test_npy_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        raise FileNotFoundError(f"official repo_dir not found: {repo_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not test_npy_path.exists():
        raise FileNotFoundError(f"windows npy not found: {test_npy_path}")

    script = Path(__file__).resolve().parent / "official_predict_from_npy.py"
    if not script.exists():
        raise FileNotFoundError(f"missing helper script: {script}")

    cmd = [
        "python",
        str(script),
        "--repo_dir",
        str(repo_dir),
        "--ckpt",
        str(checkpoint_path),
        "--windows_npy",
        str(test_npy_path),
        "--out_dir",
        str(out_dir),
        "--device",
        str(device),
        "--history_len",
        str(int(history_len)),
        "--pred_len",
        str(int(pred_len)),
        "--batch_size",
        str(int(batch_size)),
    ]
    _run(cmd)

    for name in ("pred.npy", "y_pred.npy"):
        p = out_dir / name
        if p.exists():
            return p

    raise RuntimeError(f"predict-only ran but no pred file found in: {out_dir}")
