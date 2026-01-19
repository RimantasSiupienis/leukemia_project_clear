from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _add_repo_to_syspath(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def _load_windows(path: Path) -> np.ndarray:
    x = np.load(path)
    if x.ndim != 3:
        raise ValueError(f"windows_npy must be 3D [N,T,C]. Got: {x.shape}")
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    return x.astype(np.float32, copy=False)


def _extract_history(windows: np.ndarray, history_len: int) -> np.ndarray:
    if windows.shape[1] < history_len:
        raise ValueError(f"windows T={windows.shape[1]} < history_len={history_len}")
    return windows[:, :history_len, :]


def _load_state_dict(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict", "net", "network"):
            sd = ckpt.get(k)
            if isinstance(sd, dict):
                model.load_state_dict(sd, strict=False)
                return

        # raw state_dict fallback
        # many tensor values
        tensor_keys = [k for k, v in ckpt.items() if torch.is_tensor(v)]
        if len(tensor_keys) >= 10:
            model.load_state_dict(ckpt, strict=False)
            return

    raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")


def _ensure_pred_shape(y: np.ndarray, pred_len: int, c: int) -> np.ndarray:
    """
    Return [B, pred_len, C]
    Accepts:
      - [B, pred_len, C]
      - [B, C, pred_len]
      - [B, pred_len*C]
    """
    if y.ndim == 3:
        if y.shape[1:] == (pred_len, c):
            return y
        if y.shape[1:] == (c, pred_len):
            return np.transpose(y, (0, 2, 1))
        raise ValueError(f"Unexpected 3D output shape: {y.shape}")

    if y.ndim == 2:
        if y.shape[1] == pred_len * c:
            return y.reshape(y.shape[0], pred_len, c)
        raise ValueError(f"Unexpected 2D output shape: {y.shape}")

    raise ValueError(f"Unexpected output ndim={y.ndim}, shape={y.shape}")


def _predict_batch(model: torch.nn.Module, x_btc: torch.Tensor, pred_len: int) -> torch.Tensor:
    """
    Predict only call without Trainer(debugging).
    Trying a small set of  entrypoints.
    """
    # Preferred: ARMD class often exposes generate_mts(x)
    if hasattr(model, "generate_mts") and callable(getattr(model, "generate_mts")):
        return model.generate_mts(x_btc)  # type: ignore[attr-defined]

    # Common diffusion-forecast names(debugging)
    if hasattr(model, "sample_forecast") and callable(getattr(model, "sample_forecast")):
        return model.sample_forecast(x_btc, pred_len=pred_len)  # type: ignore[misc]

    if hasattr(model, "sample") and callable(getattr(model, "sample")):
        # Some repos use [x, pred_len], some just [x]
        try:
            return model.sample(x_btc, pred_len)  # type: ignore[misc]
        except TypeError:
            return model.sample(x_btc)  # type: ignore[misc]

    # Last hope
    return model(x_btc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", required=True, help="Path to official ARDM repo root")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    ap.add_argument("--windows_npy", required=True, help="Input windows (N, history+pred, C), future zeros")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--history_len", type=int, required=True)
    ap.add_argument("--pred_len", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    windows_path = Path(args.windows_npy).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _add_repo_to_syspath(repo_dir)

    # some ARMD repos use a pred_len constant from the module
    import Models.autoregressive_diffusion.armd as armd_module  # type: ignore
    armd_module.pred_len = int(args.pred_len) # matching CLI

    from Models.autoregressive_diffusion.armd import ARMD  # type: ignore

    windows = _load_windows(windows_path)
    history = _extract_history(windows, int(args.history_len))
    n, _, c = history.shape

    device = torch.device(args.device)
    model = ARMD(
        seq_length=int(args.history_len),
        feature_size=int(c),
        timesteps=int(args.pred_len),
        sampling_timesteps=1,
        loss_type="l1",
        beta_schedule="cosine",
        w_grad=True,
    ).to(device)
    model.eval()

    _load_state_dict(model, ckpt_path, device)

    pred = np.zeros((n, int(args.pred_len), int(c)), dtype=np.float32)

    bs = int(args.batch_size)
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(n, s + bs)
            x = torch.from_numpy(history[s:e]).to(device)  # (B,T,C)
            y = _predict_batch(model, x, pred_len=int(args.pred_len))

            if isinstance(y, (tuple, list)):
                y = y[0]
            if not torch.is_tensor(y):
                raise RuntimeError(f"Model returned non-tensor output type: {type(y)}")

            y_np = y.detach().cpu().numpy()
            y_np = _ensure_pred_shape(y_np, pred_len=int(args.pred_len), c=int(c))
            if y_np.shape[0] != (e - s):
                raise RuntimeError(f"Batch N mismatch: got {y_np.shape[0]} expected {e - s}")

            pred[s:e] = y_np.astype(np.float32, copy=False)

    out_path = out_dir / "pred.npy"
    np.save(out_path, pred)
    print(f"[predict-only] Wrote: {out_path}  shape={pred.shape}")


if __name__ == "__main__":
    main()
