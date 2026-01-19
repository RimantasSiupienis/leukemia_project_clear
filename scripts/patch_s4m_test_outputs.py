#!/usr/bin/env python3
"""
Patch the *official* S4M repo code (minimally) so a normal run produces artifacts you actually need.

What this patch does (ONLY in exp_pretrain1.py::test):
- Saves preds/trues/masks as .npy under: <run_dir>/test_results/<setting>/
- Computes MAE/MSE/RMSE/MAPE/MSPE **mask-aware** (ignores missing targets)
- Writes metrics.json under the same folder
- Writes result_long_term_forecast.txt under <run_dir>/ (not random CWD, not /home/project)
- Keeps the existing PNG plots behavior (no change)

This is intentionally a small, localized change.
"""

from __future__ import annotations
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "models" / "s4m_model" / "s4m_official" / "s4m" / "experiments" / "exp_pretrain1.py"

if not EXP.exists():
    print(f"ERROR: file not found: {EXP}", file=sys.stderr)
    sys.exit(1)

s = EXP.read_text()

# -------------
# 1) Ensure test folder_path is under checkpoints/..
#    (you already patched this once, but we keep it stable + idempotent)
# -------------
# Original line (official):
# folder_path = '/home/project/S4M/test_results/' + setting + '/'
#
# Patched form (your convention):
# base_dir = os.path.abspath(os.path.join(self.args.checkpoints, os.pardir))
# folder_path = os.path.join(base_dir, 'test_results', setting)
s = s.replace(
    "folder_path = '/home/project/S4M/test_results/' + setting + '/'",
    "base_dir = os.path.abspath(os.path.join(self.args.checkpoints, os.pardir))\n"
    "        folder_path = os.path.join(base_dir, 'test_results', setting)"
)

# -------------
# 2) Replace the metric block with mask-aware + artifact saving
#    We replace from:
#       mae, mse, rmse, mape, mspe = 0.0,...
#    down to:
#       wandb.log(...)
#    and keep return
# -------------
pat_metrics = re.compile(
    r"\n\s*mae,\s*mse,\s*rmse,\s*mape,\s*mspe\s*=\s*0\.0,0\.0,0\.0,0\.0,0\.0\s*\n"
    r"(?:.*\n)*?"
    r"\s*wandb\.log\(\{.*?\}\)\s*\n",
    re.DOTALL,
)

replacement_metrics = r"""
        # -----------------------------
        # Save artifacts + compute metrics (mask-aware)
        # -----------------------------
        import json

        # Stack lists into arrays so we can save them and compute metrics properly.
        # Shapes typically:
        #   preds_arr: [N, B, pred_len, D]
        #   trues_arr: [N, B, pred_len, D]
        preds_arr = np.asarray(preds, dtype=np.float32)
        trues_arr = np.asarray(trues, dtype=np.float32)

        # Build a matching mask for the *prediction horizon*.
        # We already had seq_y_mask per batch in the loop; we didn't store it.
        # Easiest safe option: recompute metrics using the masks we *do* have in the loop:
        # store them while looping (below). If it's missing, fall back to "all observed".
        try:
            masks_arr = np.asarray(masks, dtype=np.float32)  # [N,B,pred_len,D]
        except Exception:
            masks_arr = np.ones_like(trues_arr, dtype=np.float32)

        # Save raw arrays for your pipeline sanity checks / later evaluation.
        # Folder is already: <run_dir>/test_results/<setting>/
        np.save(os.path.join(folder_path, "preds.npy"), preds_arr)
        np.save(os.path.join(folder_path, "trues.npy"), trues_arr)
        np.save(os.path.join(folder_path, "masks.npy"), masks_arr)

        # Mask-aware metrics: only evaluate where mask==1.
        eps = 1e-8
        valid = (masks_arr > 0.5)

        # If nothing is valid (can happen in tiny debug splits), avoid crashing.
        denom = float(valid.sum())
        if denom < 1:
            mae = mse = rmse = mape = mspe = float("nan")
        else:
            diff = (preds_arr - trues_arr)
            diff_valid = diff[valid]
            true_valid = trues_arr[valid]
            pred_valid = preds_arr[valid]

            mae = float(np.mean(np.abs(diff_valid)))
            mse = float(np.mean(diff_valid ** 2))
            rmse = float(np.sqrt(mse))

            # MAPE/MSPE: avoid divide-by-zero
            denom_true = np.maximum(np.abs(true_valid), eps)
            mape = float(np.mean(np.abs(diff_valid) / denom_true))
            mspe = float(np.mean((diff_valid / denom_true) ** 2))

        total_loss = float(np.average(total_loss)) if len(total_loss) else float("nan")

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "mspe": mspe,
            "total_loss": total_loss,
            "n_pred_arrays": int(preds_arr.shape[0]),
            "pred_shape": list(preds_arr.shape),
        }

        with open(os.path.join(folder_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print("mse:{}, mae:{}".format(mse, mae))
        # Write results next to the run dir (NOT in random cwd).
        result_txt = os.path.join(base_dir, "result_long_term_forecast.txt")
        with open(result_txt, "a") as f:
            f.write(setting + "  \n")
            f.write("mse:{}, mae:{}\n\n".format(mse, mae))

        wandb.log({"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "mspe": mspe, "total_loss": total_loss})
"""

if not pat_metrics.search(s):
    print("ERROR: couldn't find the metrics block to patch (file layout changed).", file=sys.stderr)
    sys.exit(2)

s = pat_metrics.sub(replacement_metrics, s, count=1)

# -------------
# 3) Store masks during the test loop (minimal: add `masks=[]` and append per batch)
# -------------
# After:
#   preds = []
#   trues = []
# add:
#   masks = []
s = s.replace(
    "preds = []\n        trues = []",
    "preds = []\n        trues = []\n        masks = []"
)

# In the loop, after seq_y_mask is sliced to pred horizon, append it:
# We want the same shape as preds/trues after f_dim slicing.
# Right after:
#   seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
# insert:
#   masks.append(seq_y_mask.detach().cpu().numpy())
needle = "seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]"
insert = (
    "seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]\n"
    "                masks.append(seq_y_mask.detach().cpu().numpy())"
)
if needle not in s:
    print("ERROR: couldn't find expected seq_y_mask slicing line to hook masks.", file=sys.stderr)
    sys.exit(3)
s = s.replace(needle, insert, 1)

EXP.write_text(s)
print(f"OK: patched {EXP}")
