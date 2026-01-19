# scripts/s4m_forward_only_smoke.py
import os

# hard safety limits (no GPU, no W&B, no BLAS storms)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from utils.tools import dotdict
from model.S4M import Model as S4MModel


def main():
    cfg = dotdict({
        # task flags
        "task_name": "long_term_forecast",
        "classification": 0,
        "plot": 0,

        # sequence sizes (TINY)
        "seq_len": 8,
        "label_len": 0,
        "pred_len": 2,
        "short_len": 4,   # MUST be >= W

        # dimensions (TINY)
        "enc_in": 8,
        "dec_in": 8,
        "c_out": 8,
        "d_var": 8,
        "d_model": 32,
        "d_ff": 64,

        # conv/attn
        "W": 2,
        "n": 1,
        "n_heads": 1,
        "e_layers": 1,
        "d_layers": 1,
        "factor": 1,
        "dropout": 0.0,

        # S4 specifics (TINY)
        "ssm": "s4",
        "d_state": 8,
        "bidirectional": False,

        # flags
        "mask": False,
        "s4_pred": False,
        "s4_pred_inner": False,
        "pos_emb": False,
        "individual": 1,
        "distil": False,
        "output_attention": False,

        # encoder / memnet
        "en_conv_hidden_size": 32,
        "en_rnn_hidden_sizes": [8, 8],
        "input_keep_prob": 1.0,
        "output_keep_prob": 1.0,

        "memnet": 1,
        "mem_type": 1,
        "M": 2,
        "K": 2,
        "topK": 2,
        "topM": 10,
        "max_k": 100,
        "memory_size": 32,
        "per_mem_size": 8,
        "mem_repeat": 1,
        "momentum": 0.99,
        "tau": 32,
        "no_renew": False,
        "pretrain": False,

        # misc
        "embed": "timeF",
        "freq": "h",
        "activation": "gelu",
        "moving_avg": 25,
        "weight": 1.0,

        # device
        "use_gpu": False,
        "device": torch.device("cpu"),
    })

    model = S4MModel(cfg).cpu().eval()

    B, L, D = 1, cfg.seq_len, cfg.d_var

    # --- Inputs must match S4M.forward signature expectations ---
    # seq_x: [B, L, d_var] float
    seq_x = torch.randn(B, L, D, dtype=torch.float32)

    # seq_x_mask: [B, L, d_var] float mask in {0,1}
    seq_x_mask = torch.ones(B, L, D, dtype=torch.float32)

    # max_idx/min_idx MUST be float and [B, L, d_var] because they do mean/std over dim=1
    # and then pass through Linear(d_var -> d_var)
    max_idx = torch.randn(B, L, D, dtype=torch.float32)
    min_idx = torch.randn(B, L, D, dtype=torch.float32)

    # max_value/min_value used elementwise with seq_x/seq_x_mask -> keep [B, L, d_var] float
    max_value = torch.randn(B, L, D, dtype=torch.float32)
    min_value = torch.randn(B, L, D, dtype=torch.float32)
    # -----------------------------------------------------------

    with torch.no_grad():
        y = model(seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value)

    print("\n✅ S4M forward pass OK")
    print("  out.shape =", tuple(y.shape) if hasattr(y, "shape") else type(y))


if __name__ == "__main__":
    main()
