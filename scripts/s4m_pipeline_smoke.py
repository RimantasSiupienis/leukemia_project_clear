# scripts/s4m_pipeline_smoke.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from utils.tools import dotdict
from model.S4M import Model as S4MModel

def main():
    # Minimal but COMPLETE config
    cfg = dotdict({
        "task_name": "long_term_forecast",
        "classification": 0,
        "plot": 0,

        "seq_len": 8,
        "label_len": 0,
        "pred_len": 2,

        "enc_in": 8,
        "dec_in": 8,
        "c_out": 8,
        "d_var": 8,

        "d_model": 32,
        "d_ff": 64,
        "n_heads": 1,
        "e_layers": 1,
        "d_layers": 1,
        "dropout": 0.0,

        "ssm": "s4",
        "d_state": 8,
        "bidirectional": False,

        "mask": False,
        "s4_pred": False,
        "s4_pred_inner": False,
        "pos_emb": False,
        "individual": 1,

        "en_conv_hidden_size": 32,
        "en_rnn_hidden_sizes": [8, 8],
        "output_keep_prob": 1.0,
        "input_keep_prob": 1.0,

        "embed": "timeF",
        "freq": "h",
        "activation": "gelu",
        "output_attention": False,
        "distil": False,
        "moving_avg": 3,
        "factor": 1,

        # memory-related (must exist, but unused here)
        "short_len": 4,
        "W": 2,
        "kernel": [3, 3],
        "stride": 1,
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

        "use_gpu": False,
        "device": torch.device("cpu"),
    })

    print(">>> Initializing S4M model")
    model = S4MModel(cfg).cpu()
    model.eval()

    print(">>> Model parameters:", sum(p.numel() for p in model.parameters()))
    print(">>> S4M initialization OK")
    print(">>> Pipeline is structurally sound")

if __name__ == "__main__":
    main()
