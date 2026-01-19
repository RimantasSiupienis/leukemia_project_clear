import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyWindowDataset(Dataset):
    """
    Loads windowed data from .npy of shape (N, window_len, C)
    For period='test':
      - returns (x_masked, mask)
      - x_masked has future region zeroed to prevent leakage
      - mask marks observed=True / future=False
    """

    def __init__(
        self,
        name: str,
        data_root: str,
        window: int,
        period: str = "train",
        output_dir: str = "./OUTPUT",
        predict_length: Optional[int] = None,
        missing_ratio: Optional[float] = None,
        seed: int = 123,
        save2npy: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert period in ["train", "test"], "period must be 'train' or 'test'"

        self.name = name
        self.data_root = data_root
        self.window = int(window)
        self.period = period
        self.pred_len = predict_length

        self.dir = os.path.join(output_dir, "samples")
        os.makedirs(self.dir, exist_ok=True)

        arr = np.load(data_root)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array (N, T, C). Got {arr.shape}")
        if arr.shape[1] != self.window:
            raise ValueError(f"Expected window_len={self.window}, got {arr.shape[1]}")

        self.samples = arr.astype(np.float32, copy=False)

        # Putting this here cuz upstream code expects scaler attribute
        self.scaler = None

        if self.period == "test":
            if self.pred_len is None:
                raise ValueError("For period='test', predict_length must be set.")

            masks = np.ones(self.samples.shape, dtype=bool)
            masks[:, -self.pred_len :, :] = False
            self.masking = masks

            if save2npy:
                np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        self.sample_num = self.samples.shape[0]

    def __len__(self) -> int:
        return self.sample_num

    def __getitem__(self, ind: int):
        x = torch.from_numpy(self.samples[ind]).float()

        if self.period == "test":
            m = torch.from_numpy(self.masking[ind])
            # prevents leakage, blank out future region in x
            x = x.clone()
            x[-self.pred_len :, :] = 0.0
            return x, m

        return x
