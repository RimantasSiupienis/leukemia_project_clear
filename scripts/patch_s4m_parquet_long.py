#!/usr/bin/env python3
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
S4M_PKG = REPO_ROOT / "models" / "s4m_model" / "s4m_official" / "s4m"
FACTORY = S4M_PKG / "data_provider" / "data_factory.py"
LOADER = S4M_PKG / "data_provider" / "data_loader.py"

DATASET_CLASS_NAME = "Dataset_ParquetLong"

DATASET_CODE = r'''
class Dataset_ParquetLong(Dataset):
    """
    Long-format parquet adapter for S4M official training loop.

    Input parquet must be long:
      - patient_id column
      - t column (numeric time index)
      - feature columns (numeric), may contain NaNs

    Returns exactly the 10-tuple expected by experiments/exp_pretrain1.py:
      (seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y, max_idx, min_idx, max_value, min_value)
    """
    def __init__(self, root_path, args, flag='train', size=None,
                 features='M', data_path='physionet2012_long.parquet',
                 target='OT', scale=True, timeenc=0, freq='h', mask=None, pad='zero'):
        assert flag in ['train', 'val', 'test']
        assert size is not None, "size must be [seq_len, label_len, pred_len]"
        self.args = args
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.mask = bool(mask)
        self.pad = pad

        self.seq_len = size[0]
        self.label_len = size[1]  # unused by exp_pretrain1, but kept for consistency
        self.pred_len = size[2]

        # fixed column names (pipeline canonical)
        self.id_col = "patient_id"
        self.time_col = "t"

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        parquet_path = os.path.join(self.root_path, self.data_path)
        df = pd.read_parquet(parquet_path)

        if self.id_col not in df.columns or self.time_col not in df.columns:
            raise ValueError(f"Parquet must contain columns '{self.id_col}' and '{self.time_col}'. Got: {list(df.columns)}")

        # select feature columns: numeric, excluding id/time
        feat_cols = [c for c in df.columns if c not in (self.id_col, self.time_col)]
        feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(feat_cols) == 0:
            raise ValueError("No numeric feature columns found (excluding patient_id,t).")

        df = df.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

        # Build patient series
        series = []
        masks = []
        for _, g in df.groupby(self.id_col, sort=False):
            g = g.sort_values(self.time_col)
            x_raw = g[feat_cols].to_numpy(dtype=np.float32)  # may include NaNs
            if x_raw.shape[0] < (self.seq_len + self.pred_len):
                continue

            if self.mask:
                m = (~np.isnan(x_raw)).astype(np.float32)
            else:
                m = np.ones_like(x_raw, dtype=np.float32)

            # model input cannot rely on NaNs -> fill with 0, mask tells truth
            x_in = np.nan_to_num(x_raw, nan=0.0) if self.pad == 'zero' else x_raw.copy()

            series.append((x_in, x_raw, m))

        if len(series) == 0:
            raise ValueError("No patients have length >= seq_len + pred_len. Cannot build windows.")

        # patient split (no leakage): 70/10/20 by patient order
        n_pat = len(series)
        n_train = int(n_pat * 0.7)
        n_test = int(n_pat * 0.2)
        n_val = n_pat - n_train - n_test

        train_p = set(range(0, n_train))
        val_p = set(range(n_train, n_train + n_val))
        test_p = set(range(n_train + n_val, n_pat))

        if self.flag == 'train':
            keep = train_p
        elif self.flag == 'val':
            keep = val_p
        else:
            keep = test_p

        # fit scaler on TRAIN patients only (on input-filled values)
        self.scaler = StandardScaler()
        if self.scale:
            train_vals = np.concatenate([series[p][0] for p in sorted(train_p)], axis=0)
            self.scaler.fit(train_vals)

        # Store only kept patients, but keep a mapping (old index -> new index)
        self.patients = []
        old_to_new = {}
        for old_idx in sorted(list(keep)):
            x_in, x_raw, m = series[old_idx]
            if self.scale:
                x_in = self.scaler.transform(x_in)
                # keep "true" aligned to scaled space too (since model/loss compares in same space)
                x_raw = np.nan_to_num(x_raw, nan=0.0)
                x_raw = self.scaler.transform(x_raw)
            else:
                x_raw = np.nan_to_num(x_raw, nan=0.0)

            new_idx = len(self.patients)
            old_to_new[old_idx] = new_idx
            self.patients.append((x_in.astype(np.float32), x_raw.astype(np.float32), m.astype(np.float32)))

        # Build window index: (patient_idx, start)
        self.index = []
        for p_idx, (x_in, _, _) in enumerate(self.patients):
            L = x_in.shape[0]
            max_start = L - self.seq_len - self.pred_len + 1
            for s in range(max_start):
                self.index.append((p_idx, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        p, s_begin = self.index[idx]
        x_in, x_true, m = self.patients[p]

        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = x_in[s_begin:s_end]               # (seq_len, D)
        seq_y = x_in[r_begin:r_end]               # (pred_len, D)

        # masks
        seq_x_mask = m[s_begin:s_end]
        seq_y_mask = m[r_begin:r_end]

        # "true" arrays (used by some models/metrics; exp_pretrain1 includes them in loader tuple)
        seq_true_x = x_true[s_begin:s_end]
        seq_true_y = x_true[r_begin:r_end]

        # max/min positional features exactly like Dataset_Custom4
        max_idx = np.argmax(seq_x, axis=0)
        min_idx = np.argmin(seq_x, axis=0)
        idx_grid = np.expand_dims(np.arange(seq_x.shape[0]), axis=1).repeat(seq_x.shape[1], axis=1)
        max_idx = np.abs(max_idx - idx_grid).astype(np.float32)
        min_idx = np.abs(min_idx - idx_grid).astype(np.float32)

        max_value = np.expand_dims(np.max(seq_x, axis=0), axis=0).repeat(seq_x.shape[0], axis=0).astype(np.float32)
        min_value = np.expand_dims(np.min(seq_x, axis=0), axis=0).repeat(seq_x.shape[0], axis=0).astype(np.float32)

        # special case in official loader: BiaTCGNet expects an extra dim
        if getattr(self.args, "model", "") == 'BiaTCGNet':
            seq_x = np.expand_dims(seq_x, axis=-1)
            seq_x_mask = np.expand_dims(seq_x_mask, axis=-1)

        return seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y, max_idx, min_idx, max_value, min_value

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
'''

def patch_factory():
    s = FACTORY.read_text()

    # Ensure Dataset_ParquetLong is imported from data_loader
    if DATASET_CLASS_NAME not in s:
        # Try to extend the existing import line that pulls many Dataset_* symbols
        s = re.sub(
            r"(from data_provider\.data_loader import [^\n]+)",
            r"\1," + DATASET_CLASS_NAME,
            s,
            count=1
        )

    # Register in data_dict
    if "'parquet_long'" not in s:
        s = re.sub(
            r"(data_dict\s*=\s*\{)",
            r"\1\n    'parquet_long': " + DATASET_CLASS_NAME + ",",
            s,
            count=1
        )

    FACTORY.write_text(s)

def patch_loader():
    s = LOADER.read_text()
    if f"class {DATASET_CLASS_NAME}" in s:
        return
    # Append at end
    LOADER.write_text(s.rstrip() + "\n\n" + DATASET_CODE.strip() + "\n")

def main():
    if not FACTORY.exists() or not LOADER.exists():
        raise SystemExit(f"Expected S4M official files at:\n  {FACTORY}\n  {LOADER}\nIs the repo cloned under models/s4m_model/s4m_official?")

    patch_factory()
    patch_loader()
    print("Patched S4M official repo with parquet_long dataset support.")
    print(f"- {FACTORY}")
    print(f"- {LOADER}")
    print("Next: run with --data parquet_long --root_path <dir> --data_path <file.parquet>")

if __name__ == "__main__":
    main()
