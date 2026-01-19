# iTransformer
Inverted Transformers Are Effective for Time Series Forecasting

## What happens when I run this

1. I start with a **long-format parquet dataset**
   (one row per patient per time step).

2. Patients are split into **train / val / test**
   at the **patient level** (never by time, so no leakage).

3. If enabled, a **static baseline table** (from CTGAN) is merged:
   - one row per patient
   - repeated across all time steps
   - used as extra input features

4. The iTransformer model is trained on training patients only.

5. For each test patient:
    takea the last `context_len` observations
    forecasts the next `horizon` steps

6. Predictions are reshaped into a long table
    old idea, may change in future

## Config file (`configs/itransformer_base.yaml`)
All the dataset-specific stuff
Most vals are placeholders until I have the real dataset.
Things I will definitely need to change later:
- `data.long_path`
- `data.id_col`
- `data.time_col`
- `data.target_cols`

Things that may need tuning:
- `task.context_len`
- `task.horizon`
- train / val / test split ratios

## Main script (`src/model.py`)

1. Loads YAML config
2. Sets random seeds
3. Loads parquet data
4. Checks required columns
5. Merges static baseline (if enabled)
6. Splits patients
7. Initializes iTransformer
8. Trains model
9. Runs forecasting on test patients
10. Saves outputs + metadata

## Helper functions

### `patient_split(...)`
- Splits **patients**, not rows
- Prevents leakage
- Saves `split_ids.json`

If evaluation looks off check this first.

### `load_static_table(...)`
- Loads CTGAN output:
  - `real_static.parquet` or
  - `ctgan_static.parquet`
- Enforces one row per patient

### `attach_static_as_time_covariates(...)`
- Merges static features onto long data
- Repeats them over time
- Does not use future information

## Output files

Each run writes into:

----add later-----

Always contains:
- `config_resolved.yaml`  exact config used
- `split_ids.json`  patient-level split
- `run_meta.json`  metadata 
- `preds_long.parquet`  model predictions

### `preds_long.parquet` format

[id_col, target, horizon_step, pred]