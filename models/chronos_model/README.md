# Chronos
This folder contains my integration of Chronos into the shared
evaluation pipeline for longitudinal leukemia time-series forecasting.

The actual model is not modified.  
Chronos is used via `sktime.forecasting.chronos.ChronosForecaster`.

## What happens when I run this
1. Loads a **long-format parquet dataset**
   (one row per patient per time step).

2. Splits patients into **train and test** at the patient level  
   (never by time therefor no leakage).

3. For each **test patient only**:
   - takes the **last `context_len` observations**
   - forecasts the next **`horizon` steps**
   - does this **independently for each target variable**

4. Collects:
   - the real future (`real_future_long.parquet`)
   - the Chronos forecast (`synth_future_long.parquet`)

5. Saves all standard pipeline artifacts so evaluation scripts can run without knowing
   that Chronos produced them.


## Files

### `configs/chronos_base.yaml`
This is where all dataset-specific configuration is.
It controls:
- where the data is loaded from
- which columns are used
- how long the context and forecast horizon are
- how patients are split
- which Chronos model is used

If something is wrong with the data or evaluation, **this file is probably the reason**.

### `src/model.py`
Main runner script.

High-level flow inside `main()`:
1. Loads config
2. Sets random seeds
3. Loads parquet data
4. Validates data format
5. Splits patients
6. Initializes chronos forecaster
7. Runs last-window-per-patient forecasting
8. Writes standardized output files


## Important functions

- `repo_root()`
  - Finds the project root.

- `import_shared_utils()`
  - Imports shared helpers from s4m:
    - `set_seed`
    - `load_parquet`
    - `df_checks`
    - `ensure_dir`
    - `save_json`

- `load_cfg(path)`
  - Loads the YAML config file.

- `resolve_path(p)`
  - Resolves relative paths against the repo root.

- `split_patients(...)`
  - Performs patient-level splitting
  - Supports:
    - `patient_random` (seeded)
    - `patient_hash` (stable hash + salting)

- `as_float_1d(x)`
  - Normalizes chronos outputs to a 1D float array.

- `main(cfg_path)`
  - Orchestrates the entire run, this is what gets executed

## Outputs

Always contains:
- `config_resolved.yaml` – snapshot of the config used
- `split_ids.json` – train/test patient IDs and skipped ones
- `run_meta.json` – metadata (counts, seeds, timing)
- `real_future_long.parquet`
- `synth_future_long.parquet`

Optional:
- `preds_long.parquet` but only if `save_preds_long: true`

Both `real_future_long` and `synth_future_long` have schema:
  [id_col, time_col] + target_cols
  they are aligned so evaluation scripts can compare them directly.

## Notes (things to change once real data is available)

Before running on the real dataset, I'll need to update the YAML config:

- `data.long_path`
  - path to the real parquet file
- `data.id_col`
  - patient identifier column name
- `data.time_col`
  - time index column name
- `data.target_cols`
  - actual variables to forecast

Probs also needs adjustment:
- `task.context_len`
- `task.horizon`
- `task.min_history`
In theory no code changes should be needed if the dataset is the expected long format.