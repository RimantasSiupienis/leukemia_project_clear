# CTGAN

This CTGAN module is used only to generate a **static, one-row-per-patient** table.  
Not a time-series model. Nothing here models temporal dynamics.


The purpose of this model is:
- to generate **static patient covariates** that are later consumed by longitudinal models.

## Role in the pipeline
- Input: real patient data (long format)
- Output: static baseline (one row per patient)
- Usage:
  - evaluated directly as a distributional baseline
  - used as nput features for downstream time-series models

Patient identity is preserved end-to-end (basically, no mixing across patients).

## What the current code does

When `run_ctgan.py` is executed:
1. Loads a long-format dataset from a fixed path.
2. Collapses long data into a **static snapshot per patient**.
3. Drops identifier columns from the feature matrix.
4. Splits data at **patient level** (no test split).
5. Trains SDV CTGAN on the static training table.
6. Samples a synthetic static table with the **same number of rows** as the training cohort.
7. Re-attaches patient IDs so downstream models can join by `patient_id`.
8. Saves both real and synthetic static tables and configuration metadata.


## Output files
All outputs are written to:
  results/ctgan/


Files produced:
- `real_static.parquet`  
  Real static baseline, the training cohort only.

- `ctgan_static.parquet`  
  Synthetic static baseline, one row per patient.

- `ctgan_config.json`  
  Snapshot of paths, columns, and CTGAN configuration used for the run.

## run_ctgan.py

- `load_and_prepare_data(...)`  
  Loads the dataset and applies the long to static transformation.

- `make_static_snapshot(...)`  
  Collapses multiple rows per patient into a single row.

- `train_ctgan(...)`  
  Fits SDV CTGAN on the static feature table.

- `sample_ctgan(...)`  
  Generates synthetic rows from the trained CTGAN.

## Assumptions made:
- Data is patient-centric, stuff is identifiable by patient ID column.
- Static features are sufficient for baseline comparison.
- Synthetic data must preserves patient count and identity alignment for a good sim.