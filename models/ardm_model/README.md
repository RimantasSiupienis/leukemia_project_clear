# ARDM – notes on how this is implemented

This folder contains my integration of **ARDM (Auto-Regressive Diffusion Models)** into my time-series evaluation pipeline.

So the actual ARDM model, training logic, diffusion steps, and sampling are taken
directly from the **official ARMD repository**.  
What I added is glue code so it works with my data format and evaluation setup.
---

## What happens when I run this

1. I start with a **long-format parquet dataset**
   (one row per patient per time step).

2. The data is split **by patient**, never by time.
   This avoids leakage and matches how all other models in the project are evaluated.

3. The long data is converted into **windowed numpy arrays**
   because ARDM expects windowed input, not long tables.

4. The **official ARDM training script** is called exactly as intended.

5. For test patients, I only use the **last available window per patient**
   and generate a future trajectory of length `pred_len`.

6. The predictions are converted back into **long format**
   so the evaluation scripts can directly compare real vs synthetic futures.

---

## The config file (`ardm_base.yaml`)

This is where all dataset-specific stuff lives.

Things here are mostly placeholders right now:
- `parquet_path`
- `id_col`
- `time_col`
- `target_cols`

Once I get the real dataset, **this is the main file I need to edit**. Everything else in the code assumes this config is correct.


## Dataset adapter (`ardm_adapter_dataset.py`)

This file is responsible for **everything data-related**.

### What it does in simple terms

- Loads the parquet file.
- Checks that required columns exist.
- Sorts data by `(patient, time)`.
- Fills missing values per patient (forward fill, then zero).
- Splits patients into train / val / test.
- Builds sliding windows using `build_windows`.
- Selects:
  - all windows for train and validation
  - **only the last window per test patient** for evaluation
- Saves:
  - `train.npy`, `val.npy`, `test.npy`
  - `real_future_long.parquet` (ground truth future)

### Why it’s written this way

ARDM works on windowed arrays, but my evaluation works on long tables.
So this file is basically the translator between those two worlds.

The most important thing is that **test windows are selected correctly**:
- one window per patient
- always the last one
- no future info leaks into inputs

If something breaks in evaluation, it’s probably here.

---

## Runner (`ardm_runner.py`)

This is the main script that ties everything together.

### Rough flow

- Load config
- Set random seeds
- Create output directory
- Call `prep_dataset(...)`
- Call official ARDM training script
- Call official ARDM evaluation script
- Load predictions
- Reshape predictions into `(patient, time, variables)`
- Align predictions with `real_future_long`
- Write all required output files

### Important helper functions

- `reshape_preds(...)`
  - ARDM outputs predictions in slightly different shapes depending on settings
  - this function just normalizes everything to one expected format

- `build_synth_future_long_from_real_index(...)`
  - makes sure synthetic output lines up **exactly**
    with the real future (same patient order, same timestamps)

This alignment step is critical for downstream evaluation.

---

## Official API wrapper (`official_api.py`)

This file does **almost nothing on purpose**.

It just:
- builds the command line calls for the official ARDM scripts
- runs them in the cloned repo directory
- returns paths to outputs

There is no dataset logic here.
If something breaks here, it probably means the **official repo interface changed**.

---

## Output files

Each run writes into:

output.dir / output.run_name /


Always contains:
- `config_resolved.yaml`
- `split_ids.json`
- `run_meta.json`
- `real_future_long.parquet`
- `synth_future_long.parquet`

These files follow the **same schema as all other models** in the project,
so evaluation scripts can run without knowing which model produced them.

---

## Reminders for myself:

- If results look weird:
  1. check dataset adapter
  2. check window selection
  3. check alignment of real vs synthetic futures
- All dataset-specific fixes should go into the config, not the code.