# iTransformer

## What the current code does
When the iTransformer runner is executed:

1. Loads a long-format parquet file with patient trajectories.
2. Optionally joins static patient features from CTGAN.
3. Splits patients into train / val / test at patient level.
4. Builds sliding windows:
    history window (`context_len`)
    future window (`horizon`)
5. Trains iTransformer on training windows.
6. Selects best model using validation loss.
7. For each test patient:
    takes the last available history window
    predicts the next `horizon` time steps
8. Saves outputs in standardized formats for evaluation.

## Input data format
A long-format parquet file with one row per patient per time step.
Each row must contain:
- patient identifier
- time column (numeric index or datetime)
- multiple numeric target vars
Honestly will probably change this.