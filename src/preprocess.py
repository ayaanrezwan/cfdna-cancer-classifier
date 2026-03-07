import pandas as pd
import numpy as np
import pickle
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__), '..', 'data')
BETA_PATH       = os.path.join(DATA_DIR, 'beta_matrix.pkl')
META_PATH       = os.path.join(DATA_DIR, 'metadata.csv')
PROCESSED_PATH  = os.path.join(DATA_DIR, 'processed.pkl')

TOP_N_PROBES    = 5000   # keep the N most variable CpG probes
NA_THRESHOLD    = 0.2    # drop probes missing in more than 20% of samples

# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    print("Loading beta matrix...")
    with open(BETA_PATH, 'rb') as f:
        beta = pickle.load(f)                      # shape: probes × samples
    meta = pd.read_csv(META_PATH, index_col=0)
    print(f"  Beta : {beta.shape}")
    print(f"  Meta : {meta.shape}")
    return beta, meta

# ── Step 1: Drop high-missingness probes ─────────────────────────────────────
def filter_missing(beta: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drop any probe where more than `threshold` fraction of samples are NaN.
    Example: threshold=0.2 drops probes missing in >20% of samples.
    """
    missing_rate = beta.isna().mean(axis=1)        # fraction missing per probe
    keep         = missing_rate <= threshold
    beta_filtered = beta[keep]
    print(f"\nMissing value filter:")
    print(f"  Probes before : {beta.shape[0]:,}")
    print(f"  Probes after  : {beta_filtered.shape[0]:,}  "
          f"(dropped {(~keep).sum():,})")
    return beta_filtered

# ── Step 2: Impute remaining NaNs ────────────────────────────────────────────
def impute(beta: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaNs with per-probe mean using numpy for speed.
    """
    n_missing = beta.isna().sum().sum()
    print(f"\nImputation:")
    print(f"  Remaining NaNs before imputation: {n_missing:,}")

    arr = beta.values                              # convert to raw numpy array
    row_means = np.nanmean(arr, axis=1)            # mean per probe, ignoring NaN
    inds = np.where(np.isnan(arr))                 # find NaN positions
    arr[inds] = row_means[inds[0]]                 # fill with row mean

    beta_imputed = pd.DataFrame(arr, index=beta.index, columns=beta.columns)
    print(f"  NaNs after imputation: {beta_imputed.isna().sum().sum()}")
    return beta_imputed

# ── Step 3: Variance filtering ────────────────────────────────────────────────
def select_top_variable(beta: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Keep only the `top_n` probes with the highest variance across samples.
    High variance = this probe's methylation differs a lot between samples
                  = potentially useful for distinguishing cancer from healthy.
    Low variance  = methylation is similar in everyone
                  = not useful for classification.
    """
    print(f"\nVariance filtering:")
    print(f"  Computing variance across {beta.shape[0]:,} probes...")
    probe_var = beta.var(axis=1)                   # variance per probe
    top_probes = probe_var.nlargest(top_n).index
    beta_filtered = beta.loc[top_probes]
    print(f"  Kept top {top_n:,} most variable probes")
    print(f"  Variance range: {probe_var[top_probes].min():.4f} "
          f"– {probe_var[top_probes].max():.4f}")
    return beta_filtered

# ── Step 4: Transpose ─────────────────────────────────────────────────────────
def transpose_to_samples(beta: pd.DataFrame) -> pd.DataFrame:
    """
    Flip the matrix so rows = samples, columns = CpG probes.
    This is the format every sklearn model expects:
      X.shape = (n_samples, n_features)
    """
    return beta.T                                  # shape: samples × probes

# ── Step 5: Align with metadata ───────────────────────────────────────────────
def align(X: pd.DataFrame, meta: pd.DataFrame):
    """
    Make sure X and meta have the same samples in the same order.
    """
    common = X.index.intersection(meta.index)
    X    = X.loc[common]
    meta = meta.loc[common]
    print(f"\nAlignment:")
    print(f"  Common samples: {len(common)}")
    return X, meta

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    beta, meta = load_data()

    # 2. Filter missing
    beta = filter_missing(beta, NA_THRESHOLD)

    # 3. Impute
    beta = impute(beta)

    # 4. Variance filter
    beta = select_top_variable(beta, TOP_N_PROBES)

    # 5. Transpose to samples × features
    X = transpose_to_samples(beta)

    # 6. Align
    X, meta = align(X, meta)

    # 7. Extract labels
    y = meta['label']

    # 8. Save
    print(f"\nSaving processed data to {PROCESSED_PATH}")
    with open(PROCESSED_PATH, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)

    print("\n✓ Done. Summary:")
    print(f"  X shape : {X.shape}  (samples × features)")
    print(f"  y shape : {y.shape}")
    print(f"  Classes : {sorted(y.unique())}")
    print(f"  Label counts:\n{y.value_counts()}")

if __name__ == '__main__':
    main()