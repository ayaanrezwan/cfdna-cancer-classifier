import urllib.request
import gzip
import io
import pandas as pd
import numpy as np
import os
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
URL       = 'https://gdc.xenahubs.net/download/TCGA-LUAD.methylation450.tsv.gz'
OUT_DIR   = os.path.dirname(__file__)
BETA_PATH = os.path.join(OUT_DIR, 'beta_matrix.pkl')
META_PATH = os.path.join(OUT_DIR, 'metadata.csv')

MAX_TUMOR  = 100   # how many tumor samples to keep
MAX_NORMAL = 32    # keep all normals (only 32 exist)
CHUNK_SIZE = 65536 # bytes to read at a time while streaming

# ── Step 1: Stream header and select sample columns ───────────────────────────
def get_sample_columns(url: str):
    """
    Stream just enough of the file to read the header row.
    Returns lists of tumor and normal sample IDs to keep.
    """
    print("Streaming header to identify samples...")
    req = urllib.request.urlopen(url, timeout=60)

    buf = b''
    while len(buf) < 500_000:
        chunk = req.read(CHUNK_SIZE)
        if not chunk:
            break
        buf += chunk
    req.close()

    with gzip.open(io.BytesIO(buf), 'rt') as f:
        header = f.readline().strip().split('\t')

    # Sample type code is at characters 13-15 of the TCGA barcode
    # '01' = primary solid tumor, '11' = solid tissue normal
    tumor_cols  = [s for s in header[1:] if len(s) > 13 and s[13:15] == '01']
    normal_cols = [s for s in header[1:] if len(s) > 13 and s[13:15] == '11']

    print(f"  Found {len(tumor_cols)} tumor samples, {len(normal_cols)} normal samples")

    # Subsample tumor to keep things manageable
    np.random.seed(42)
    tumor_keep  = list(np.random.choice(tumor_cols,
                                         min(MAX_TUMOR, len(tumor_cols)),
                                         replace=False))
    normal_keep = normal_cols[:MAX_NORMAL]

    selected = set(tumor_keep + normal_keep)
    print(f"  Keeping {len(tumor_keep)} tumor + {len(normal_keep)} normal = "
          f"{len(selected)} total samples")
    return tumor_keep, normal_keep, selected, header[0]

# ── Step 2: Stream full file, keeping only selected columns ───────────────────
def stream_beta_matrix(url: str, selected_cols: set, probe_col: str):
    """
    Stream the full gzipped TSV row by row.
    For each row (CpG probe), keep only the selected sample columns.
    This avoids loading the full 1.7GB file into memory.
    """
    print(f"\nStreaming full matrix (this will take 5-15 minutes)...")
    print("  Downloading and filtering on the fly — do not interrupt.")

    req = urllib.request.urlopen(url, timeout=300)

    # We need to buffer the entire download to decompress it
    # but we process row by row to keep RAM usage low
    rows      = {}
    col_index = None
    line_buf  = b''
    total_rows = 0
    dot_every  = 50_000

    with gzip.GzipFile(fileobj=req) as gz:
        for raw_line in gz:
            line = raw_line.decode('utf-8').rstrip('\n')

            if col_index is None:
                # First line is the header
                cols      = line.split('\t')
                col_index = {name: i for i, name in enumerate(cols)}
                keep_idx  = [col_index[c] for c in selected_cols if c in col_index]
                keep_names = [cols[i] for i in keep_idx]
                print(f"  Confirmed {len(keep_idx)} columns to extract")
                continue

            # Every subsequent line is one CpG probe
            parts    = line.split('\t')
            probe_id = parts[0]
            values   = []
            for i in keep_idx:
                val = parts[i] if i < len(parts) else 'NA'
                values.append(float(val) if val not in ('', 'NA', 'nan') else np.nan)

            rows[probe_id] = values
            total_rows += 1

            if total_rows % dot_every == 0:
                print(f"  Processed {total_rows:,} probes...")

    req.close()

    beta = pd.DataFrame.from_dict(rows, orient='index', columns=keep_names)
    print(f"\n  Beta matrix shape: {beta.shape}  (probes × samples)")
    return beta

# ── Step 3: Build metadata DataFrame from sample IDs ─────────────────────────
def build_metadata(tumor_cols: list, normal_cols: list) -> pd.DataFrame:
    """
    Since we decoded labels from the barcode, we can build metadata directly.
    No external phenotype file needed.
    """
    records = []
    for s in tumor_cols:
        records.append({'sample_id': s, 'label': 'Lung Cancer'})
    for s in normal_cols:
        records.append({'sample_id': s, 'label': 'Healthy'})

    df = pd.DataFrame(records).set_index('sample_id')
    print(f"\n  Label distribution:")
    print(df['label'].value_counts())
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Get sample columns from header
    tumor_keep, normal_keep, selected, probe_col = get_sample_columns(URL)

    # 2. Stream and filter the full matrix
    beta = stream_beta_matrix(URL, selected, probe_col)

    # 3. Build metadata from barcodes
    meta = build_metadata(tumor_keep, normal_keep)

    # 4. Align — ensure beta columns match metadata index
    common = beta.columns.intersection(meta.index)
    beta   = beta[common]
    meta   = meta.loc[common]
    print(f"\n  Aligned samples: {len(common)}")

    # 5. Save
    print(f"\nSaving beta matrix to {BETA_PATH}")
    with open(BETA_PATH, 'wb') as f:
        pickle.dump(beta, f)

    print(f"Saving metadata to {META_PATH}")
    meta.to_csv(META_PATH)

    print("\n✓ Done. Summary:")
    print(f"  Beta matrix : {beta.shape}  (probes × samples)")
    print(f"  Metadata    : {meta.shape}")
    print(f"  Classes     : {sorted(meta['label'].unique())}")

if __name__ == '__main__':
    main()