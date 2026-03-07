import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TopVarianceSelector

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
PROCESSED   = os.path.join(DATA_DIR, 'processed.pkl')
MODEL_PATH  = os.path.join(DATA_DIR, 'model.pkl')

# Tumor fractions to simulate — 1% to 100%
# In real plasma cfDNA, tumor fraction is typically 0.1–5%
TUMOR_FRACTIONS = [1.00, 0.50, 0.25, 0.10, 0.05, 0.01]
N_SIMULATIONS   = 50    # synthetic samples per fraction per class
RANDOM_STATE    = 42

# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    with open(PROCESSED, 'rb') as f:
        data = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    X   = data['X']
    y   = data['y']
    clf = model_data['model']
    le  = model_data['label_encoder']

    return X, y, clf, le

# ── Simulate plasma cfDNA at a given tumor fraction ───────────────────────────
def simulate_plasma(X_tumor: np.ndarray,
                    X_healthy: np.ndarray,
                    tumor_fraction: float,
                    n_simulations: int,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Simulate what a plasma cfDNA methylation profile looks like at a given
    tumor fraction.

    In plasma, cfDNA is a mixture:
      - (tumor_fraction)     of DNA shed by tumor cells
      - (1 - tumor_fraction) of DNA shed by healthy cells

    The measured beta value at each CpG site is approximately a weighted
    average of the tumor and healthy methylation levels:

        beta_plasma = tf * beta_tumor + (1 - tf) * beta_healthy

    This is a simplification — real cfDNA mixing is more complex due to
    different cell-type contributions and fragment-level effects — but it
    is the standard in silico mixing model used in the literature
    (Moss et al. 2018, Cristiano et al. 2019).

    Args:
        X_tumor:        numpy array of tumor beta values (n_tumor x n_features)
        X_healthy:      numpy array of healthy beta values (n_healthy x n_features)
        tumor_fraction: float between 0 and 1
        n_simulations:  how many synthetic plasma samples to generate
        rng:            numpy random generator for reproducibility

    Returns:
        Synthetic plasma beta matrix (n_simulations x n_features)
    """
    simulated = []
    for _ in range(n_simulations):
        # Randomly sample one tumor and one healthy profile
        t_idx = rng.integers(0, len(X_tumor))
        h_idx = rng.integers(0, len(X_healthy))

        # Linear mixing
        plasma = (tumor_fraction * X_tumor[t_idx] +
                  (1 - tumor_fraction) * X_healthy[h_idx])

        # Add small technical noise (array measurement noise ~0.01-0.02)
        noise  = rng.normal(0, 0.01, size=plasma.shape)
        plasma = np.clip(plasma + noise, 0, 1)

        simulated.append(plasma)

    return np.array(simulated)

# ── Run simulation across all tumor fractions ─────────────────────────────────
def run_simulation(X, y, clf, le):
    """
    For each tumor fraction, generate synthetic plasma samples and evaluate
    the classifier's AUC. This shows how model performance degrades as the
    tumor signal gets more diluted — directly modeling the cfDNA challenge.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(RANDOM_STATE)

    # Separate tumor and healthy profiles
    X_arr     = X.values
    y_arr     = y.values
    X_tumor   = X_arr[y_arr == 'Lung Cancer']
    X_healthy = X_arr[y_arr == 'Healthy']

    print(f"Tumor profiles:  {len(X_tumor)}")
    print(f"Healthy profiles: {len(X_healthy)}")
    print(f"\nSimulating {N_SIMULATIONS} plasma samples per tumor fraction...")
    print(f"{'Tumor Fraction':>16}  {'AUC':>6}  {'Accuracy':>9}  {'Interpretation'}")
    print("─" * 65)

    results = []
    for tf in TUMOR_FRACTIONS:
        # Generate synthetic plasma samples (all are "cancer positive")
        X_sim_tumor   = simulate_plasma(X_tumor, X_healthy, tf,
                                         N_SIMULATIONS, rng)
        # Generate pure healthy samples as negatives
        X_sim_healthy = simulate_plasma(X_healthy, X_healthy, 0.0,
                                         N_SIMULATIONS, rng)

        X_sim = np.vstack([X_sim_tumor, X_sim_healthy])
        y_sim = np.array([1] * N_SIMULATIONS + [0] * N_SIMULATIONS)

        # Predict
        y_prob = clf.predict_proba(X_sim)[:, 1]
        y_pred = clf.predict(X_sim)
        auc    = roc_auc_score(y_sim, y_prob)
        acc    = (y_pred == y_sim).mean()

        # Clinical interpretation
        if tf >= 0.25:
            interp = "Bulk tissue / late-stage"
        elif tf >= 0.10:
            interp = "Early-stage tumor shedding"
        elif tf >= 0.05:
            interp = "Low tumor burden"
        else:
            interp = "Typical plasma cfDNA range"

        print(f"{tf*100:>15.0f}%  {auc:>6.3f}  {acc:>9.3f}  {interp}")
        results.append({'tumor_fraction': tf, 'auc': auc, 'accuracy': acc})

    return pd.DataFrame(results)

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_degradation_curve(results: pd.DataFrame):
    """
    The key figure: AUC vs tumor fraction.
    This is the clinically relevant question — at what tumor fraction does
    the classifier break down? This is what liquid biopsy research is trying
    to push lower.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogx(results['tumor_fraction'] * 100,
                results['auc'],
                'o-', color='#2196F3', lw=2.5, markersize=8,
                label='XGBoost classifier (AUC)')

    ax.semilogx(results['tumor_fraction'] * 100,
                results['accuracy'],
                's--', color='#FF5722', lw=1.5, markersize=6,
                label='Accuracy')

    # Shade the clinically relevant cfDNA zone
    ax.axvspan(0.1, 5, alpha=0.12, color='orange',
               label='Typical plasma cfDNA range (0.1–5%)')

    ax.axhline(0.5, color='gray', lw=1, linestyle=':', label='Random classifier')

    ax.set_xlabel('Tumor Fraction in Simulated Plasma (%)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance vs Tumor Fraction\n'
                 'Simulating cfDNA Dilution in Plasma', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(['1%', '5%', '10%', '25%', '50%', '100%'])

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'tumor_fraction_simulation.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data and model...")
    X, y, clf, le = load_data()

    print("\nRunning in silico tumor fraction simulation...")
    print("=" * 65)
    results = run_simulation(X, y, clf, le)

    plot_degradation_curve(results)

    print("\n✓ Simulation complete")
    print("\nKey finding:")
    low_tf = results[results['tumor_fraction'] <= 0.05]
    print(f"  At ≤5% tumor fraction (clinical cfDNA range):")
    print(f"  Mean AUC = {low_tf['auc'].mean():.3f}")
    print(f"  This {'suggests the model may generalize to cfDNA' if low_tf['auc'].mean() > 0.7 else 'highlights the gap between bulk tissue and cfDNA classification'}")

if __name__ == '__main__':
    main()