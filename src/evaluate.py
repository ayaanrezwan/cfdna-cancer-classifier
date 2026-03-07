import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import TopVarianceSelector

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
PROCESSED   = os.path.join(DATA_DIR, 'processed.pkl')
MODEL_PATH  = os.path.join(DATA_DIR, 'model.pkl')

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
def load_all():
    with open(PROCESSED, 'rb') as f:
        data = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    X   = data['X']
    y   = data['y']
    clf = model_data['model']
    le  = model_data['label_encoder']
    selected_probes = model_data['selected_probes']
    y_enc = le.transform(y)

    # Convert to numpy array — pipeline uses integer indexing internally
    X_arr = X.values

    print(f"Loaded: X={X_arr.shape}, y={y.shape}")
    return X_arr, y, y_enc, clf, le, selected_probes

# ── Plot 1: Confusion Matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(clf, X, y_enc, le):
    """
    A confusion matrix shows exactly where your model makes mistakes.
    Rows = actual class, Columns = predicted class.
    The diagonal = correct predictions.
    Off-diagonal = errors (false positives / false negatives).

    In cancer diagnostics, false negatives (missed cancers) are far more
    dangerous than false positives (unnecessary follow-up tests).
    This plot makes that tradeoff visible.
    """
    print("Plotting confusion matrix...")

    # Use cross-val predictions so we're evaluating on held-out data
    from sklearn.model_selection import cross_val_predict
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred  = cross_val_predict(clf, X, y_enc, cv=cv)
    cm      = confusion_matrix(y_enc, y_pred)
    labels  = le.classes_

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (5-Fold Cross-Validation)', fontsize=13)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")
    print("\nClassification report:")
    print(classification_report(y_enc, y_pred, target_names=labels))

# ── Plot 2: ROC Curve ─────────────────────────────────────────────────────────
def plot_roc_curve(clf, X, y_enc, le):
    """
    The ROC (Receiver Operating Characteristic) curve plots the tradeoff
    between sensitivity (catching real cancers) and specificity (avoiding
    false alarms) at every possible classification threshold.

    AUC (Area Under the Curve) summarizes this in one number:
      0.5 = random guessing
      1.0 = perfect classifier
      >0.9 = very strong performance

    This is the standard evaluation metric in clinical ML papers.
    """
    print("\nPlotting ROC curve...")
    from sklearn.model_selection import cross_val_predict
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob     = cross_val_predict(clf, X, y_enc, cv=cv, method='predict_proba')
    fpr, tpr, _ = roc_curve(y_enc, y_prob[:, 1])
    roc_auc    = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#2196F3', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
            label='Random classifier')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve — Lung Cancer vs Healthy', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}  (AUC = {roc_auc:.3f})")

# ── Plot 3: Learning Curve ────────────────────────────────────────────────────
def plot_learning_curve(clf, X, y_enc):
    """
    A learning curve plots model performance as a function of training size.

    If the model is genuinely learning:
      - Training score stays high as data increases
      - Validation score improves as data increases
      - The two lines converge

    If the model is memorizing (overfitting):
      - Training score = 1.0 at all sizes
      - Validation score stays low or erratic
      - Large gap between the two lines

    This is how we diagnose our suspiciously perfect CV scores.
    """
    print("\nPlotting learning curve...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y_enc,
        cv=cv,
        scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Training AUC')
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color='#2196F3')
    ax.plot(train_sizes, val_mean, 'o-', color='#FF5722', label='Validation AUC')
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color='#FF5722')
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('ROC-AUC Score', fontsize=12)
    ax.set_title('Learning Curve — Does the Model Generalize?', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim([0.4, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'learning_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

# ── Plot 4: Top Feature Importances ──────────────────────────────────────────
def plot_feature_importances(rf, selected_probes, le, top_n=20):
    print(f"\nPlotting top {top_n} feature importances...")
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({
        'probe':      selected_probes,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(feat_df['probe'][::-1],
            feat_df['importance'][::-1],
            color='#4CAF50', edgecolor='white')
    ax.set_xlabel('Mean Decrease in Impurity (Feature Importance)', fontsize=11)
    ax.set_title(f'Top {top_n} Most Predictive CpG Probes', fontsize=13)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'feature_importances.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")
    print(f"\n  Top 5 predictive probes:")
    print(feat_df.head(5).to_string(index=False))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X, y, y_enc, clf, le, selected_probes = load_all()

    plot_confusion_matrix(clf, X, y_enc, le)
    plot_roc_curve(clf, X, y_enc, le)
    plot_learning_curve(clf, X, y_enc)

    rf = clf.named_steps['clf']
    plot_feature_importances(rf, selected_probes, le)

    print("\n✓ All plots saved to results/figures/")

if __name__ == '__main__':
    main()