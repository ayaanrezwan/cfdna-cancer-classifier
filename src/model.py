import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED   = os.path.join(DATA_DIR, 'processed.pkl')
MODEL_PATH  = os.path.join(DATA_DIR, 'model.pkl')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── Load ──────────────────────────────────────────────────────────────────────
def load_processed():
    with open(PROCESSED, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    print(f"Loaded: X={X.shape}, y={y.shape}")
    print(f"Classes: {sorted(y.unique())}")
    return X, y

# ── Train ─────────────────────────────────────────────────────────────────────
def train(X: pd.DataFrame, y: pd.Series):
    """
    Train a Random Forest classifier with class_weight='balanced'.

    WHY BALANCED WEIGHTS:
    We have 100 tumor vs 32 normal samples — a ~3:1 imbalance.
    Without correction, the model learns to just predict 'Lung Cancer'
    for everything and gets 76% accuracy while being completely useless.
    'balanced' tells sklearn to weight each class inversely proportional
    to its frequency, so mistakes on the minority class (Healthy) are
    penalized more heavily during training.
    """

    # Encode string labels to integers (sklearn requirement)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Compute class weights explicitly so we can inspect them
    weights = compute_class_weight('balanced',
                                    classes=np.unique(y_enc),
                                    y=y_enc)
    print(f"Class weights: { dict(zip(le.classes_, weights.round(3))) }")

    # ── Cross-validation ──────────────────────────────────────────────────────
    # WHY CROSS-VALIDATION:
    # With only 132 samples, a single train/test split is unreliable —
    # you might get lucky or unlucky with which samples land where.
    # Stratified K-Fold splits the data into 5 folds, trains on 4, tests on 1,
    # repeats 5 times, and averages the results. 'Stratified' means each fold
    # preserves the class ratio (so you don't get a fold with zero normals).
    print("\nRunning 5-fold stratified cross-validation...")
    clf = RandomForestClassifier(
        n_estimators=200,       # number of trees — more = better, diminishing returns past ~200
        max_depth=None,         # let trees grow fully
        class_weight='balanced',
        random_state=42,
        n_jobs=-1               # use all CPU cores
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        clf, X, y_enc, cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1_weighted'],
        return_train_score=True
    )

    print("\n── Cross-validation results ──────────────────────────────")
    for metric in ['accuracy', 'roc_auc', 'f1_weighted']:
        test_scores  = scores[f'test_{metric}']
        train_scores = scores[f'train_{metric}']
        print(f"  {metric:15s}  "
              f"train={train_scores.mean():.3f}±{train_scores.std():.3f}  "
              f"test={test_scores.mean():.3f}±{test_scores.std():.3f}")

    # ── Final model — train on all data ───────────────────────────────────────
    # After cross-validation tells us the model works, we retrain on the
    # full dataset so the final model has seen as much data as possible.
    print("\nTraining final model on full dataset...")
    clf.fit(X, y_enc)

    return clf, le, scores

# ── Save ──────────────────────────────────────────────────────────────────────
def save_model(clf, le, scores):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {'model': clf, 'label_encoder': le, 'cv_scores': scores}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\nModel saved to {MODEL_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X, y   = load_processed()
    clf, le, scores = train(X, y)
    save_model(clf, le, scores)
    print("\n✓ Model training complete")

if __name__ == '__main__':
    main()