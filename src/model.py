import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED  = os.path.join(DATA_DIR, 'processed.pkl')
MODEL_PATH = os.path.join(DATA_DIR, 'model.pkl')

TOP_N_PROBES = 5000

# ── Custom transformer: Top-N variance selector ───────────────────────────────
class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """
    Selects the top N features by variance.

    Why build a custom transformer instead of using sklearn's VarianceThreshold?
    VarianceThreshold drops features BELOW a threshold — it doesn't let you
    say 'keep exactly the top N'. We need exactly top-N so the feature count
    is deterministic across folds (required for the Random Forest to work,
    since each fold must produce the same number of input features... 
    actually each fold is independent, but consistent top-N makes the
    final model's feature list meaningful).

    By inheriting BaseEstimator and TransformerMixin and implementing
    fit() and transform() separately, sklearn's Pipeline knows to:
      - call fit()       only on training data
      - call transform() on both train and test data using fitted state
    This is exactly what prevents leakage.
    """
    def __init__(self, top_n=5000):
        self.top_n = top_n

    def fit(self, X, y=None):
        # Compute variance on TRAINING data only
        # Store the indices of top-N probes — these become the features
        variances       = np.var(X, axis=0)
        self.top_idx_   = np.argsort(variances)[::-1][:self.top_n]
        return self                                    # always return self

    def transform(self, X, y=None):
        # Apply the probe indices fitted on training data to any split
        return X[:, self.top_idx_]

    def get_feature_names_out(self, feature_names_in=None):
        if feature_names_in is not None:
            return np.array(feature_names_in)[self.top_idx_]
        return self.top_idx_

# ── Load ──────────────────────────────────────────────────────────────────────
def load_processed():
    with open(PROCESSED, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    print(f"Loaded: X={X.shape}, y={y.shape}")
    return X, y

# ── Build pipeline ────────────────────────────────────────────────────────────
def build_pipeline():
    """
    A Pipeline chains steps sequentially. Each step is a (name, estimator) tuple.
    During cross-validation, sklearn calls fit_transform() on training data
    and transform() on test data for every step except the last, which gets
    fit() and predict().

    Step 1 — TopVarianceSelector: reduces 413k features → 5k, fit on train only
    Step 2 — RandomForestClassifier: trains on the 5k selected features
    """
    return Pipeline([
        ('selector', TopVarianceSelector(top_n=TOP_N_PROBES)),
        ('clf',      RandomForestClassifier(
                         n_estimators=200,
                         class_weight='balanced',
                         random_state=42,
                         n_jobs=-1
                     ))
    ])

# ── Train + evaluate ──────────────────────────────────────────────────────────
def train(X: pd.DataFrame, y: pd.Series):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Convert to numpy — Pipeline works faster with arrays than DataFrames
    X_arr = X.values

    pipeline = build_pipeline()
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nRunning 5-fold cross-validation with leak-free pipeline...")
    print(f"  Each fold: fit variance selector on train, transform test independently")
    print(f"  This will take a few minutes...\n")

    scores = cross_validate(
        pipeline, X_arr, y_enc, cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1_weighted'],
        return_train_score=True,
        verbose=1
    )

    print("\n── Cross-validation results ──────────────────────────────")
    for metric in ['accuracy', 'roc_auc', 'f1_weighted']:
        test_scores  = scores[f'test_{metric}']
        train_scores = scores[f'train_{metric}']
        print(f"  {metric:15s}  "
              f"train={train_scores.mean():.3f}±{train_scores.std():.3f}  "
              f"test={test_scores.mean():.3f}±{test_scores.std():.3f}")

    # Retrain on full data for final model + feature name recovery
    print("\nTraining final model on full dataset...")
    pipeline.fit(X_arr, y_enc)

    # Recover which probe names were selected in the final fit
    selected_idx   = pipeline.named_steps['selector'].top_idx_
    selected_probes = X.columns[selected_idx].tolist()

    return pipeline, le, scores, selected_probes

# ── Save ──────────────────────────────────────────────────────────────────────
def save_model(pipeline, le, scores, selected_probes):
    payload = {
        'model':           pipeline,
        'label_encoder':   le,
        'cv_scores':       scores,
        'selected_probes': selected_probes
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Model saved → {MODEL_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X, y                             = load_processed()
    pipeline, le, scores, probes     = train(X, y)
    save_model(pipeline, le, scores, probes)
    print("\n✓ Model training complete")

if __name__ == '__main__':
    main()