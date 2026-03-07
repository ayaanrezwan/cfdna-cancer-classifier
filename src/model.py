import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED  = os.path.join(DATA_DIR, 'processed.pkl')
MODEL_PATH = os.path.join(DATA_DIR, 'model.pkl')

TOP_N_PROBES  = 5000
HOLDOUT_SIZE  = 0.20   # 20% held out — never seen during training or CV
RANDOM_STATE  = 42

# ── Custom transformer: Top-N variance selector ───────────────────────────────
class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """
    Selects the top N features by variance, fit only on training data.
    Prevents data leakage into cross-validation test folds.
    """
    def __init__(self, top_n=5000):
        self.top_n = top_n

    def fit(self, X, y=None):
        variances     = np.var(X, axis=0)
        self.top_idx_ = np.argsort(variances)[::-1][:self.top_n]
        return self

    def transform(self, X, y=None):
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
def build_pipeline(scale_pos_weight: float):
    """
    Pipeline:
      Step 1 — TopVarianceSelector : 413k features → 5k, fit on train only
      Step 2 — XGBClassifier       : gradient boosted trees with regularization

    WHY XGBoost OVER RANDOM FOREST:
    XGBoost builds trees sequentially — each tree corrects the errors of
    the previous one. It adds L1 (reg_alpha) and L2 (reg_lambda) penalties
    directly into the optimization objective, explicitly discouraging
    memorization. Random Forest has no such regularization.

    learning_rate=0.05: each tree contributes only 5% of its prediction,
    forcing the model to build consensus slowly rather than overfit fast.

    max_depth=4: shallow trees can't memorize complex patterns.
    """
    return Pipeline([
        ('selector', TopVarianceSelector(top_n=TOP_N_PROBES)),
        ('clf', XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,          # each tree sees 80% of samples (reduces overfitting)
            colsample_bytree=0.8,   # each tree sees 80% of features (reduces overfitting)
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.1,          # L1 regularization
            reg_lambda=1.0,         # L2 regularization
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=-1
        ))
    ])

# ── Train ─────────────────────────────────────────────────────────────────────
def train(X: pd.DataFrame, y: pd.Series):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Holdout split ─────────────────────────────────────────────────────────
    # Split BEFORE any fitting — holdout set is never seen by the model
    # Stratified ensures class ratio is preserved in both splits
    X_arr = X.values
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X_arr, y_enc,
        test_size=HOLDOUT_SIZE,
        stratify=y_enc,
        random_state=RANDOM_STATE
    )
    print(f"\nHoldout split:")
    print(f"  Development set : {X_dev.shape[0]} samples")
    print(f"  Holdout set     : {X_holdout.shape[0]} samples (never used in training)")

    # Class balance in dev set
    n_neg = (y_dev == 0).sum()
    n_pos = (y_dev == 1).sum()
    spw   = n_neg / n_pos
    print(f"  scale_pos_weight: {spw:.2f}  ({n_neg} healthy / {n_pos} tumor)")

    pipeline = build_pipeline(scale_pos_weight=spw)

    # ── Cross-validation on dev set only ─────────────────────────────────────
    print(f"\nRunning 5-fold CV on development set ({X_dev.shape[0]} samples)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline, X_dev, y_dev, cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1_weighted'],
        return_train_score=True,
        verbose=0
    )

    print("\n── Cross-validation results (dev set) ────────────────────────────")
    for metric in ['accuracy', 'roc_auc', 'f1_weighted']:
        test_s  = scores[f'test_{metric}']
        train_s = scores[f'train_{metric}']
        print(f"  {metric:15s}  "
              f"train={train_s.mean():.3f}±{train_s.std():.3f}  "
              f"test={test_s.mean():.3f}±{test_s.std():.3f}")

    # ── Final model: train on full dev set ────────────────────────────────────
    print("\nTraining final model on full development set...")
    pipeline.fit(X_dev, y_dev)

    # ── Holdout evaluation ────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    y_holdout_pred  = pipeline.predict(X_holdout)
    y_holdout_prob  = pipeline.predict_proba(X_holdout)[:, 1]
    holdout_acc     = accuracy_score(y_holdout, y_holdout_pred)
    holdout_auc     = roc_auc_score(y_holdout, y_holdout_prob)

    print("\n── Holdout set results (truly unseen data) ───────────────────────")
    print(f"  Accuracy : {holdout_acc:.3f}")
    print(f"  ROC-AUC  : {holdout_auc:.3f}")
    print("\n  Classification report:")
    print(classification_report(y_holdout, y_holdout_pred,
                                 target_names=le.classes_))

    # Recover selected probe names from final fit
    selected_idx    = pipeline.named_steps['selector'].top_idx_
    selected_probes = X.columns[selected_idx].tolist()

    return pipeline, le, scores, selected_probes, {
        'X_holdout': X_holdout,
        'y_holdout': y_holdout,
        'y_holdout_pred': y_holdout_pred,
        'y_holdout_prob': y_holdout_prob,
        'accuracy': holdout_acc,
        'roc_auc':  holdout_auc
    }

# ── Save ──────────────────────────────────────────────────────────────────────
def save_model(pipeline, le, scores, selected_probes, holdout_results):
    payload = {
        'model':           pipeline,
        'label_encoder':   le,
        'cv_scores':       scores,
        'selected_probes': selected_probes,
        'holdout_results': holdout_results
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\nModel saved → {MODEL_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X, y = load_processed()
    pipeline, le, scores, probes, holdout = train(X, y)
    save_model(pipeline, le, scores, probes, holdout)
    print("\n✓ Model training complete")

if __name__ == '__main__':
    main()